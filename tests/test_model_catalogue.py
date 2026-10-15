import http.client
from decimal import Decimal

import pytest
from pydantic import ValidationError
from test_generation_units import unit_context
from ui_fixtures import synthetic_catalogue

from finrag.config import FinragError
from finrag.generation import citation_units
from finrag.generation.model_catalogue import (
    GenerationSettings,
    budget_error,
    compatibility_errors,
    estimate_cost,
    fetch_catalogue,
    parse_catalogue,
    price,
)
from finrag.ui.model_controls import price_label, read_latest_catalogue, store_catalogue


def test_prices_units_missing_and_invalid_values():
    assert price_label(price("0.00000068")) == "0.68 USD / million de tokens"
    for value in (None, "", "NaN", "Infinity", "-1", True, {}):
        assert price(value) is None
    assert price("0") == 0
    assert price_label(None) == "non disponible"
    raw = {"data": [{"id": "fictional/missing"}]}
    model = parse_catalogue(raw, "2026-10-15").models[0]
    assert model.context_length is None and model.input_price is None
    assert model.max_output_tokens is None and model.request_price is None
    assert compatibility_errors(model, 1024)


def test_compatibility_and_documented_reasoning_per_model():
    model = synthetic_catalogue().models[0]
    assert compatibility_errors(model, 1024) == []
    assert model.efforts == ["none", "high"]
    assert model.model_copy(update={"reasoning": {"supported_efforts": None}}).efforts
    mandatory = model.model_copy(
        update={
            "reasoning": {
                "supported_efforts": ["none", "high"],
                "mandatory": True,
            }
        }
    )
    assert mandatory.efforts == ["high"]
    assert not model.model_copy(update={"reasoning": {}}).efforts
    for changes in (
        {"parameters": ["response_format", "max_tokens"]},
        {"model_id": "openrouter/auto"},
        {"context_length": 512},
        {"max_output_tokens": 512},
        {"output_modalities": ["image"]},
    ):
        assert compatibility_errors(model.model_copy(update=changes), 1024)
    settings = GenerationSettings(
        model=model.model_id,
        reasoning_effort="high",
        supported_reasoning_efforts=model.efforts,
        max_completion_tokens=2048,
    )
    payload = citation_units.request_payload(unit_context(), settings)
    assert payload["reasoning"] == {"effort": "high"} and payload["max_completion_tokens"] == 2048
    with pytest.raises(ValidationError):
        GenerationSettings(model=model.model_id, reasoning_effort="none")
    omitted = GenerationSettings(model=model.model_id)
    assert "reasoning" not in citation_units.request_payload(unit_context(), omitted)
    assert citation_units.request_identity(
        unit_context(), omitted
    ) != citation_units.request_identity(
        unit_context(),
        settings,
    )


def test_budget_known_and_unknown_without_character_token_conversion():
    model = synthetic_catalogue().models[0]
    known = estimate_cost(model, 1024, input_tokens=100)
    assert known.total_usd == Decimal("0.002148")
    assert budget_error(known, Decimal("0.002"), True)
    assert budget_error(known, Decimal("0.003"), False) is None
    unknown = estimate_cost(model, 1024)
    assert unknown.total_usd is None and unknown.input_tokens is None
    assert budget_error(unknown, Decimal("0.05"), False)
    assert budget_error(unknown, Decimal("0.05"), True) is None
    assert budget_error(unknown, Decimal("0.001"), True)
    for budget in (Decimal("0"), Decimal("NaN"), Decimal("-1")):
        assert budget_error(unknown, budget, True)
    assert (
        estimate_cost(model.model_copy(update={"request_price": None}), 1024, 100).total_usd is None
    )
    assert estimate_cost(model.model_copy(update={"output_price": None}), 1024).total_usd is None


def test_catalogue_get_never_sends_key_or_inference(monkeypatch):
    calls = []

    class Response:
        status = 200

        def read(self, size):
            assert size == 8 * 1024 * 1024 + 1
            return b'{"data":[{"id":"fictional/model"}]}'

    class Connection:
        def request(self, *args, **kwargs):
            calls.append((args, kwargs))

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-be-sent")
    monkeypatch.setattr(http.client, "HTTPSConnection", lambda *args, **kwargs: Connection())
    catalogue = fetch_catalogue()
    assert len(catalogue.models) == 1
    assert calls == [(("GET", "/api/v1/models"), {"headers": {"Accept": "application/json"}})]


def test_snapshot_roundtrip_does_not_replace_prior_file(tmp_path):
    first = store_catalogue(tmp_path, synthetic_catalogue())
    content = first.read_bytes()
    second = store_catalogue(tmp_path, synthetic_catalogue())
    assert second != first and first.read_bytes() == content
    assert read_latest_catalogue(tmp_path) == synthetic_catalogue()
    with pytest.raises(FinragError):
        parse_catalogue({}, "today")
