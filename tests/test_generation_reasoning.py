import http.client

import pytest
from pydantic import ValidationError
from test_generation import SimulatedProvider, fixture_context, raw_response

from finrag.generation.answers import (
    MISTRAL_LARGE_4,
    PROMPT_VERSION,
    AnswerSettings,
    request_identity,
    request_payload,
)
from finrag.generation.cache import generate
from finrag.generation.openrouter import GenerationFailure
from finrag.ingestion.hashing import hash_bytes, hash_file
from finrag.ingestion.manifest import json_bytes, read_json


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real API forbidden")

    from ui_fixtures import synthetic_catalogue

    from finrag.ui import model_controls as controls

    monkeypatch.setattr(controls, "read_latest_catalogue", lambda _: synthetic_catalogue())
    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fictional-private-key")


def test_none_changes_only_reasoning_in_wire_payload_and_preserves_legacy_identity() -> None:
    context, _, _ = fixture_context()
    original = AnswerSettings(model=MISTRAL_LARGE_4)
    explicit = AnswerSettings(model=MISTRAL_LARGE_4, reasoning_effort="none")
    before = request_payload(context, original)
    after = request_payload(context, explicit)
    assert after == {**before, "reasoning": {"effort": "none"}}
    assert after["max_completion_tokens"] == 1024
    assert len(context.passages) == 5
    assert after["messages"] == before["messages"]
    assert after["provider"] == {"require_parameters": True, "allow_fallbacks": False}
    legacy_key = hash_bytes(
        json_bytes(
            {
                "context": context.model_dump(mode="json"),
                "settings": {
                    "model": MISTRAL_LARGE_4,
                    "timeout_seconds": 60,
                    "max_completion_tokens": 1024,
                },
                "prompt_version": PROMPT_VERSION,
                "payload": before,
            }
        )
    )
    assert request_identity(context, original) == legacy_key
    assert request_identity(context, explicit) != legacy_key


def test_new_cache_is_separate_and_preserves_failed_attempt_costs(tmp_path) -> None:
    context, _, _ = fixture_context()
    original = AnswerSettings(model=MISTRAL_LARGE_4)
    explicit = AnswerSettings(model=MISTRAL_LARGE_4, reasoning_effort="none")
    response = raw_response(model=MISTRAL_LARGE_4)
    response["choices"][0]["finish_reason"] = "length"
    response["usage"]["cost"] = 0.0044542
    old_provider = SimulatedProvider(response=response)
    with pytest.raises(GenerationFailure):
        generate(tmp_path, context, original, old_provider)
    root = tmp_path / "artifacts/private/local_generation/v1"
    old_hashes = {p: hash_file(p) for p in root.rglob("*.json")}
    provider = SimulatedProvider(response=raw_response(model=MISTRAL_LARGE_4))
    first = generate(tmp_path, context, explicit, provider)
    again = generate(tmp_path, context, explicit, provider)
    assert first.request_identity != request_identity(context, original)
    assert again.reused and len(provider.calls) == 1
    assert all(hash_file(p) == digest for p, digest in old_hashes.items())
    old_outcome = read_json(root / "outcomes" / f"{request_identity(context, original)}.json")
    assert old_outcome["usage"]["cost"] == 0.0044542
    snapshot = read_json(root / "requests" / f"{first.request_identity}.json")
    assert snapshot["reasoning_parameters_sent"] == {"reasoning": {"effort": "none"}}
    with pytest.raises(GenerationFailure) as caught:
        generate(tmp_path, context, original, old_provider)
    assert caught.value.category == "previous_attempt" and len(old_provider.calls) == 1


def test_none_is_not_silently_applied_to_other_models() -> None:
    context, _, _ = fixture_context()
    assert "reasoning" not in request_payload(context, AnswerSettings(model="fictional/generator"))
    with pytest.raises(ValidationError):
        AnswerSettings(model="fictional/generator", reasoning_effort="none")
    with pytest.raises(ValidationError):
        AnswerSettings(model=MISTRAL_LARGE_4, reasoning_effort="high")


def test_streamlit_mistral_generation_explicit_none_only_after_click(monkeypatch, tmp_path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from finrag.ui import generation_panel as panel

    monkeypatch.setenv("OPENROUTER_GENERATION_MODEL", MISTRAL_LARGE_4)
    from test_generation_units import unit_answer

    provider = SimulatedProvider(response=raw_response(answer=unit_answer(), model=MISTRAL_LARGE_4))
    captured = []

    def simulated(project, context, settings):
        captured.append(settings)
        return generate(tmp_path, context, settings, provider)

    monkeypatch.setattr(panel, "generate", simulated)
    script = tmp_path / "panel.py"
    script.write_text(
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from finrag.ui.model_controls import model_controls\n"
        "with st.sidebar:\n"
        "    st.session_state['generation_choice'] = model_controls(Path.cwd())\n"
        "from test_generation import fixture_context\n"
        "from finrag.ui.generation_panel import generation_panel\n"
        "context, hits, document = fixture_context()\n"
        "generation_panel(Path.cwd(), context.question, hits, [document], "
        "'precomputed_BGE', True, lambda *args: None)\n",
        encoding="utf-8",
    )
    ui = AppTest.from_file(str(script), default_timeout=15).run()
    assert not ui.exception and not captured
    assert any("reasoning.effort=none" in c.value for c in ui.caption)
    from ui_fixtures import accept_uncertain_cost

    accept_uncertain_cost(ui)
    ui.button(key="generate_answer").click().run()
    assert not ui.exception and len(provider.calls) == 1
    assert captured[0].reasoning_effort == "none"
    assert provider.calls[0]["reasoning"] == {"effort": "none"}
    ui.run()
    assert len(provider.calls) == 1
