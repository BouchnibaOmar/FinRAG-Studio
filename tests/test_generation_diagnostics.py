import http.client
import json

import pytest
from test_generation import SimulatedProvider, fixture_context, raw_response, valid_answer

from finrag.generation.answers import AnswerSettings, request_identity, request_payload
from finrag.generation.cache import generate
from finrag.generation.diagnostics import response_snapshot
from finrag.generation.openrouter import GenerationFailure
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import read_json


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No real API permitted")

    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fictional-private-key")


def test_length_preserves_response_and_detailed_usage_without_guessing_json(tmp_path) -> None:
    context, _, _ = fixture_context()
    settings = AnswerSettings(model="fictional/generator")
    response = raw_response(id="gen-fictional")
    response["choices"][0].update(finish_reason="length", native_finish_reason="max_tokens")
    response["usage"] = {
        "prompt_tokens": 3403,
        "completion_tokens": 1024,
        "total_tokens": 4427,
        "cost": 0.0044542,
        "completion_tokens_details": {"reasoning_tokens": 700},
    }
    provider = SimulatedProvider(response=response)
    with pytest.raises(GenerationFailure, match="finish_reason=length") as caught:
        generate(tmp_path, context, settings, provider)
    assert caught.value.diagnostic["validation_stage"] == "completion"
    assert caught.value.diagnostic["content_analysis"]["stage"] == "complete"
    root = tmp_path / "artifacts/private/local_generation/v1"
    key = request_identity(context, settings)
    outcome = read_json(root / "outcomes" / f"{key}.json")
    saved = read_json(root / "responses" / f"{key}.json")
    assert outcome["id"] == "gen-fictional"
    assert outcome["model"] == "fictional/generator"
    assert outcome["native_finish_reason"] == "max_tokens"
    assert outcome["usage"]["completion_tokens_details"]["reasoning_tokens"] == 700
    assert (
        saved["body_json_redacted"]["choices"][0]["message"]["content"]
        == response["choices"][0]["message"]["content"]
    )
    assert json.loads(saved["body_text_redacted"]) == response
    assert not list((root / "records").glob("*.json"))


@pytest.mark.parametrize(
    ("content", "stage", "code"),
    [
        ("", "content", "empty_content"),
        ('{"statut":', "json", "invalid_json"),
        ('{"bogus": 1}', "schema", "invalid_schema"),
    ],
)
def test_content_failures_have_separate_diagnostics(tmp_path, content, stage, code) -> None:
    context, _, _ = fixture_context()
    response = raw_response()
    response["choices"][0]["message"]["content"] = content
    with pytest.raises(GenerationFailure) as caught:
        generate(
            tmp_path,
            context,
            AnswerSettings(model="fictional/generator"),
            SimulatedProvider(response=response),
        )
    analysis = caught.value.diagnostic["content_analysis"]
    assert analysis["stage"] == stage and analysis["code"] == code
    if stage == "json":
        assert analysis["error_at_end"] and analysis["incompleteness_not_proven_by_parser_alone"]
    if stage == "schema":
        assert all(set(e) == {"type", "location"} for e in analysis["errors"])


def test_citation_failure_retains_exact_rule_and_content(tmp_path) -> None:
    context, _, _ = fixture_context()
    answer = valid_answer()
    answer["citations"][0]["extrait"] = "Invented financial evidence"
    response = raw_response(answer)
    with pytest.raises(GenerationFailure) as caught:
        generate(
            tmp_path,
            context,
            AnswerSettings(model="fictional/generator"),
            SimulatedProvider(response=response),
        )
    analysis = caught.value.diagnostic["content_analysis"]
    assert analysis["stage"] == "grounding"
    assert "n’existe pas exactement" in analysis["rule"]


def test_request_capture_does_not_change_reasoning_limit_or_replay(tmp_path) -> None:
    context, _, _ = fixture_context()
    settings = AnswerSettings(model="fictional/generator")
    key = request_identity(context, settings)
    payload = request_payload(context, settings)
    response = raw_response()
    response["choices"][0]["finish_reason"] = "length"
    provider = SimulatedProvider(response=response)
    with pytest.raises(GenerationFailure):
        generate(tmp_path, context, settings, provider)
    root = tmp_path / "artifacts/private/local_generation/v1"
    hashes = {p: hash_file(p) for p in root.rglob("*.json")}
    snapshot = read_json(root / "requests" / f"{key}.json")
    assert snapshot["payload"] == payload == provider.calls[0]
    assert payload["max_completion_tokens"] == 1024
    assert snapshot["reasoning_parameters_sent"] == {}
    assert set(snapshot["reasoning_parameters_omitted"]) == {
        "reasoning",
        "reasoning_effort",
        "include_reasoning",
    }
    assert snapshot["context_for_offline_validation"] == context.model_dump(mode="json")
    with pytest.raises(GenerationFailure) as caught:
        generate(tmp_path, context, settings, provider)
    assert caught.value.category == "previous_attempt" and len(provider.calls) == 1
    assert hashes == {p: hash_file(p) for p in root.rglob("*.json")}


def test_response_capture_removes_credentials_even_on_failure() -> None:
    body = json.dumps(
        {
            "id": "gen-private",
            "headers": {"Authorization": "Bearer another-token"},
            "error": {"message": "fictional-private-key and Bearer another-token"},
            "reasoning": "sk-or-v1-fictionalOtherKey",
        }
    ).encode()
    snapshot = response_snapshot(body, 400, 0.1)
    serialized = json.dumps(snapshot)
    assert "fictional-private-key" not in serialized
    assert "another-token" not in serialized
    assert "sk-or-v1-fictionalOtherKey" not in serialized
    assert snapshot["redaction_applied"]
    assert not snapshot["authentication_headers_recorded"]
    assert "body_text_redacted" not in snapshot


def test_non_json_http_body_is_preserved_redacted() -> None:
    snapshot = response_snapshot(b'{"unfinished": "fictional-private-key', 200, 0.1)
    assert snapshot["body_text_redacted"] == '{"unfinished": "[REDACTED]'
    assert snapshot["redaction_applied"]
