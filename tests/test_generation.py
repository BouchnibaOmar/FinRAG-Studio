import copy
import http.client
import json

import pytest
from pydantic import ValidationError
from test_local_ui import search_fixture

from finrag.config import FinragError
from finrag.generation.answers import (
    REFUSAL,
    AnswerSettings,
    build_context,
    request_identity,
    request_payload,
    validate_answer,
)
from finrag.generation.cache import generate, parse_reply
from finrag.generation.openrouter import GenerationFailure, OpenRouterGenerator
from finrag.retrieval.candidate_reranking import TransportReply


def forbidden(*args, **kwargs):
    raise AssertionError("Real network or annotation access forbidden")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from ui_fixtures import synthetic_catalogue

    from finrag.ui import model_controls as controls

    monkeypatch.setattr(controls, "read_latest_catalogue", lambda _: synthetic_catalogue())
    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fictional-private-key")
    monkeypatch.setenv("OPENROUTER_GENERATION_MODEL", "fictional/generator")


def fixture_context(question="Quel revenu ?", ranking="free_bm25"):
    _, doc = search_fixture()
    hits = [
        {
            "chunk_id": f"chunk-{i}",
            "document_id": doc.document_id,
            "source_filename": doc.source_filename,
            "source_sha256": doc.source_sha256,
            "physical_page_number": i + 1,
            "rank": i + 1,
            "text": f"Revenu 2025 : {100 + i} millions EUR. Revenu 2024 : 90 millions EUR.",
            "reference_answer": "ANNOTATION_SECRET",
            "local_path": "C:\\private\\secret.pdf",
        }
        for i in range(10)
    ]
    return build_context(question, hits, [doc], ranking), hits, doc


def valid_answer():
    return {
        "statut": "reponse",
        "affirmations": [
            {"texte": "Le revenu 2025 est de 100 millions EUR.", "source_ids": ["S1"]}
        ],
        "citations": [{"passage_id": "S1", "extrait": "Revenu 2025 : 100 millions EUR."}],
        "comparaison": False,
        "operandes": [],
    }


def raw_response(answer=None, **changes):
    return {
        "model": "fictional/generator",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(valid_answer() if answer is None else answer)},
            }
        ],
        "usage": {"total_tokens": 100, "cost": 0.001},
        **changes,
    }


class SimulatedProvider:
    def __init__(self, response=None, status=200, failure=None):
        self.response = raw_response() if response is None else response
        self.status, self.failure, self.calls = status, failure, []

    def request(self, payload):
        self.calls.append(copy.deepcopy(payload))
        if self.failure:
            raise self.failure
        return TransportReply(self.status, json.dumps(self.response).encode(), 0.25)


def test_valid_answer_and_exact_top_five_payload_without_annotations() -> None:
    context, hits, _ = fixture_context()
    payload = request_payload(context, AnswerSettings(model="fictional/generator"))
    data = json.loads(payload["messages"][1]["content"])
    assert data["question"] == "Quel revenu ?"
    assert [p["texte"] for p in data["passages"]] == [h["text"] for h in hits[:5]]
    assert [p["passage_id"] for p in data["passages"]] == ["S1", "S2", "S3", "S4", "S5"]
    assert not any(
        value in json.dumps(payload)
        for value in (
            "ANNOTATION_SECRET",
            "local_path",
            "reference_answer",
            "source_filename",
            "chunk-0",
        )
    )
    assert payload["max_completion_tokens"] == 1024
    assert payload["provider"] == {"allow_fallbacks": False, "require_parameters": True}
    assert validate_answer(context, valid_answer()).statut == "reponse"


@pytest.mark.parametrize(
    "damage",
    [
        "unknown_id",
        "invented_quote",
        "no_sources",
        "no_citation",
        "invented_page",
        "missing_claims",
    ],
)
def test_invalid_citations_or_claims_rejected(damage) -> None:
    context, _, _ = fixture_context()
    answer = valid_answer()
    if damage == "unknown_id":
        answer["citations"][0]["passage_id"] = "S9"
    elif damage == "invented_quote":
        answer["citations"][0]["extrait"] = "Revenu : 999 milliards USD."
    elif damage == "no_sources":
        answer["affirmations"][0]["source_ids"] = []
    elif damage == "no_citation":
        answer["citations"] = []
    elif damage == "invented_page":
        answer["citations"][0]["page"] = 999
    else:
        answer["affirmations"] = []
    with pytest.raises(FinragError):
        validate_answer(context, answer)


def test_comparison_missing_operand_rejected_or_explicitly_refused() -> None:
    context, _, _ = fixture_context("Compare les revenus 2025 et 2024.")
    answer = valid_answer()
    with pytest.raises(FinragError, match="deux opérandes"):
        validate_answer(context, answer)
    answer["comparaison"] = True
    answer["operandes"] = [{"libelle": "2025", "valeur": "100", "source_ids": ["S1"]}]
    with pytest.raises(FinragError, match="deux opérandes"):
        validate_answer(context, answer)
    insufficient = {
        "statut": "preuves_insuffisantes",
        "affirmations": [],
        "citations": [],
        "comparaison": True,
        "operandes": [],
    }
    assert validate_answer(context, insufficient).statut == "preuves_insuffisantes"
    assert REFUSAL == "Les passages retrouvés ne suffisent pas pour répondre."
    answer["citations"][0]["extrait"] += " Revenu 2024 : 90 millions EUR."
    answer["operandes"].append({"libelle": "2024", "valeur": "90", "source_ids": ["S1"]})
    assert validate_answer(context, answer).comparaison
    answer["operandes"][1]["valeur"] = "999"
    with pytest.raises(FinragError, match="opérande"):
        validate_answer(context, answer)


@pytest.mark.parametrize(
    "damage", ["empty", "non_text", "truncated", "different_model", "bad_json"]
)
def test_provider_response_validation(damage) -> None:
    context, _, _ = fixture_context()
    response = raw_response()
    if damage == "empty":
        response["choices"][0]["message"]["content"] = ""
    elif damage == "non_text":
        response["choices"][0]["message"]["content"] = []
    elif damage == "truncated":
        response["choices"][0]["finish_reason"] = "length"
    elif damage == "different_model":
        response["model"] = "fictional/substitute"
    else:
        response["choices"][0]["message"]["content"] = "{"
    with pytest.raises(GenerationFailure) as caught:
        parse_reply(context, AnswerSettings(model="fictional/generator"), response)
    assert caught.value.category == "invalid_output"


def test_cache_reuse_and_no_annotation_reads(tmp_path, monkeypatch) -> None:
    from finrag.generation import cache

    context, _, _ = fixture_context()
    settings = AnswerSettings(model="fictional/generator")
    provider = SimulatedProvider()
    original_read = cache.read_json

    def only_generation_cache(path):
        assert path.is_relative_to(tmp_path / "artifacts/private/local_generation/v1")
        return original_read(path)

    monkeypatch.setattr(cache, "read_json", only_generation_cache)
    first = generate(tmp_path, context, settings, provider)
    again = generate(tmp_path, context, settings, provider)
    assert not first.reused and again.reused and len(provider.calls) == 1
    assert first.answer == again.answer
    records = list((tmp_path / "artifacts/private/local_generation/v1/records").glob("*.json"))
    assert len(records) == 1
    assert "fictional-private-key" not in records[0].read_text(encoding="utf-8")


@pytest.mark.parametrize("change", ["question", "passage", "order", "page", "ranking", "model"])
def test_cache_identity_changes_with_effective_configuration(change) -> None:
    context, hits, doc = fixture_context()
    settings = AnswerSettings(model="fictional/generator")
    original = request_identity(context, settings)
    if change == "model":
        settings = AnswerSettings(model="fictional/other")
    else:
        if change == "passage":
            hits[0]["text"] += " Additional context."
        if change == "order":
            hits[0], hits[1] = hits[1], hits[0]
        if change == "page":
            hits[0]["physical_page_number"] += 1
        context = build_context(
            "Autre question ?" if change == "question" else context.question,
            hits,
            [doc],
            "precomputed_BGE" if change == "ranking" else context.ranking,
        )
    assert request_identity(context, settings) != original


@pytest.mark.parametrize("failure", ["network", "invalid", "http"])
def test_failed_attempts_are_preserved_and_never_replayed(tmp_path, failure) -> None:
    context, _, _ = fixture_context()
    settings = AnswerSettings(model="fictional/generator")
    provider = SimulatedProvider(
        failure=TimeoutError("private message must not enter logs")
        if failure == "network"
        else None,
        response={} if failure == "invalid" else {"error": {"message": "fictional-private-key"}},
        status=503 if failure == "http" else 200,
    )
    with pytest.raises(GenerationFailure):
        generate(tmp_path, context, settings, provider)
    with pytest.raises(GenerationFailure) as caught:
        generate(tmp_path, context, settings, provider)
    assert caught.value.category == "previous_attempt" and len(provider.calls) == 1
    root = tmp_path / "artifacts/private/local_generation/v1"
    assert len(list((root / "attempts").glob("*.json"))) == 1
    assert len(list((root / "outcomes").glob("*.json"))) == 1
    text = next((root / "outcomes").glob("*.json")).read_text(encoding="utf-8")
    assert "fictional-private-key" not in text and "private message" not in text
    diagnostic = json.loads(text)
    assert diagnostic["status"] in ("failed_api", "invalid_output")
    if failure in ("network", "http"):
        assert diagnostic["status"] == "failed_api"


def test_insufficient_evidence_is_a_valid_cached_result(tmp_path) -> None:
    context, _, _ = fixture_context()
    provider = SimulatedProvider(
        response=raw_response(
            {
                "statut": "preuves_insuffisantes",
                "affirmations": [],
                "citations": [],
                "comparaison": False,
                "operandes": [],
            }
        )
    )
    settings = AnswerSettings(model="fictional/generator")
    assert generate(tmp_path, context, settings, provider).answer.statut == "preuves_insuffisantes"
    assert generate(tmp_path, context, settings, provider).reused
    assert len(provider.calls) == 1


def test_exact_model_and_credentials_are_required(monkeypatch) -> None:
    for model in ("", "model-without-provider", "openrouter/auto"):
        with pytest.raises(ValidationError):
            AnswerSettings(model=model)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(GenerationFailure, match="OPENROUTER_API_KEY"):
        OpenRouterGenerator(AnswerSettings(model="fictional/generator"))


def test_transport_single_post_no_retry_and_original_cause(monkeypatch) -> None:
    calls = []

    class Connection:
        def request(self, *args, **kwargs):
            calls.append((args, kwargs))
            raise TimeoutError("Do not log private error")

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(http.client, "HTTPSConnection", lambda host, timeout: Connection())
    provider = OpenRouterGenerator(AnswerSettings(model="fictional/generator"))
    with pytest.raises(GenerationFailure) as caught:
        provider.request({"model": "fictional/generator"})
    assert len(calls) == 2 and calls[-1] == "closed"
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert caught.value.diagnostic["timeout"] is True


def test_secret_in_question_is_not_sent_or_cached(tmp_path) -> None:
    context, _, _ = fixture_context("Que vaut fictional-private-key ?")
    provider = SimulatedProvider()
    with pytest.raises(GenerationFailure, match="contient la clé"):
        generate(tmp_path, context, AnswerSettings(model="fictional/generator"), provider)
    assert not provider.calls
    assert not list(tmp_path.rglob("*.json"))


def test_prompt_and_schema_are_part_of_cache_identity(monkeypatch) -> None:
    from finrag.generation import answers

    context, _, _ = fixture_context()
    settings = AnswerSettings(model="fictional/generator")
    before = request_identity(context, settings)
    monkeypatch.setattr(answers, "INSTRUCTION", answers.INSTRUCTION + " Nouvelle version.")
    assert request_identity(context, settings) != before


def test_malformed_message_and_duplicate_json_keys_are_rejected() -> None:
    context, _, _ = fixture_context()
    settings = AnswerSettings(model="fictional/generator")
    response = raw_response()
    response["choices"][0]["message"] = None
    with pytest.raises(GenerationFailure):
        parse_reply(context, settings, response)
    response = raw_response()
    response["choices"][0]["message"]["content"] = (
        '{"statut":"reponse","statut":"preuves_insuffisantes"}'
    )
    with pytest.raises(GenerationFailure):
        parse_reply(context, settings, response)


def test_streamlit_explicit_generation_reruns_cache_and_invalidation(monkeypatch, tmp_path) -> None:
    pytest.importorskip("streamlit")
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    from finrag.retrieval.company_router import route_company
    from finrag.ui import app
    from finrag.ui import generation_panel as panel
    from finrag.ui.resources import SavedExample

    _, hits, doc = fixture_context()
    for hit in hits:
        hit["lexical_score"] = 10.0 - hit["rank"]
    searched = []

    def search(question):
        searched.append(question)
        return SimpleNamespace(
            query_used="A translated query that must not be used for generation?",
            translation_status="local_correction",
            route=route_company(question, [doc]),
            hits=[SimpleNamespace(model_dump=lambda mode, h=h: dict(h)) for h in hits],
        )

    resource = SimpleNamespace(search=search, notices=[])
    monkeypatch.setattr(app, "PROJECT", tmp_path)
    monkeypatch.setattr(app, "document_catalogue", lambda _: [doc])
    monkeypatch.setattr(app, "search_resource", lambda _: resource)
    from test_generation_units import unit_answer

    provider = SimulatedProvider(response=raw_response(answer=unit_answer()))
    monkeypatch.setattr(
        panel,
        "generate",
        lambda project, context, settings: generate(project, context, settings, provider),
    )
    bge_hits = [{**h, "bge_score": 1.0} for h in hits]
    example = SavedExample(
        "synthetic",
        "Question originale sauvegardée ?",
        "English query?",
        ["BNP Paribas"],
        hits,
        bge_hits,
    )
    monkeypatch.setattr(app, "saved_examples", lambda _: [example])
    script = tmp_path / "ui_runner.py"
    script.write_text("from finrag.ui.app import main\nmain()", encoding="utf-8")
    ui = AppTest.from_file(str(script), default_timeout=15).run()
    assert not ui.exception and not provider.calls
    ui.text_area[0].input("Quel revenu ?").run()
    ui.button(key="free_search_submit").click().run()
    assert not ui.exception and not provider.calls
    from ui_fixtures import accept_uncertain_cost

    accept_uncertain_cost(ui)
    ui.button(key="generate_answer").click().run()
    assert not ui.exception and len(provider.calls) == 1
    data = json.loads(provider.calls[0]["messages"][1]["content"])
    assert data["question"] == "Quel revenu ?" and len(data["passages"]) == 5
    assert not ui.error
    ui.select_slider(key="free_depth").set_value(20).run()
    assert not ui.exception and len(provider.calls) == 1
    assert any(t.value.startswith("Le revenu 2025 est") for t in ui.text)
    accept_uncertain_cost(ui)
    ui.button(key="generate_answer").click().run()
    assert len(provider.calls) == 1
    assert any("réutilisée depuis le cache" in c.value for c in ui.caption)
    ui.selectbox(key="generation_model").set_value("fictional/other").run()
    assert not any(t.value.startswith("Le revenu 2025 est") for t in ui.text)
    assert len(provider.calls) == 1
    ui.selectbox(key="generation_model").set_value("fictional/generator").run()
    accept_uncertain_cost(ui)
    ui.button(key="generate_answer").click().run()
    assert len(provider.calls) == 1
    ui.text_area[0].input("Une autre question ?").run()
    assert not any(t.value.startswith("Le revenu 2025 est") for t in ui.text)
    assert len(searched) == 1
    ui.sidebar.radio[0].set_value("Explorer les exemples").run()
    assert any(i.value == app.PRECOMPUTED_NOTICE for i in ui.info)
    assert not provider.calls[1:]
    accept_uncertain_cost(ui)
    ui.button(key="generate_answer").click().run()
    assert len(provider.calls) == 2
    assert (
        json.loads(provider.calls[1]["messages"][1]["content"])["question"]
        == example.original_question
    )
    ui.radio(key="saved_method").set_value("BGE").run()
    assert len(provider.calls) == 2
    assert not any(t.value.startswith("Le revenu 2025 est") for t in ui.text)
