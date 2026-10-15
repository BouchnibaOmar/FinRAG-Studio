import http.client
import json

import pytest
from test_generation import SimulatedProvider, fixture_context, raw_response

from finrag.config import FinragError
from finrag.generation import answers
from finrag.generation import citation_units as units
from finrag.generation.cache import generate, parse_reply
from finrag.generation.openrouter import GenerationFailure
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import read_json


def unit_answer() -> dict:
    return {
        "statut": "reponse",
        "affirmations": [
            {"texte": "Le revenu 2025 est de 100 millions EUR.", "unit_ids": ["S1.U001"]}
        ],
        "comparaison": False,
        "operandes": [],
    }


def unit_context(text: str | None = None) -> units.UnitContext:
    original, hits, doc = fixture_context()
    if text is not None:
        hits[0]["text"] = text
    return units.build_context(original.question, hits, [doc], original.ranking)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network call")

    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fictional-private-key")


def test_unicode_units_partition_payload_and_local_citations() -> None:
    text = "BNP\u00a0Paribas\u00a0SA\n" + "x" * 585 + "\r\n2025 : 100 EUR.\n "
    context = unit_context(text)
    first = [u for u in context.units if u.passage_id == "S1"]
    assert len(first) > 1
    assert "".join(u.text for u in first) == text
    assert first[0].start == 0 and first[-1].end == len(text)
    assert all(a.end == b.start for a, b in zip(first, first[1:], strict=False))
    for unit in first:
        assert text[unit.start : unit.end] == unit.text
    settings = answers.AnswerSettings(model=answers.MISTRAL_LARGE_4, reasoning_effort="none")
    payload = units.request_payload(context, settings)
    data = json.loads(payload["messages"][1]["content"])
    assert "".join(u["texte"] for u in data["passages"][0]["unites"]) == text
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["max_completion_tokens"] == 1024
    serialized = json.dumps(payload)
    assert "ANNOTATION_SECRET" not in serialized and "secret.pdf" not in serialized
    assert "extrait" not in payload["response_format"]["json_schema"]["schema"]["properties"]
    answer = units.validate_answer(context, unit_answer())
    citation = units.resolve_citations(context, answer)[0]
    source = context.passages[0]
    assert citation.text == text[citation.start : citation.end]
    assert "\u00a0" in citation.text and "\n" in citation.text
    assert citation.document_id == source.document_id
    assert citation.physical_page_number == source.physical_page_number
    assert citation.chunk_id == source.chunk_id


@pytest.mark.parametrize("damage", ["unknown", "unknown_unit", "empty", "duplicate", "old_quote"])
def test_invalid_references_rejected(damage) -> None:
    answer = unit_answer()
    claim = answer["affirmations"][0]
    if damage == "unknown":
        claim["unit_ids"] = ["S9.U001"]
    elif damage == "unknown_unit":
        claim["unit_ids"] = ["S1.U999"]
    elif damage == "empty":
        claim["unit_ids"] = []
    elif damage == "duplicate":
        claim["unit_ids"] *= 2
    else:
        answer["citations"] = [{"passage_id": "S1", "extrait": "Revenu"}]
    with pytest.raises(FinragError):
        units.validate_answer(unit_context(), answer)


@pytest.mark.parametrize("field,value", [("start", 1), ("end", 2), ("text", "Revenu modifié")])
def test_source_positions_and_text_cannot_be_changed(field, value) -> None:
    context = unit_context()
    altered = context.units[0].model_copy(update={field: value})
    context = context.model_copy(update={"units": [altered, *context.units[1:]]})
    with pytest.raises(FinragError, match="tranches exactes"):
        units.request_payload(context, answers.AnswerSettings(model="fictional/generator"))
    with pytest.raises(FinragError):
        units.resolve_citations(context, units.UnitAnswer.model_validate(unit_answer()))


def test_refusal_and_comparison_requirements() -> None:
    context = unit_context()
    context = context.model_copy(update={"question": "Comment évolue le revenu ?"})
    answer = unit_answer()
    with pytest.raises(FinragError, match="deux opérandes"):
        units.validate_answer(context, answer)
    answer.update(
        comparaison=True,
        operandes=[
            {"libelle": "2025", "valeur": "100", "unit_ids": ["S1.U001"]},
            {"libelle": "2024", "valeur": "90", "unit_ids": ["S1.U001"]},
        ],
    )
    assert units.validate_answer(context, answer).statut == "reponse"
    answer["operandes"][1]["valeur"] = "999"
    with pytest.raises(FinragError, match="valeur"):
        units.validate_answer(context, answer)
    refusal = {
        "statut": "preuves_insuffisantes",
        "affirmations": [],
        "comparaison": True,
        "operandes": [],
    }
    result = units.validate_answer(context, refusal)
    assert units.resolve_citations(context, result) == []
    refusal["affirmations"] = unit_answer()["affirmations"]
    with pytest.raises(FinragError, match="refus"):
        units.validate_answer(context, refusal)


def test_v2_cache_separation_reuse_and_failed_attempt_preservation(tmp_path) -> None:
    original, _, _ = fixture_context()
    context = unit_context()
    settings = answers.AnswerSettings(model="fictional/generator")
    rejected = raw_response()
    rejected["choices"][0]["message"]["content"] = "{"
    old_provider = SimulatedProvider(response=rejected)
    with pytest.raises(GenerationFailure):
        generate(tmp_path, original, settings, old_provider)
    old_root = tmp_path / "artifacts/private/local_generation/v1"
    hashes = {p: hash_file(p) for p in old_root.rglob("*.json")}
    provider = SimulatedProvider(response=raw_response(answer=unit_answer()))
    first = generate(tmp_path, context, settings, provider)
    again = generate(tmp_path, context, settings, provider)
    assert again.reused and len(provider.calls) == 1
    assert first.request_identity != answers.request_identity(original, settings)
    assert first.request_identity == units.request_identity(context, settings)
    new_root = tmp_path / "artifacts/private/local_generation/v2"
    record = read_json(new_root / "records" / f"{first.request_identity}.json")
    assert record["prompt_version"] == units.PROMPT_VERSION
    assert record["context"]["units"] == context.model_dump(mode="json")["units"]
    assert all(hash_file(p) == digest for p, digest in hashes.items())
    outcome = read_json(
        old_root / "outcomes" / f"{answers.request_identity(original, settings)}.json"
    )
    assert outcome["status"] == "invalid_output" and outcome["usage"]["cost"] == 0.001


def test_v2_invalid_output_is_not_repaired_or_retried(tmp_path) -> None:
    context = unit_context()
    settings = answers.AnswerSettings(model="fictional/generator")
    invalid = unit_answer()
    invalid["affirmations"][0]["unit_ids"] = ["S9.U001"]
    provider = SimulatedProvider(response=raw_response(answer=invalid))
    with pytest.raises(GenerationFailure) as caught:
        generate(tmp_path, context, settings, provider)
    assert caught.value.diagnostic["validation_stage"] == "grounding"
    with pytest.raises(GenerationFailure) as caught:
        generate(tmp_path, context, settings, provider)
    assert caught.value.category == "previous_attempt" and len(provider.calls) == 1
    response = raw_response(answer=unit_answer())
    response["choices"][0]["finish_reason"] = "length"
    with pytest.raises(GenerationFailure):
        parse_reply(context, settings, response)


def test_v2_identity_covers_units_prompt_and_settings(monkeypatch) -> None:
    context = unit_context()
    settings = answers.AnswerSettings(model=answers.MISTRAL_LARGE_4)
    before = units.request_identity(context, settings)
    assert before != units.request_identity(
        context,
        settings.model_copy(
            update={
                "reasoning_effort": "none",
            }
        ),
    )
    monkeypatch.setattr(units, "INSTRUCTION", units.INSTRUCTION + " Version suivante.")
    assert before != units.request_identity(context, settings)


def test_whitespace_only_unit_and_unicode_codepoint_positions_are_preserved() -> None:
    text = "\u00a0" * units.UNIT_CHARACTERS + "\n\U0001f4b6Revenu\r\n"
    context = unit_context(text)
    first = [u for u in context.units if u.passage_id == "S1"]
    assert first[0].text == "\u00a0" * units.UNIT_CHARACTERS
    assert first[1].start == units.UNIT_CHARACTERS
    assert "".join(u.text for u in first) == text
    units.verify_units(context)


def test_modified_source_hash_and_duplicate_sources_are_rejected() -> None:
    context = unit_context()
    altered = context.passages[0].model_copy(update={"text": "Autre texte"})
    with pytest.raises(FinragError, match="empreinte"):
        units.verify_units(context.model_copy(update={"passages": [altered]}))
    with pytest.raises(FinragError, match="dupliqués"):
        units.verify_units(context.model_copy(update={"passages": [altered, altered]}))
