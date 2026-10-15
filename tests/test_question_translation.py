import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_bm25 import synthetic_chunks
from test_retrieval_metrics import evaluation_question

from finrag.config import FinragError
from finrag.evaluation.bm25_translation_experiment import (
    dry_run,
    evaluate,
    gain_regression,
    selected_query,
    translation_plan,
)
from finrag.evaluation.question_translation import (
    OpenRouterTranslator,
    TranslationInput,
    TranslationSettings,
    fetch_official_model_metadata,
    input_identity,
    official_price_estimate,
    parse_response,
    request_payload,
    translation_checks,
)
from finrag.evaluation.translation_cache import (
    cached_translation,
    import_corrections,
    load_translation_set,
    save_translation_set,
)
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes, read_json, write_atomic


def source(question: str = "Quelle valeur fictive en 2025 ?") -> TranslationInput:
    return TranslationInput(question=question, source_language="fr", target_language="en")


def settings(**overrides) -> TranslationSettings:
    return TranslationSettings(model="fictional/model", **overrides)


def response(text: str = "What is the fictional value in 2025?") -> dict:
    return {
        "model": "fictional/model",
        "id": "offline-response",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"total_tokens": 10, "prompt_tokens": 8, "completion_tokens": 2},
    }


class OfflineTranslatorDouble:
    def __init__(self, usage_total: int = 10):
        self.calls = []
        self.usage_total = usage_total

    def translate(self, item: TranslationInput) -> dict:
        self.calls.append(item.model_dump(mode="json"))
        payload = response()
        payload["usage"]["total_tokens"] = self.usage_total
        return {**parse_response(item, payload), "source": "offline_double"}


def test_translator_input_and_wire_payload_exclude_all_reference_fields() -> None:
    with pytest.raises(ValidationError):
        TranslationInput.model_validate({**source().model_dump(), "reference_answer": "forbidden"})
    payload = request_payload(source(), settings())
    body = json.loads(payload["messages"][1]["content"])
    assert set(body) == {"question", "source_language", "target_language"}
    assert body == source().model_dump()
    assert "models" not in payload and payload["model"] == "fictional/model"
    assert payload["provider"] == {"allow_fallbacks": False, "require_parameters": True}
    assert payload["temperature"] == 0
    assert "temperature" not in request_payload(source(), settings(temperature_zero=False))


def test_plan_excludes_unanswerable_and_rejects_test() -> None:
    base = evaluation_question(synthetic_chunks(["fictional", "fictional"]))
    unanswerable = evaluation_question(synthetic_chunks(["fictional"]), False)
    names = {base.document_id: "bnp_urd_2025_en.pdf"}
    inputs, ids = translation_plan(
        [base.model_copy(update={"language": "fr"}), unanswerable], names
    )
    assert len(inputs) == len(ids) == 1
    with pytest.raises(FinragError, match="excludes the test"):
        translation_plan([base.model_copy(update={"split": "test"})], names)


def test_seventeen_translated_and_eleven_unchanged_queries() -> None:
    base = evaluation_question(synthetic_chunks(["fictional", "fictional"]))
    other = "sha256:" + "b" * 64
    specs = [("fr", base.document_id)] * 15 + [("en", other)] * 2
    specs += [("en", base.document_id)] * 4 + [("fr", other)] * 7
    questions = [
        base.model_copy(
            update={
                "question_id": f"finrag_v1_bnp_{i + 1:02}",
                "question": f"Fictional question {i + 1} in 2025?",
                "language": language,
                "document_id": doc,
            }
        )
        for i, (language, doc) in enumerate(specs)
    ]
    names = {base.document_id: "bnp_urd_2025_en.pdf", other: "sg_urd_2026_fy2025_fr.pdf"}
    inputs, ids = translation_plan(questions, names)
    assert len(inputs) == 17 and sum(not row["needs_translation"] for row in ids) == 11
    for question, identity in zip(questions, ids, strict=True):
        if not identity["needs_translation"]:
            assert selected_query(question, identity["target_language"], {}) == (
                question.question,
                "unchanged",
            )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"error": {"message": "do not log"}},
        {**response(), "choices": []},
        {
            **response(),
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "Question?"},
                }
            ],
        },
        response(""),
        response("The answer is 100."),
        response("Translation: Question?"),
        response("First line\nSecond line?"),
        {**response(), "model": ""},
        {**response(), "usage": {"total_tokens": -1}},
    ],
)
def test_invalid_api_responses_are_rejected(payload: dict) -> None:
    with pytest.raises(FinragError):
        parse_response(source(), payload)


def test_missing_usage_and_cost_are_not_invented() -> None:
    payload = response()
    payload.pop("usage")
    parsed = parse_response(source(), payload)
    assert parsed["usage"] is None and parsed["reported_cost"] is None
    payload["usage"] = {"total_tokens": 10, "cost": None, "prompt_tokens_details": None}
    parsed = parse_response(source(), payload)
    assert parsed["usage"]["total_tokens"] == 10 and parsed["reported_cost"] is None


def test_numeric_dates_percent_and_sign_checks_remain_candidates() -> None:
    item = source("Quel taux fictif +13,5% en 2025 au 31/12/2025 ?")
    checked = translation_checks(item, "What fictional rate +13.5% in 2025 on 31/12/2025?")
    assert checked["numbers_preserved"] and checked["numeric_dates_preserved_literally"]
    assert checked["review_status"] == "candidate_not_validated"
    assert not translation_checks(item, "What fictional rate -13.5% in 2024?")["numbers_preserved"]
    assert not translation_checks(item, "What fictional rate +13.5% in 2025 on 12/31/2025?")[
        "numeric_dates_preserved_literally"
    ]


def test_cache_reuse_does_not_call_or_overwrite(tmp_path: Path) -> None:
    double = OfflineTranslatorDouble()
    entry = cached_translation(tmp_path, source(), settings(), double)
    before = hash_file(Path(entry["path"]))
    assert cached_translation(tmp_path, source(), settings(), double) == entry
    assert len(double.calls) == 1 and hash_file(Path(entry["path"])) == before
    assert read_json(Path(entry["path"]))["source"] == "offline_double"


def test_call_limit_and_token_limit_block_before_transport(tmp_path: Path) -> None:
    double = OfflineTranslatorDouble()
    cached_translation(tmp_path / "calls", source(), settings(max_calls=1), double)
    with pytest.raises(FinragError, match="budget exhausted"):
        cached_translation(
            tmp_path / "calls",
            source("Quelle autre valeur en 2025 ?"),
            settings(max_calls=1),
            double,
        )
    with pytest.raises(FinragError, match="budget exhausted"):
        cached_translation(tmp_path / "tokens", source(), settings(max_total_tokens=2303), double)
    assert len(double.calls) == 1


def test_failed_attempt_is_not_replayed_on_resume(tmp_path: Path) -> None:
    class FailingOfflineDouble:
        def __init__(self):
            self.calls = 0

        def translate(self, item):
            self.calls += 1
            raise FinragError("Explicit offline failure")

    double = FailingOfflineDouble()
    with pytest.raises(FinragError, match="offline failure"):
        cached_translation(tmp_path, source(), settings(), double)
    with pytest.raises(FinragError, match="no replay"):
        cached_translation(tmp_path, source(), settings(), double)
    assert double.calls == 1


def test_budget_lock_and_supplier_overrun_stop_new_calls(tmp_path: Path) -> None:
    double = OfflineTranslatorDouble(3000)
    with pytest.raises(FinragError, match="usage exceeded"):
        cached_translation(tmp_path, source(), settings(), double)
    with pytest.raises(FinragError, match="allowance overrun"):
        cached_translation(tmp_path, source("Quelle autre valeur en 2025 ?"), settings(), double)
    assert len(double.calls) == 1
    (tmp_path / "budget.lock").mkdir()
    with pytest.raises(FinragError, match="budget lock"):
        cached_translation(tmp_path, source(), settings(), double)


def test_local_correction_preserves_original_history_and_is_not_validated(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    entry = cached_translation(root, source(), settings(), OfflineTranslatorDouble())
    old_digest = hash_file(Path(entry["path"]))
    base_set = save_translation_set(root, [entry])
    corrections = tmp_path / "corrections.json"
    write_atomic(
        corrections,
        json_bytes(
            {
                "base_set_sha256": hash_file(base_set),
                "corrections": [
                    {
                        "input_id": entry["input_id"],
                        "translation": "Which fictional value in 2025?",
                        "reason": "Local wording correction",
                    }
                ],
            }
        ),
    )
    new_set = import_corrections(root, base_set, corrections)
    corrected = load_translation_set(root, new_set)[entry["input_id"]]
    assert hash_file(Path(entry["path"])) == old_digest
    assert corrected["translation"] == "Which fictional value in 2025?"
    assert corrected["source"] == "local_correction"
    assert corrected["usage"] is None and corrected["reported_cost"] is None
    assert corrected["review_status"] == "corrected_locally_not_validated"
    assert corrected["supersedes"]["sha256"] == old_digest


def test_numeric_change_blocks_evaluation_until_corrected() -> None:
    question = evaluation_question(synthetic_chunks(["fictional", "fictional"])).model_copy(
        update={"language": "fr", "question": "Quelle valeur fictive en 2025 ?"}
    )
    item = source(question.question)
    translations = {
        input_identity(item): {
            "input": item.model_dump(),
            "translation": "What fictional value in 2024?",
            "source": "offline_double",
        }
    }
    with pytest.raises(FinragError, match="Numeric/date checks"):
        selected_query(question, "en", translations)


def test_gains_and_regressions_include_complete_multi_proof_coverage() -> None:
    changes = gain_regression(
        {"hit@5": True, "complete_excerpt_coverage@50": False},
        {"hit@5": False, "complete_excerpt_coverage@50": True},
    )
    assert changes["hit@5"]["regression"]
    assert changes["complete_excerpt_coverage@50"]["gain"]


def test_official_pricing_uses_exact_model_and_is_an_estimate() -> None:
    metadata = {
        "id": "fictional/model",
        "pricing": {"prompt": "0.000001", "completion": "0.000002", "request": "0"},
    }
    cost = official_price_estimate(metadata, settings(), 17)
    assert cost["estimated_usd"] == "0.043520"
    with pytest.raises(FinragError, match="exact configured model"):
        official_price_estimate(metadata, TranslationSettings(model="different/model"), 17)


def test_no_model_or_key_no_transport_and_secret_not_in_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(FinragError, match="No exact"):
        OpenRouterTranslator(TranslationSettings())
    with pytest.raises(FinragError, match="not configured"):
        OpenRouterTranslator(settings())
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-secret")

    class FailingConnectionDouble:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            raise OSError("offline-test-secret")

        def close(self):
            pass

    monkeypatch.setattr("http.client.HTTPSConnection", FailingConnectionDouble)
    with pytest.raises(FinragError) as captured:
        OpenRouterTranslator(settings()).translate(source())
    assert "offline-test-secret" not in str(captured.value)


def test_dry_run_lists_17_requests_without_constructing_transport(
    tmp_path: Path, monkeypatch
) -> None:
    experiment = tmp_path / "artifacts/private/experiment"
    items = [source(f"Quelle valeur fictive {index} en 2025 ?").model_dump() for index in range(17)]
    path = experiment / "translator/inputs.jsonl"
    write_atomic(path, jsonl_bytes(items))
    write_atomic(
        experiment / "translator/inputs_digest.json", json_bytes({"sha256": hash_file(path)})
    )

    def forbidden_transport(*args, **kwargs):
        raise AssertionError("Dry-run must never construct a network connection")

    monkeypatch.setattr("http.client.HTTPSConnection", forbidden_transport)
    monkeypatch.chdir(tmp_path)
    output = experiment / "dry.json"
    report = dry_run(experiment, TranslationSettings(), output, None)
    assert report["paid_calls"] == 0 and report["cost_estimate"] is None
    assert len(read_json(output)["requests"]) == 17
    with pytest.raises(FinragError, match="output exists"):
        dry_run(experiment, TranslationSettings(), output, None)


def test_official_metadata_is_selected_by_exact_id_with_an_offline_connection(monkeypatch) -> None:
    records = [
        {"id": "other/model", "pricing": {"prompt": "1", "completion": "1"}},
        {"id": "fictional/model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
    ]

    class CatalogResponseDouble:
        status = 200

        def read(self, size):
            return json.dumps({"data": records}).encode()

    class CatalogConnectionDouble:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, method, path):
            assert (method, path) == ("GET", "/api/v1/models")

        def getresponse(self):
            return CatalogResponseDouble()

        def close(self):
            pass

    monkeypatch.setattr("http.client.HTTPSConnection", CatalogConnectionDouble)
    snapshot = fetch_official_model_metadata("fictional/model")
    assert snapshot["model"]["id"] == "fictional/model"
    with pytest.raises(FinragError, match="not found uniquely"):
        fetch_official_model_metadata("absent/model")


def test_offline_evaluation_pipeline_preserves_all_eleven_untranslated_results(
    tmp_path: Path, monkeypatch
) -> None:
    from test_bm25 import lexical_config

    from finrag.evaluation.document_oracle_metrics import evaluate_candidates
    from finrag.retrieval.bm25 import BM25

    chunks = synthetic_chunks(["fictional 2025", "fictional 2025"])
    other = "sha256:" + "b" * 64
    english = [
        chunk.model_copy(update={"source_filename": "bnp_urd_2025_en.pdf"}) for chunk in chunks
    ]
    french = [
        chunk.model_copy(
            update={
                "document_id": other,
                "source_filename": "sg_urd_2026_fy2025_fr.pdf",
                "source_sha256": "b" * 64,
                "chunk_id": f"fr-{i}",
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    config = lexical_config("B")
    indexes = {english[0].document_id: BM25(english, config), other: BM25(french, config)}
    base = evaluation_question(chunks)
    specs = [("fr", english[0].document_id)] * 15 + [("en", other)] * 2
    specs += [("en", english[0].document_id)] * 4 + [("fr", other)] * 7
    questions = [
        base.model_copy(
            update={
                "question_id": f"finrag_v1_bnp_{i + 1:02}",
                "question": f"Fictional query {chr(65 + i)} in 2025?",
                "language": language,
                "document_id": doc,
            }
        )
        for i, (language, doc) in enumerate(specs)
    ]
    names = {english[0].document_id: "bnp_urd_2025_en.pdf", other: "sg_urd_2026_fy2025_fr.pdf"}
    inputs, ids = translation_plan(questions, names)
    experiment = tmp_path / "artifacts/private/experiment"
    originals = []
    for question in questions:
        index = indexes[question.document_id]
        hits = [
            hit.model_dump(mode="json")
            for hit in index.search(question.question, len(index.chunks))
        ]
        originals.append(
            {
                "question_id": question.question_id,
                "hits": hits,
                "metrics": evaluate_candidates(question, hits, True),
            }
        )
    write_atomic(experiment / "evaluation/original_top50.jsonl", jsonl_bytes(originals))
    write_atomic(experiment / "evaluation/identities.json", json_bytes(ids))
    statistics = {
        doc: {
            "passages": len(index.chunks),
            "average_length": index.average_length,
            "total_lexical_tokens": sum(index.lengths),
        }
        for doc, index in indexes.items()
    }
    write_atomic(
        experiment / "experiment.json",
        json_bytes(
            {
                "prepared_hashes": {
                    "evaluation/identities.json": hash_file(
                        experiment / "evaluation/identities.json"
                    )
                },
                "source_hashes": {},
                "document_statistics": statistics,
            }
        ),
    )

    class IdentityOfflineDouble:
        def translate(self, item):
            return {**parse_response(item, response(item.question)), "source": "offline_double"}

    cache = experiment / "translator/cache"
    translation_set = save_translation_set(
        cache,
        [
            cached_translation(
                cache, TranslationInput.model_validate(item), settings(), IdentityOfflineDouble()
            )
            for item in inputs
        ],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "finrag.evaluation.bm25_translation_experiment.load_control",
        lambda project: (english + french, config, questions, indexes),
    )
    result = evaluate(experiment, translation_set, experiment / "results")
    assert result["unchanged_queries_verified"] == 11
    saved = [
        json.loads(line)
        for line in (experiment / "results/translated_top50.jsonl")
        .read_text(encoding="utf-8")
        .split("\n")
        if line
    ]
    assert len(saved) == 28
    for original, candidate in zip(originals, saved, strict=True):
        if candidate["translation_source"] == "unchanged":
            assert candidate["hits"] == original["hits"] and candidate["metrics"] == json.loads(
                json_bytes(original["metrics"])
            )
