import copy
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_candidate_reranking import fixture_input

from finrag.config import FinragError
from finrag.evaluation.local_reranker_experiment import (
    evaluate,
    execute,
    local_key,
    smoke,
    technical_failure,
)
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes, read_json, write_atomic
from finrag.retrieval.candidate_reranking import input_identity
from finrag.retrieval.local_reranker import (
    LocalReranker,
    count_pairs,
    hardware_sample,
    rank_scores,
    validate_local_record,
)
from finrag.retrieval.local_reranker_assets import MAX_PAIR_TOKENS


class OfflinePairTokenizer:
    def __init__(self, length=12):
        self.length, self.calls = length, []

    def __call__(self, question, passage, **kwargs):
        assert kwargs == {
            "add_special_tokens": True,
            "truncation": False,
            "padding": False,
            "return_attention_mask": False,
        }
        self.calls.append((question, passage))
        return {"input_ids": list(range(self.length))}


class OfflineLocalScorer:
    def __init__(self, failure=False):
        self.calls, self.failure = [], failure

    def score_pairs(self, pairs):
        assert len(pairs) == 1
        self.calls.append(pairs[0])
        if self.failure:
            raise RuntimeError("secret document content must not be logged")
        return [0.25]

    def synchronize(self):
        pass

    def gpu_memory(self, reset=False):
        return None


def test_complete_pair_tokens_and_effective_query_without_e5_prefixes() -> None:
    source, *_ = fixture_input(50)
    source = source.model_copy(update={"question": "Actual translated query?"})
    tokenizer = OfflinePairTokenizer(8192)
    counts = count_pairs([source], tokenizer)
    assert len(counts["pairs"]) == 50 and not counts["overflows"]
    assert counts["maximum_tokens"] == MAX_PAIR_TOKENS
    assert tokenizer.calls == [(source.question, c.text) for c in source.candidates]
    with pytest.raises(ValidationError):
        type(source).model_validate({**source.model_dump(), "reference_answer": "forbidden"})


def test_overflow_blocks_inference_before_any_pair_is_scored() -> None:
    source, *_ = fixture_input(50)
    counts = count_pairs([source], OfflinePairTokenizer(8193))
    backend = OfflineLocalScorer()
    with pytest.raises(FinragError, match="overflows"):
        smoke([source], counts, backend)
    assert len(counts["overflows"]) == 50 and not backend.calls


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scores_are_failures(value) -> None:
    source, *_ = fixture_input()
    with pytest.raises(FinragError, match="finite"):
        rank_scores(source, [value] * len(source.candidates))


def test_sort_descending_and_ties_by_opaque_id_keeps_every_score() -> None:
    source, *_ = fixture_input(50)
    ranking = rank_scores(source, [1.0] * 49 + [2.0])
    assert len(ranking) == 50 and ranking[0]["candidate_id"] == source.candidates[-1].candidate_id
    assert [r["candidate_id"] for r in ranking[1:]] == sorted(
        c.candidate_id for c in source.candidates[:-1]
    )


def test_cache_reuse_and_config_mismatch(tmp_path, monkeypatch) -> None:
    source, *_ = fixture_input(50)
    config = {"device": "cpu", "dtype": "float32", "revision": "fixed"}
    counts = {"configuration": config, "overflows": []}
    backend = OfflineLocalScorer()
    monkeypatch.setattr(
        "finrag.evaluation.local_reranker_experiment.resident_memory", lambda: {"rss_bytes": 123}
    )
    execute([source], counts, backend, tmp_path)
    path = tmp_path / f"{local_key(source, config)}.json"
    before = hash_file(path)
    execute([source], counts, backend, tmp_path)
    assert len(backend.calls) == 50 and before == hash_file(path)
    record = read_json(path)
    assert validate_local_record(source, record, config) == record["ranking"]
    with pytest.raises(FinragError, match="configuration"):
        validate_local_record(source, record, {**config, "dtype": "float16"})
    corrupt = copy.deepcopy(record)
    corrupt["ranking"][0]["candidate_id"] = corrupt["ranking"][1]["candidate_id"]
    with pytest.raises(FinragError, match="duplicate"):
        validate_local_record(source, corrupt, config)


def test_failure_preserves_chain_and_never_logs_document_text_or_retries(tmp_path) -> None:
    source, *_ = fixture_input(50)
    config = {"device": "cpu"}
    counts = {"configuration": config, "overflows": []}
    backend = OfflineLocalScorer(failure=True)
    with pytest.raises(FinragError) as caught:
        execute([source], counts, backend, tmp_path)
    assert isinstance(caught.value.__cause__, RuntimeError)
    record = read_json(next(tmp_path.glob("*.json")))
    assert record["status"] == "failed" and "ranking" not in record
    assert "secret document content" not in str(record)
    with pytest.raises(FinragError, match="no automatic retry"):
        execute([source], counts, backend, tmp_path)
    assert len(backend.calls) == 1
    assert (
        technical_failure(RuntimeError("secret"), "load", {})["original_exception_type"]
        == "builtins.RuntimeError"
    )


def test_hardware_sample_is_length_based_and_batch_one_with_real_measurement_fields() -> None:
    source, *_ = fixture_input(50)
    counts = count_pairs([source], OfflinePairTokenizer())
    counts.update(
        {"configuration": {"device": "cpu"}, "input_ids": [counts["pairs"][0]["input_id"]]}
    )
    before = copy.deepcopy(counts)
    sample = hardware_sample(counts["pairs"])
    assert len(sample) == 9 and counts == before
    backend = OfflineLocalScorer()
    report = smoke([source], counts, backend, memory=lambda: {"rss_bytes": 123})
    assert report["measured_pairs"] == 9 and len(backend.calls) == 10
    assert report["seconds"] > 0 and report["pairs_per_second"] > 0
    assert report["estimate_is_not_guarantee"] and report["gpu_memory"] is None


def test_cuda_unavailable_never_loads_model_or_falls_back(monkeypatch, tmp_path) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Model loading must not happen")

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=forbidden),
            AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=forbidden),
        ),
    )
    with pytest.raises(FinragError, match="no CPU fallback"):
        LocalReranker(tmp_path, "cuda", "float16")


def test_evaluator_preserves_28_and_fixed_mistral_12_subsets(tmp_path, monkeypatch) -> None:
    from test_retrieval_metrics import evaluation_question

    from finrag.evaluation.document_oracle_metrics import evaluate_candidates

    source, mapping, hits, chunks, _ = fixture_input(50)
    example = evaluation_question(chunks)
    inputs = [source.model_copy(update={"question": f"Synthetic {i}?"}) for i in range(28)]
    questions = [
        example.model_copy(update={"question_id": f"finrag_v1_bnp_{i + 1:02}"}) for i in range(28)
    ]
    baseline = [
        {
            "input_id": input_identity(item),
            "question_id": q.question_id,
            "hits": hits,
            "candidate_id_to_chunk_id": mapping,
            "metrics": evaluate_candidates(q, hits, True),
        }
        for item, q in zip(inputs, questions, strict=True)
    ]
    experiment = tmp_path / "artifacts/private/experiment"
    refs = tmp_path / "artifacts/private/bm25_query_translation_dev/evaluation/references.jsonl"
    write_atomic(refs, jsonl_bytes(q.model_dump(mode="json") for q in questions))
    write_atomic(experiment / "evaluation/baseline.jsonl", jsonl_bytes(baseline))
    write_atomic(
        tmp_path / "artifacts/private/bm25_company_routing_dev/run_1/comparison.json",
        json_bytes({"input_hashes": {str(refs): hash_file(refs)}}),
    )
    mistral = tmp_path / "mistral"
    write_atomic(mistral / "settings.json", json_bytes({"model": "fictional/model"}))
    write_atomic(
        mistral / "analysis_run_1/analysis.json",
        json_bytes(
            {
                "question_metrics": [
                    {"question_id": q.question_id, "status": "success" if i < 12 else "failed_api"}
                    for i, q in enumerate(questions)
                ]
            }
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "finrag.evaluation.local_reranker_experiment.sealed_inputs", lambda path: ({}, inputs)
    )
    monkeypatch.setattr(
        "finrag.evaluation.local_reranker_experiment.score_cached_outputs",
        lambda *args: {
            "predictions": [
                {
                    "question_id": q.question_id,
                    "status": "success" if i < 12 else "failed_api",
                    "metrics": {
                        m: False for m in ("hit@1", "hit@5", "complete_excerpt_coverage@5")
                    },
                }
                for i, q in enumerate(questions)
            ]
        },
    )
    config = {"device": "cpu"}
    cache = tmp_path / "local_cache"
    record = {
        "status": "success",
        "input_id": input_identity(inputs[0]),
        "configuration": config,
        "ranking": rank_scores(inputs[0], [1.0] * 50),
        "seconds": 1.0,
    }
    write_atomic(cache / f"{local_key(inputs[0], config)}.json", json_bytes(record))
    report = evaluate(experiment, cache, config, mistral)
    assert report["questions"] == 28 and report["local_28"]["hit@5"]["denominator"] == 28
    assert report["local_matched12"]["hit@5"]["denominator"] == 12
    assert report["baseline_matched12"]["hit@5"]["denominator"] == 12
    assert len(report["matched_mistral_question_ids"]) == 12
    assert sum(r["status"] == "not_attempted" for r in report["rows"]) == 27


def test_fake_model_assets_reject_revision_and_hash_drift(tmp_path, monkeypatch) -> None:
    from finrag.retrieval import local_reranker_assets as assets

    for name in assets.FILES:
        (tmp_path / name).write_text("offline fixture", encoding="utf-8")
        metadata = tmp_path / ".cache/huggingface/download" / (name + ".metadata")
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(assets.REVISION + "\netag\n0", encoding="utf-8")
    (tmp_path / "config.json").write_bytes(
        json_bytes(
            {
                "architectures": ["XLMRobertaForSequenceClassification"],
                "max_position_embeddings": 8194,
                "pad_token_id": 1,
                "id2label": {"0": "LABEL_0"},
            }
        )
    )
    (tmp_path / "tokenizer_config.json").write_bytes(json_bytes({"model_max_length": 8192}))
    monkeypatch.setattr(assets, "WEIGHT_SHA256", hash_file(tmp_path / "model.safetensors"))
    manifest = assets.seal_model(tmp_path)
    assert assets.validate_model(tmp_path) == manifest
    (tmp_path / "tokenizer.json").write_text("drift", encoding="utf-8")
    with pytest.raises(FinragError, match="manifest differ"):
        assets.validate_model(tmp_path)
    (tmp_path / ".cache/huggingface/download/config.json.metadata").write_text(
        "wrong revision", encoding="utf-8"
    )
    with pytest.raises(FinragError, match="revision differs"):
        assets.seal_model(tmp_path)
