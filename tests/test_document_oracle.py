import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_bm25 import lexical_config, synthetic_chunks
from test_dense_retrieval import model_chunks
from test_retrieval_metrics import evaluation_question, prediction

from finrag.config import FinragError
from finrag.evaluation.document_oracle_metrics import aggregate_metrics, evaluate_candidates
from finrag.evaluation.document_oracle_worker import document_rows, run_worker
from finrag.evaluation.retrieval_metrics import question_metrics
from finrag.evaluation.run_document_oracle import CachedOracleInputs, run_oracle
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes, read_json, write_atomic
from finrag.retrieval.bm25 import BM25


def saved_hits(document: str, pages_and_text: list[tuple[int, str]]) -> list[dict]:
    return [
        {"rank": rank, "document_id": document, "physical_page_number": page, "text": text}
        for rank, (page, text) in enumerate(pages_and_text, 1)
    ]


def test_page_hit_does_not_imply_supporting_excerpt() -> None:
    chunks = synthetic_chunks(["fictional", "fictional"])
    question = evaluation_question(chunks)
    hits = saved_hits(question.document_id, [(1, "unrelated"), (1, "fictional"), (2, "fictional")])
    metrics = evaluate_candidates(question, hits, True)
    assert metrics["hit@1"] is True
    assert metrics["reference_excerpts_found@1"] == 0
    assert metrics["first_reference_segment_rank"] == 1
    assert metrics["first_supporting_excerpt_rank"] == 2
    assert metrics["complete_page_coverage@1"] is False
    assert metrics["complete_excerpt_coverage@5"] is True
    assert metrics["all_reference_excerpts_rank"] == 3


def test_comparative_requires_all_pages_and_excerpts_for_full_coverage() -> None:
    question = evaluation_question(synthetic_chunks(["fictional", "fictional"]))
    metrics = evaluate_candidates(
        question, saved_hits(question.document_id, [(1, "fictional")]), True
    )
    assert metrics["hit@50"] is True
    assert metrics["complete_page_coverage@50"] is False
    assert metrics["complete_excerpt_coverage@50"] is False
    assert metrics["all_reference_pages_rank"] is None


def test_wrong_document_or_alternative_page_does_not_cover_v1_excerpt() -> None:
    question = evaluation_question(synthetic_chunks(["fictional", "fictional"]))
    hits = saved_hits("sha256:" + "b" * 64, [(1, "fictional")])
    hits += saved_hits(question.document_id, [(9, "fictional")])
    metrics = evaluate_candidates(question, hits, True)
    assert metrics["hit@5"] is False
    assert metrics["reference_excerpts_found@50"] == 0


def test_full_oracle_rank_beyond_saved_top50_is_not_censored() -> None:
    question = evaluation_question(synthetic_chunks(["fictional", "fictional"]))
    hits = saved_hits(question.document_id, [(9, "other")] * 50 + [(1, "fictional")])
    full = evaluate_candidates(question, hits, True)
    limited = evaluate_candidates(question, hits[:50], False)
    assert full["first_reference_segment_rank"] == 51
    assert full["first_reference_distinct_page_rank"] == 2
    assert full["hit@50"] is False
    assert full["distinct_page_hit@5"] is True
    assert limited["first_reference_segment_rank"] is None


def test_saved_top10_does_not_claim_global_hit50() -> None:
    question = evaluation_question(synthetic_chunks(["fictional", "fictional"]))
    metrics = evaluate_candidates(
        question, saved_hits(question.document_id, [(1, "fictional")]), False
    )
    summary = aggregate_metrics([metrics])
    assert metrics["hit@50"] is None
    assert summary["hit@50"]["denominator"] == 0
    assert summary["hit@50"]["unavailable"] == 1


def test_historical_definition_is_reproduced_and_bm25_does_not_pad_zero_scores() -> None:
    chunks = synthetic_chunks(["fictional profit", "fictional loss", "unrelated"])
    question = evaluation_question(chunks)
    candidates = BM25(chunks, lexical_config()).search("profit", top_k=50)
    historical = question_metrics(question, prediction(question, candidates))
    metrics = evaluate_candidates(
        question, [hit.model_dump(mode="json") for hit in candidates], True
    )
    assert len(candidates) == 1
    for k in (1, 5, 10):
        assert metrics[f"hit@{k}"] == bool(historical[f"hit@{k}"])


def test_document_selection_preserves_original_vector_rows() -> None:
    chunks = model_chunks(["fictional one", "fictional two", "fictional three"])
    chunks[1] = chunks[1].model_copy(update={"document_id": "sha256:" + "b" * 64})
    assert document_rows(chunks)[chunks[0].document_id] == [0, 2]


def test_worker_refuses_torch_before_loading_faiss(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(FinragError, match="must not import Torch"):
        run_worker(tmp_path / "not-read.json")


def test_output_refused_before_cache_or_test_access(tmp_path: Path) -> None:
    output = tmp_path / "artifacts/private/keep"
    output.mkdir(parents=True)
    with pytest.raises(FinragError, match="new private output"):
        run_oracle(tmp_path, output)


def test_unanswerable_is_rejected_by_metrics() -> None:
    question = evaluation_question(synthetic_chunks(["fictional"]), False)
    with pytest.raises(FinragError, match="answerable dev"):
        evaluate_candidates(question, [], True)


def test_estimate_over_five_minutes_stops_before_any_search(monkeypatch, tmp_path: Path) -> None:
    cached = CachedOracleInputs(
        [],
        [],
        {},
        {},
        {},
        {
            "estimated_search_seconds": 301,
            "five_minute_gate_passed": False,
        },
    )
    monkeypatch.setattr(
        "finrag.evaluation.run_document_oracle.verify_cached_inputs", lambda root: cached
    )

    def forbidden_search(*args):
        raise AssertionError("Search must not start above the five-minute gate.")

    monkeypatch.setattr("finrag.evaluation.run_document_oracle.dense_in_worker", forbidden_search)
    output = tmp_path / "artifacts/private/gated"
    result = run_oracle(tmp_path, output)
    assert result["search_started"] is False
    assert "--allow-long-search" in result["manual_command"]
    assert (output / "preflight.json").is_file()
    assert not (output / "oracle_request.json").exists()


def test_worker_refuses_changed_input_before_loading_vectors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "data/private/input"
    output = tmp_path / "artifacts/private/output"
    source.mkdir(parents=True)
    output.mkdir(parents=True)
    path = source / "changed.jsonl"
    path.write_text("changed", encoding="utf-8")
    request = output / "request.json"
    write_atomic(
        request,
        json_bytes(
            {
                "output": str(output),
                "input_root": str(source),
                "inputs": {"chunks": {"path": str(path), "sha256": "0" * 64}},
            }
        ),
    )
    with pytest.raises(FinragError, match="path/hash mismatch"):
        run_worker(request)


def test_worker_rejects_output_outside_project_private_roots(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    request = tmp_path / "request.json"
    write_atomic(request, json_bytes({"output": str(tmp_path / "public_output")}))
    with pytest.raises(FinragError, match="Output must be inside"):
        run_worker(request)


def test_real_faiss_worker_searches_only_reference_document_without_torch(tmp_path: Path) -> None:
    source = tmp_path / "data/private/source"
    output = tmp_path / "artifacts/private/new_oracle"
    source.mkdir(parents=True)
    output.mkdir(parents=True)
    chunks = model_chunks(["fictional profit", "fictional loss", "wrong document"])
    chunks[2] = chunks[2].model_copy(
        update={
            "document_id": "sha256:" + "b" * 64,
            "source_sha256": "b" * 64,
            "source_filename": "wrong.pdf",
            "chunk_id": "wrong-doc-chunk",
        }
    )
    question = evaluation_question(chunks)
    passages = np.zeros((3, 384), dtype=np.float32)
    passages[0, 0] = passages[2, 0] = 1
    passages[1, 1] = 1
    queries = passages[:1].copy()
    write_atomic(source / "chunks.jsonl", jsonl_bytes([c.model_dump(mode="json") for c in chunks]))
    np.save(source / "passages.npy", passages, allow_pickle=False)
    np.save(source / "queries.npy", queries, allow_pickle=False)
    inputs = {
        name: {"path": str(source / filename), "sha256": hash_file(source / filename)}
        for name, filename in (
            ("chunks", "chunks.jsonl"),
            ("passages", "passages.npy"),
            ("queries", "queries.npy"),
        )
    }
    request = output / "request.json"
    write_atomic(
        request,
        json_bytes(
            {
                "output": str(output),
                "input_root": str(source),
                "inputs": inputs,
                "cached_query_count": 1,
                "questions": [{"query_row": 0, "question": question.model_dump(mode="json")}],
            }
        ),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "finrag.evaluation.document_oracle_worker",
            "--request",
            str(request),
        ],
        capture_output=True,
        cwd=tmp_path,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = read_json(output / "dense_worker_results.json")
    assert result["worker_torch_loaded"] is False
    assert result["results"][0]["document_passages_searched"] == 2
    assert len(result["results"][0]["hits"]) == 2
    assert all(hit["document_id"] == question.document_id for hit in result["results"][0]["hits"])
    assert result["results"][0]["hits"][0]["physical_page_number"] == 1
    assert result["results"][0]["metrics"]["complete_excerpt_coverage@50"] is True
    assert {name: hash_file(Path(record["path"])) for name, record in inputs.items()} == {
        name: record["sha256"] for name, record in inputs.items()
    }
    assert json.loads(completed.stdout)["worker_torch_loaded"] is False
