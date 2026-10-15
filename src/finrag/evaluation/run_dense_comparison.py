from pathlib import Path

from finrag.config import FinragError, filesystem_operation, private_output
from finrag.domain.dense import DenseHit, DensePrediction
from finrag.domain.evaluation import EvaluationQuestion
from finrag.domain.retrieval import LexicalPrediction, PageTarget, RetrievalHit
from finrag.evaluation.dense_preflight import encoding_fingerprint, prepare_dense_inputs
from finrag.evaluation.retrieval_metrics import grouped_metrics, question_metrics
from finrag.evaluation.run_lexical_experiment import load_config
from finrag.evaluation.split_guard import load_dev_questions
from finrag.evaluation.verify_lexical_metrics import verify_metric_groups
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes, read_json, read_jsonl, write_atomic
from finrag.retrieval.bm25 import BM25
from finrag.retrieval.e5 import E5Encoder
from finrag.retrieval.embedding_cache import EmbeddingCache
from finrag.retrieval.faiss_process import search_in_separate_process
from finrag.retrieval.hybrid import reciprocal_rank_fusion


def lexical_hits(hits: list[RetrievalHit]) -> list[DenseHit]:
    return [
        DenseHit(
            **{
                key: value
                for key, value in hit.model_dump().items()
                if key not in ("lexical_score", "configuration_id")
            },
            score=hit.lexical_score,
            configuration_id="bm25-control-v1",
        )
        for hit in hits
    ]


def choose_method(groups: dict) -> str:
    order = ("bm25", "dense", "hybrid")
    return max(
        order,
        key=lambda name: (
            *(
                groups[name]["overall"]["metrics"][metric]
                for metric in ("hit@5", "mrr@10", "hit@1")
            ),
            -order.index(name),
        ),
    )


def diagnostics(question: EvaluationQuestion, hits: list[DenseHit | RetrievalHit]) -> dict:
    pairs = {(hit.document_id, hit.physical_page_number) for hit in hits[:10]}
    result = {
        "distinct_pages_top10": len(pairs),
        "duplicate_page_slots_top10": len(hits[:10]) - len(pairs),
    }
    for k in (5, 10):
        supported = sum(
            any(
                hit.document_id == question.document_id
                and hit.physical_page_number == evidence.physical_page_number
                and evidence.text in hit.text
                for hit in hits[:k]
            )
            for evidence in question.evidence
        )
        result[f"exact_evidence_excerpt_coverage@{k}"] = (
            supported / len(question.evidence) if question.evidence else None
        )
    return result


def public_comparison(questions: list[EvaluationQuestion], predictions: dict, groups: dict) -> dict:
    by_method = {name: {row.question_id: row for row in rows} for name, rows in predictions.items()}
    comparisons = []
    for question in questions:
        row = {
            "question_id": question.question_id,
            "answerable": question.answerable,
            "methods": {},
        }
        for name, mapping in by_method.items():
            prediction = mapping[question.question_id]
            row["methods"][name] = {
                "metrics": question_metrics(question, prediction),
                "diagnostic": diagnostics(question, prediction.retrieved),
            }
        comparisons.append(row)
    changes = {}
    for name in ("dense", "hybrid"):
        recovered, regressed = [], []
        for row in comparisons:
            if not row["answerable"]:
                continue
            control = row["methods"]["bm25"]["metrics"]["hit@5"]
            candidate = row["methods"][name]["metrics"]["hit@5"]
            if candidate > control:
                recovered.append(row["question_id"])
            if candidate < control:
                regressed.append(row["question_id"])
        changes[name] = {
            "recovered_control_failures": recovered,
            "regressions_against_control": regressed,
        }
    return {
        "split": "dev",
        "groups": groups,
        "selected_method": choose_method(groups),
        "comparison": comparisons,
        "changes": changes,
        "unanswerable_question_ids": [q.question_id for q in questions if not q.answerable],
        "evidence_diagnostic": "Exact excerpt containment after ranking; not answer accuracy",
        "limitations": [
            "Small development dataset; no generalisation claim",
            "No refusal-quality measurement",
            "Language is not an established cause of differences",
        ],
        "test_metrics_generated": False,
    }


def _write_or_validate(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FinragError(f"Existing result differs from validated inputs: {path.name}")
    else:
        write_atomic(path, content)


def run_comparison(
    model: Path,
    root: Path,
    device: str,
    batch_size: int,
    allow_long_encoding: bool = False,
    *,
    check_files: bool = False,
) -> dict:
    with filesystem_operation("resolve private experiment directory", root):
        root = private_output(root, Path.cwd())
    benchmark = read_json(root / "sample_benchmark.json")
    with filesystem_operation("load pinned local E5 model", model):
        encoder = E5Encoder(model, device)
    chunks, preflight = prepare_dense_inputs(encoder, root)
    fingerprint = encoding_fingerprint(encoder, preflight, batch_size)
    if benchmark["fingerprint"] != fingerprint:
        raise FinragError("Encoding settings differ from measured preflight.")
    cache = EmbeddingCache(root / "embeddings", fingerprint, len(chunks))
    if check_files:
        return {
            "startup_file_validation": "passed",
            "output_root": str(root),
            "completed_passage_rows": len(cache.completed),
            "total_passage_rows": len(chunks),
            "encoding_started": False,
        }
    if (
        benchmark["requires_manual_execution"]
        and len(cache.completed) < len(chunks)
        and not allow_long_encoding
    ):
        raise FinragError(
            "Measured encoding exceeds five minutes. Use the documented manual command."
        )
    vectors = cache.resume(
        [chunk.text for chunk in chunks],
        lambda texts: encoder.encode(texts, "passage"),
        batch_size,
        root / "encoding_progress.log",
    )
    dataset = Path("data/private/evaluation/finrag_v1")
    questions = load_dev_questions(dataset)
    query_fingerprint = {
        "encoding": fingerprint,
        "dev_sha256": hash_file(dataset / "dev.jsonl"),
        "kind": "query",
    }
    query_cache = EmbeddingCache(root / "queries", query_fingerprint, len(questions))
    queries = query_cache.resume(
        [question.question for question in questions],
        lambda texts: encoder.encode(texts, "query"),
        batch_size,
        root / "query_progress.log",
    )
    dense_candidates = search_in_separate_process(root, chunks, vectors, queries, fingerprint)
    bm25 = BM25(chunks, load_config(Path("experiments/phase2a/configs/B.json")))
    reference_path = Path("artifacts/private/phase2a/B/predictions.jsonl")
    if (
        hash_file(reference_path)
        != "5a025fdf2336b4d15044a5f0d3d58472ff4381d9d892ecff2289497e70614cc5"
    ):
        raise FinragError("Preserved Phase 2A B reference predictions changed.")
    reference = [LexicalPrediction.model_validate(row) for row in read_jsonl(reference_path)]
    predictions = {"bm25": [], "dense": [], "hybrid": [], "B_reference": reference}
    for question, dense in zip(questions, dense_candidates, strict=True):
        lexical = lexical_hits(bm25.search(question.question, top_k=50))
        ranked = {
            "bm25": lexical[:10],
            "dense": dense[:10],
            "hybrid": reciprocal_rank_fusion(lexical, dense),
        }
        for name, hits in ranked.items():
            predictions[name].append(
                DensePrediction(
                    question_id=question.question_id,
                    split="dev",
                    configuration_id=f"{name}-e5-v1",
                    method=name,
                    answerable=question.answerable,
                    expected_pages=[
                        PageTarget(document_id=question.document_id, physical_page_number=page)
                        for page in question.relevant_pages
                    ],
                    retrieved=hits,
                )
            )
    if preflight["B_identities_preserved"]:
        for control, original in zip(predictions["bm25"], reference, strict=True):
            if control.question_id != original.question_id or [
                (hit.chunk_id, hit.score) for hit in control.retrieved
            ] != [(hit.chunk_id, hit.lexical_score) for hit in original.retrieved]:
                raise FinragError(
                    "Unchanged-chunk BM25 control differs from preserved B reference."
                )
    groups = {name: grouped_metrics(questions, rows) for name, rows in predictions.items()}
    verified = {
        name: verify_metric_groups(
            questions, [row.model_dump(mode="json") for row in rows], groups[name]
        )
        for name, rows in predictions.items()
    }
    report = public_comparison(questions, predictions, groups)
    report.update(
        {
            "fingerprint": fingerprint,
            "independently_verified_groups": verified,
            "B_reference_predictions_sha256": hash_file(reference_path),
            "matched_control_uses_children": not preflight["B_identities_preserved"],
            "search_execution": (
                "isolated FAISS process; four threads; validated float32 NPY and JSON"
            ),
        }
    )
    for name in ("bm25", "dense", "hybrid"):
        _write_or_validate(
            root / name / "predictions.jsonl",
            jsonl_bytes([row.model_dump(mode="json") for row in predictions[name]]),
        )
    _write_or_validate(root / "comparison.json", json_bytes(report))
    _write_or_validate(Path("experiments/phase2b/reports/dev_results.json"), json_bytes(report))
    return {
        "split": "dev",
        "selected_method": report["selected_method"],
        "groups": groups,
        "independently_verified": True,
    }
