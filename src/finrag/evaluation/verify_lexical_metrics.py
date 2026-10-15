import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from finrag.config import FinragError
from finrag.domain.evaluation import EvaluationQuestion
from finrag.evaluation.split_guard import load_dev_questions
from finrag.ingestion.manifest import read_json, read_jsonl


def independent_summary(questions: list[EvaluationQuestion], predictions: dict[str, dict]) -> dict:
    totals = defaultdict(Fraction)
    answerable = 0
    for question in questions:
        if not question.answerable:
            continue
        answerable += 1
        expected = {(question.document_id, page) for page in question.relevant_pages}
        hits = predictions[question.question_id]["retrieved"][:10]
        relevance = [(hit["document_id"], hit["physical_page_number"]) in expected for hit in hits]
        first = next((rank for rank, relevant in enumerate(relevance, start=1) if relevant), None)
        totals["mrr@10"] += Fraction(1, first) if first else Fraction(0)
        totals["document_accuracy@1"] += int(
            bool(hits) and hits[0]["document_id"] == question.document_id
        )
        for k in (1, 3, 5, 10):
            found = set()
            for hit in hits[:k]:
                pair = (hit["document_id"], hit["physical_page_number"])
                if pair in expected:
                    found.add(pair)
            totals[f"hit@{k}"] += int(bool(found))
            totals[f"page_recall@{k}"] += Fraction(len(found), len(expected))
    return {
        "questions": len(questions),
        "answerable": answerable,
        "unanswerable": len(questions) - answerable,
        "metrics": {name: float(value / answerable) for name, value in totals.items()}
        if answerable
        else {},
    }


def verify_metric_groups(
    questions: list[EvaluationQuestion], predictions: list[dict], groups: dict
) -> int:
    by_id = {row["question_id"]: row for row in predictions}
    if (
        len(by_id) != len(predictions)
        or set(by_id) != {question.question_id for question in questions}
        or any(row["split"] != "dev" for row in predictions)
    ):
        raise FinragError("Independent verification accepts matching dev predictions only.")
    subsets = {("overall", None): questions}
    for question in questions:
        for field in ("company", "language", "category"):
            subsets.setdefault((field, getattr(question, field)), []).append(question)
        if question.answerable:
            name = "multi_page" if len(question.relevant_pages) > 1 else "single_page"
            subsets.setdefault(("relevance", name), []).append(question)
    for (field, name), rows in subsets.items():
        actual = independent_summary(rows, by_id)
        stored = groups[field] if name is None else groups[field][name]
        if any(stored[key] != actual[key] for key in ("questions", "answerable", "unanswerable")):
            raise FinragError("Independent group counts differ from stored metrics.")
        if set(stored["metrics"]) != set(actual["metrics"]) or any(
            not math.isclose(stored["metrics"][key], value, rel_tol=0, abs_tol=1e-12)
            for key, value in actual["metrics"].items()
        ):
            raise FinragError("Independent rational page metrics differ from stored metrics.")
    return len(subsets)


def verify_runs(dataset: Path, artifact_root: Path) -> dict:
    questions = load_dev_questions(dataset)
    verified = {}
    for name in ("A", "B", "C"):
        directory = artifact_root / name
        predictions = read_jsonl(directory / "predictions.jsonl")
        manifest = read_json(directory / "experiment_manifest.json")
        verified[name] = verify_metric_groups(questions, predictions, manifest["groups"])
    return {
        "split": "dev",
        "verified_groups_per_configuration": verified,
        "test_metrics_generated": False,
    }
