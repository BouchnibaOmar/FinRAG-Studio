from collections import defaultdict

from finrag.config import FinragError
from finrag.domain.dense import DensePrediction
from finrag.domain.evaluation import EvaluationQuestion
from finrag.domain.retrieval import LexicalPrediction

TOP_K = (1, 3, 5, 10)


def question_metrics(
    question: EvaluationQuestion, prediction: LexicalPrediction | DensePrediction
) -> dict[str, float] | None:
    if not question.answerable:
        return None
    expected = {(question.document_id, page) for page in question.relevant_pages}
    if not expected:
        raise FinragError("Answerable relevance must contain a physical page.")
    hits = prediction.retrieved[:10]
    first_rank = next(
        (hit.rank for hit in hits if (hit.document_id, hit.physical_page_number) in expected), None
    )
    result = {
        "mrr@10": 1 / first_rank if first_rank else 0.0,
        "document_accuracy@1": float(bool(hits) and hits[0].document_id == question.document_id),
    }
    for k in TOP_K:
        found = {(hit.document_id, hit.physical_page_number) for hit in hits[:k]} & expected
        result[f"hit@{k}"] = float(bool(found))
        result[f"page_recall@{k}"] = len(found) / len(expected)
    return result


def summarize(
    questions: list[EvaluationQuestion], predictions: list[LexicalPrediction | DensePrediction]
) -> dict:
    by_id = {prediction.question_id: prediction for prediction in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {
        question.question_id for question in questions
    }:
        raise FinragError("Predictions must match evaluation question IDs exactly once.")
    values = []
    for question in questions:
        prediction = by_id[question.question_id]
        if prediction.split != question.split or prediction.answerable != question.answerable:
            raise FinragError("Prediction split or answerability differs from the question.")
        result = question_metrics(question, prediction)
        if result is not None:
            values.append(result)
    return {
        "questions": len(questions),
        "answerable": len(values),
        "unanswerable": len(questions) - len(values),
        "metrics": {name: sum(row[name] for row in values) / len(values) for name in values[0]}
        if values
        else {},
    }


def grouped_metrics(
    questions: list[EvaluationQuestion], predictions: list[LexicalPrediction | DensePrediction]
) -> dict:
    overall = summarize(questions, predictions)
    by_id = {prediction.question_id: prediction for prediction in predictions}
    groups = defaultdict(list)
    for question in questions:
        for field in ("company", "language", "category"):
            groups[(field, getattr(question, field))].append(question)
        if question.answerable:
            groups[
                ("relevance", "multi_page" if len(question.relevant_pages) > 1 else "single_page")
            ].append(question)
    output = {"overall": overall}
    for (field, value), rows in groups.items():
        output.setdefault(field, {})[value] = summarize(
            rows, [by_id[row.question_id] for row in rows]
        )
    return output
