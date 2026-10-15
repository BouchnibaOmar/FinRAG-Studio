import json
from pathlib import Path

import pytest
from test_bm25 import lexical_config, synthetic_chunks

from finrag.domain.evaluation import EvaluationQuestion
from finrag.domain.retrieval import LexicalPrediction, PageTarget
from finrag.evaluation.retrieval_metrics import grouped_metrics, question_metrics, summarize
from finrag.evaluation.verify_lexical_metrics import verify_metric_groups
from finrag.retrieval.bm25 import BM25


def evaluation_question(chunks, answerable: bool = True) -> EvaluationQuestion:
    path = Path(__file__).resolve().parents[1] / "data/public_examples/evaluation_question.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row.update(
        document_id=chunks[0].document_id,
        relevant_pages=[1, 2] if answerable else [],
        evidence=[{"physical_page_number": page, "text": "fictional"} for page in (1, 2)]
        if answerable
        else [],
        evidence_bundle_pages=[1, 2] if answerable else [],
        answerable=answerable,
        category="numerical" if answerable else "unanswerable",
    )
    return EvaluationQuestion.model_validate(row)


def prediction(question, hits) -> LexicalPrediction:
    return LexicalPrediction(
        question_id=question.question_id,
        split="dev",
        configuration_id="A",
        answerable=question.answerable,
        expected_pages=[
            PageTarget(document_id=question.document_id, physical_page_number=page)
            for page in question.relevant_pages
        ],
        retrieved=hits,
    )


def test_multi_page_metrics_and_distinct_page_recall() -> None:
    chunks = synthetic_chunks(["profit profit", "profit", "other"])
    question = evaluation_question(chunks)
    hits = BM25(chunks, lexical_config()).search("profit")
    result = question_metrics(question, prediction(question, hits))
    assert result["hit@1"] == 1 and result["page_recall@1"] == 0.5
    assert result["page_recall@3"] == 1 and result["mrr@10"] == 1
    groups = grouped_metrics([question], [prediction(question, hits)])
    assert (
        verify_metric_groups(
            [question], [prediction(question, hits).model_dump(mode="json")], groups
        )
        == 5
    )
    assert (
        grouped_metrics([question], [prediction(question, hits)])["relevance"]["multi_page"][
            "answerable"
        ]
        == 1
    )


def test_wrong_document_same_page_is_not_correct() -> None:
    chunks = synthetic_chunks(["profit", "earnings"])
    question = evaluation_question(chunks)
    question = EvaluationQuestion.model_validate(
        {**question.model_dump(), "document_id": "sha256:" + "a" * 64}
    )
    hit = BM25(chunks, lexical_config()).search("profit")
    result = question_metrics(question, prediction(question, hit))
    assert result["hit@10"] == 0 and result["document_accuracy@1"] == 0


def test_unanswerable_is_excluded_without_refusal_score() -> None:
    chunks = synthetic_chunks(["profit"])
    question = evaluation_question(chunks, False)
    row = prediction(question, BM25(chunks, lexical_config()).search("profit"))
    assert question_metrics(question, row) is None
    assert summarize([question], [row]) == {
        "questions": 1,
        "answerable": 0,
        "unanswerable": 1,
        "metrics": {},
    }


def test_mrr_uses_first_relevant_rank() -> None:
    chunks = synthetic_chunks(["other", "profit", "profit profit"])
    question = evaluation_question(chunks)
    hits = BM25(chunks, lexical_config()).search("profit")
    assert hits[0].physical_page_number == 3
    result = question_metrics(question, prediction(question, hits))
    assert result["mrr@10"] == pytest.approx(0.5)
