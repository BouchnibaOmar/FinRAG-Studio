import pytest
from test_bm25 import synthetic_chunks
from test_retrieval_metrics import evaluation_question

from finrag.config import FinragError
from finrag.domain.evaluation import Evidence
from finrag.evaluation.bge_depth_context import (
    aggregate,
    bounded_context,
    depth_metrics,
    passage_volume,
    render_passage,
)
from finrag.evaluation.document_oracle_metrics import evaluate_candidates


def hit(identifier: str, text: str, rank: int = 1) -> dict:
    return {
        "chunk_id": identifier,
        "text": text,
        "rank": rank,
        "document_id": "d",
        "source_filename": "é.pdf",
        "physical_page_number": 1,
    }


def test_unicode_header_and_separators_are_part_of_exact_budget() -> None:
    first, second = hit("a", "€é"), hit("b", "texte", 2)
    expected = (
        '{"chunk_id":"a","document_id":"d","physical_page_number":1,"source_filename":"é.pdf"}\n€é'
    )
    assert render_passage(first) == expected
    assert len(expected.encode("utf-8")) > len(expected)
    assert bounded_context([first], len(expected))["kept_passages"] == 1
    assert bounded_context([first], len(expected) - 1)["kept_passages"] == 0
    exact = len(expected) + 2 + len(render_passage(second))
    result = bounded_context([first, second], exact)
    assert result["context"] == expected + "\n\n" + render_passage(second)
    assert result["characters"] == exact and not result["omitted_passages"]
    assert bounded_context([first, second], exact - 1)["kept_passages"] == 1


def test_stop_at_first_overflow_without_skipping_or_truncating() -> None:
    first, big, last = hit("a", "A"), hit("b", "X" * 100, 2), hit("c", "B", 3)
    budget = len(render_passage(first)) + 2 + len(render_passage(last))
    context = bounded_context([first, big, last], budget)
    assert context["context"] == render_passage(first)
    assert context["kept_chunk_ids"] == ["a"]
    assert [r["chunk_id"] for r in context["omitted_passages"]] == ["b", "c"]
    assert context["omitted_passages"][1]["reason"] == "after_budget_stop"
    assert bounded_context([big], 1)["first_passage_exceeds_budget"]


def test_page_duplicates_and_overlap_are_counted_without_text_changes() -> None:
    hits = [hit("a", "ABC"), hit("b", "BCD", 2)]
    volume = passage_volume(hits)
    assert volume["raw_text_characters"] == 6 and volume["distinct_physical_pages"] == 1
    context = bounded_context(hits)
    assert context["kept_passages"] == 2
    assert context["characters"] == volume["serialized_characters"]
    assert context["context"] == render_passage(hits[0]) + "\n\n" + render_passage(hits[1])


def test_depth20_multi_evidence_and_micro_recall_have_distinct_denominators() -> None:
    measured = {
        "candidate_depth": 50,
        "reference_page_ranks": {1: 8, 2: 18},
        "reference_excerpt_ranks": [8, 18],
    }
    at5, at10, at20 = [depth_metrics(measured, k) for k in (5, 10, 20)]
    assert not at5["hit"] and at10["hit"] and not at10["complete_excerpts"]
    assert at20["complete_pages"] and at20["complete_excerpts"]
    result = aggregate([at10, at20])
    assert result["complete_excerpts"] == {"numerator": 1, "denominator": 2}
    assert result["individual_excerpt_recall"] == {"numerator": 3, "denominator": 4}
    context_rows = [{k: v for k, v in row.items() if k != "hit"} for row in (at10, at20)]
    assert "hit" not in aggregate(context_rows, include_hit=False)
    assert not depth_metrics(measured, 0)["hit"]
    with pytest.raises(FinragError):
        depth_metrics(measured, 51)


def test_page_presence_does_not_imply_evidence_and_headers_are_not_evidence() -> None:
    chunks = synthetic_chunks(["AB", "CD"])
    question = evaluation_question(chunks).model_copy(
        update={"relevant_pages": [1], "evidence": [Evidence(physical_page_number=1, text="ABCD")]}
    )
    hits = [
        {**chunks[0].model_dump(), "source_filename": "ABCD.pdf"},
        {**chunks[1].model_dump(), "physical_page_number": 1},
    ]
    measured = evaluate_candidates(question, hits, True)
    value = depth_metrics(measured, 2)
    assert value["complete_pages"] and not value["complete_excerpts"]
    assert "ABCD" in bounded_context(hits)["context"]
    # The header occurrence and text split across units never enter reference-rank matching.
    assert value["individual_excerpts_found"] == 0


def test_added_depth_metrics_match_existing_evaluator_and_respect_missing_evidence() -> None:
    chunks = synthetic_chunks([f"passage {i}" for i in range(50)])
    question = evaluation_question(chunks).model_copy(
        update={
            "relevant_pages": [1, 2],
            "evidence": [
                Evidence(physical_page_number=1, text="passage 0"),
                Evidence(physical_page_number=2, text="missing literal"),
            ],
        }
    )
    measured = evaluate_candidates(question, [c.model_dump() for c in chunks], False)
    for k in (1, 5, 10, 50):
        value = depth_metrics(measured, k)
        assert value["hit"] == measured[f"hit@{k}"]
        assert value["complete_pages"] == measured[f"complete_page_coverage@{k}"]
        assert value["complete_excerpts"] == measured[f"complete_excerpt_coverage@{k}"]
        assert value["individual_excerpts_found"] == measured[f"reference_excerpts_found@{k}"]
    assert not depth_metrics(measured, 20)["complete_excerpts"]
