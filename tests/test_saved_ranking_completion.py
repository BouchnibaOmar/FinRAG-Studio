from pathlib import Path

import pytest

from finrag.config import FinragError
from finrag.evaluation.saved_ranking_completion import (
    complete_saved_rankings,
    filtered_observation,
    observation_summary,
    operand_excerpt_coverage,
    page_positions,
    proposed_set_observation,
)


def hits(pages: list[tuple[str, int]]) -> list[dict]:
    return [
        {
            "document_id": document,
            "physical_page_number": page,
            "rank": rank,
            "chunk_id": f"chunk-{rank}",
            "text": "",
        }
        for rank, (document, page) in enumerate(pages, 1)
    ]


def test_segment_global_page_and_filtered_ranks_are_distinct() -> None:
    saved = hits([("other", 1), ("other", 1), ("a", 1), ("a", 1), ("a", 2)])
    position = page_positions(saved, ("a", 2))
    assert position["first_segment_rank"] == 5
    assert position["global_distinct_page_rank"] == 3
    assert position["filtered_segment_rank"] == 3
    assert position["filtered_distinct_page_rank"] == 2
    row = filtered_observation(saved, "a", [2])
    assert row["observed_hit5"] is True
    assert row["observed_hit10"] is True
    assert row["incomplete_for_hit5"] is True
    assert row["available_document_pages"] == 2


def test_incomplete_observed_negative_is_not_an_exhaustive_miss() -> None:
    row = filtered_observation(hits([("a", 1)]), "a", [2])
    assert row["observed_hit5"] is False
    assert row["observation_state5"] == "incomplete_no_reference_observed"
    assert row["first_reference_filtered_page_rank"] is None
    assert row["saved_ranking_is_nonexhaustive"] is True


def test_hit10_sufficient_cohort_differs_from_hit5_cohort() -> None:
    rows = [
        {
            "question_id": "q1",
            **filtered_observation(hits([("a", p) for p in range(1, 6)]), "a", [1]),
        },
        {
            "question_id": "q2",
            **filtered_observation(hits([("a", p) for p in range(1, 11)]), "a", [9]),
        },
        {"question_id": "q3", **filtered_observation(hits([("a", 1)]), "a", [1])},
    ]
    summary = observation_summary(rows)
    assert summary["hit5"]["observed_all_questions"]["numerator"] == 2
    assert summary["hit5"]["sufficient_saved_pages_only"]["denominator"] == 2
    assert summary["hit10"]["observed_all_questions"]["numerator"] == 3
    assert summary["hit10"]["sufficient_saved_pages_only"]["denominator"] == 1
    assert summary["hit10"]["eligible_question_ids"] == ["q2"]
    assert summary["hit5"]["observed_positive_incomplete"] == 1


def test_empty_filtered_ranking_remains_in_all_question_denominator() -> None:
    row = {"question_id": "q1", **filtered_observation(hits([("other", 1)]), "a", [1])}
    summary = observation_summary([row])
    assert summary["hit5"]["observed_all_questions"] == {
        "numerator": 0,
        "denominator": 1,
        "rate": 0,
    }
    assert summary["hit5"]["sufficient_saved_pages_only"]["rate"] is None


def test_all_segment_occurrences_are_preserved_without_duplicate_pages() -> None:
    saved = hits([("a", 1), ("a", 1), ("a", 2), ("a", 1)])
    position = page_positions(saved, ("a", 1))
    assert position["segment_ranks"] == [1, 2, 4]
    assert position["filtered_distinct_page_rank"] == 1
    assert filtered_observation(saved, "a", [1])["available_document_pages"] == 2


def test_proposed_comparison_one_operand_and_complete_pair_are_not_scored() -> None:
    partial = proposed_set_observation(hits([("a", 501)]), "a", [484, 501])
    complete = proposed_set_observation(hits([("a", 501), ("a", 484)]), "a", [484, 501])
    assert partial["observed_page_count"] == 1
    assert partial["all_proposed_pages_observed"] is False
    assert complete["all_proposed_pages_observed"] is True
    assert complete["counted_as_metric_success"] is False
    assert complete["proposal_validation_status"] == "unvalidated"


def test_page_presence_does_not_imply_operand_excerpt_presence() -> None:
    saved = hits([("a", 484), ("a", 501), ("other", 484)])
    saved[1]["text"] = "NET INCOME GROUP SHARE\n8,754 8,640"
    saved[2]["text"] = "NET INCOME GROUP SHARE 7,074 7,087"
    rows = {
        484: ("sa", "NET INCOME GROUP SHARE 7,074 7,087"),
        501: ("group", "NET INCOME GROUP SHARE 8,754 8,640"),
    }
    coverage = operand_excerpt_coverage(saved, "a", rows)
    assert coverage["operand_scopes_observed"] == ["group"]
    assert coverage["both_proposed_operand_excerpts_observed"] is False
    saved[0]["text"] = rows[484][1]
    assert operand_excerpt_coverage(saved, "a", rows)["both_proposed_operand_excerpts_observed"]


def test_wrong_document_same_page_is_not_an_alternative() -> None:
    observation = proposed_set_observation(hits([("other", 501)]), "a", [501])
    assert observation["observed_page_count"] == 0


def test_existing_output_is_preserved_before_inputs_are_read(tmp_path: Path) -> None:
    output = tmp_path / "artifacts/private/completion"
    output.mkdir(parents=True)
    with pytest.raises(FinragError, match="new directory"):
        complete_saved_rankings(tmp_path, output)
