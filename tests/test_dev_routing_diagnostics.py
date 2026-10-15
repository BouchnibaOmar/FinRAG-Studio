from pathlib import Path
from types import SimpleNamespace

import pytest
from test_retrieval_metrics import evaluation_question

from finrag.config import FinragError
from finrag.evaluation.dev_routing_diagnostics import (
    ORACLE_COMMAND_SOURCE,
    aggregate,
    breakdown,
    diagnostic_rows,
    distinct_pages,
    historical_check,
    prefix_hit,
    ranking_diagnostic,
    run_diagnostic,
    validate_ranking,
)


def hits(pages: list[tuple[str, int]]) -> list[dict]:
    return [
        {
            "document_id": document,
            "physical_page_number": page,
            "rank": rank,
            "chunk_id": f"chunk-{rank}",
        }
        for rank, (document, page) in enumerate(pages, 1)
    ]


def test_page_deduplication_preserves_first_appearance_and_document() -> None:
    saved = hits([("a", 1), ("a", 1), ("b", 1), ("a", 2), ("b", 1)])
    assert [hit["rank"] for hit in distinct_pages(saved)] == [1, 3, 4]


def test_duplicate_chunks_can_hide_a_reference_from_historical_top5() -> None:
    saved = hits([("a", 1)] * 5 + [("a", 2), ("a", 3), ("a", 4), ("a", 5)])
    row = ranking_diagnostic(saved, {("a", 2)}, "a")
    assert row["historical_hit5"] is False
    assert row["distinct_page_hit5"] is True
    assert row["first_reference_chunk_rank"] == 6
    assert row["first_reference_distinct_page_rank"] == 2
    assert row["saved_distinct_page_depth"] == 5


def test_same_page_number_in_wrong_document_is_not_a_hit() -> None:
    saved = hits([("wrong", page) for page in range(1, 6)])
    row = ranking_diagnostic(saved, {("expected", 1)}, "expected")
    assert row["historical_hit5"] is False
    assert row["distinct_page_hit5"] is False
    assert row["expected_document_present_top5"] is False
    assert row["oracle_chunk_hit5"] is None
    assert row["first_reference_distinct_page_rank"] is None


@pytest.mark.parametrize("depth", [0, 1, 4])
def test_incomplete_negative_prefix_is_unknown(depth: int) -> None:
    saved = hits([("a", page) for page in range(1, depth + 1)])
    assert prefix_hit(saved, {("a", 100)}) is None


def test_incomplete_positive_is_known_but_not_a_complete_five_page_case() -> None:
    row = ranking_diagnostic(hits([("a", 9)]), {("a", 9)}, "a")
    summary = aggregate([row])["distinct_page_hit5"]
    assert row["distinct_page_hit5"] is True
    assert summary["complete_prefix_only"] == {
        "numerator": 0,
        "denominator": 0,
        "rate": None,
    }
    assert summary["observed_positive_incomplete"] == 1
    assert summary["unknown_incomplete"] == 0


def test_oracle_filters_before_truncation_and_page_deduplication() -> None:
    saved = hits(
        [("other", page) for page in range(1, 6)]
        + [("a", 1), ("a", 1), ("a", 2), ("a", 3), ("a", 4), ("a", 5)]
    )
    row = ranking_diagnostic(saved, {("a", 4)}, "a")
    assert row["historical_hit5"] is False
    assert row["expected_document_present_top5"] is False
    assert row["oracle_saved_chunk_depth"] == 6
    assert row["oracle_saved_distinct_page_depth"] == 5
    assert row["oracle_chunk_hit5"] is True
    assert row["oracle_page_hit5"] is True
    assert row["oracle_first_reference_chunk_rank"] == 5
    assert row["oracle_first_reference_page_rank"] == 4


def test_oracle_does_not_make_a_truncated_absence_exhaustive() -> None:
    row = ranking_diagnostic(hits([("other", 1), ("a", 1)]), {("a", 2)}, "a")
    assert row["oracle_chunk_hit5"] is None
    assert row["oracle_first_reference_chunk_rank"] is None
    assert row["absence_meaning"].startswith("not observed")


def test_group_counts_and_complete_case_denominators() -> None:
    full = ranking_diagnostic(hits([("a", p) for p in range(1, 6)]), {("a", 1)}, "a")
    partial = ranking_diagnostic(hits([("b", 1)]), {("b", 7)}, "b")
    rows = [
        {**full, "language": "fr", "expected_document": "a.pdf"},
        {**partial, "language": "en", "expected_document": "b.pdf"},
    ]
    groups = breakdown(rows)
    assert groups["overall"]["historical_hit5"]["numerator"] == 1
    assert groups["overall"]["historical_hit5"]["denominator"] == 2
    assert groups["overall"]["distinct_page_hit5"]["complete_prefix_only"]["denominator"] == 1
    assert groups["overall"]["oracle_page_hit5"]["unknown_incomplete"] == 1
    assert groups["language_document:en / b.pdf"]["historical_hit5"]["denominator"] == 1


def test_historical_verification_excludes_unanswerable_and_rejects_drift() -> None:
    rows = [
        {"question_id": "q1", "answerable": True, "historical_hit5": True},
        {"question_id": "q2", "answerable": False, "historical_hit5": False},
    ]
    assert historical_check(rows, {"q1": 1})["denominator"] == 1
    with pytest.raises(FinragError, match="Historical Hit@5 differs"):
        historical_check(rows, {"q1": 0})


def test_invalid_rank_order_and_duplicate_chunk_ids_fail() -> None:
    saved = hits([("a", 1), ("a", 2)])
    with pytest.raises(FinragError, match="consecutive"):
        validate_ranking(list(reversed(saved)))
    saved[1]["chunk_id"] = saved[0]["chunk_id"]
    with pytest.raises(FinragError, match="duplicate"):
        validate_ranking(saved)


def test_existing_output_is_refused_before_reading_any_inputs(tmp_path: Path) -> None:
    output = tmp_path / "artifacts/private/existing"
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FinragError, match="existing output is preserved"):
        run_diagnostic(tmp_path, output)
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_proposed_oracle_script_is_syntax_checked_without_execution() -> None:
    compile(ORACLE_COMMAND_SOURCE, "oracle_dense_cache_propose.py", "exec")


def test_nonanswerable_rows_have_depth_but_no_hit_or_oracle_metric() -> None:
    chunks = [SimpleNamespace(document_id="sha256:" + "a" * 64)]
    question = evaluation_question(chunks, answerable=False)
    saved = {"retrieved": hits([(question.document_id, 1)])}
    answerable, excluded = diagnostic_rows(
        [question], [saved], {question.document_id: "fictional.pdf"}
    )
    assert answerable == []
    assert len(excluded) == 1 and excluded[0]["metrics"] is None
    assert excluded[0]["saved_chunk_depth"] == 1
    assert "historical_hit5" not in excluded[0]


def test_invalid_prefix_length_fails() -> None:
    with pytest.raises(FinragError, match="must be positive"):
        prefix_hit([], set(), k=0)
