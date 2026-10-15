import pytest
from test_bm25 import synthetic_chunks
from test_retrieval_metrics import evaluation_question

from finrag.config import FinragError
from finrag.domain.evaluation import Evidence
from finrag.evaluation.document_oracle_metrics import evaluate_candidates
from finrag.evaluation.saved_bm25_fusion_experiment import evidence_diagnosis, evidence_present, run
from finrag.retrieval.saved_bm25_fusion import fuse_saved_bm25


def hits(ids: list[str]) -> list[dict]:
    return [
        {
            "chunk_id": key,
            "rank": rank,
            "text": f"passage {key}",
            "document_id": "fictional",
            "physical_page_number": rank,
            "source_sha256": "source",
            "source_filename": "report.pdf",
        }
        for rank, key in enumerate(ids, 1)
    ]


def test_union_can_contain_100_and_rrf_keeps_only_50() -> None:
    original = hits([f"a{index:02}" for index in range(50)])
    translated = hits([f"b{index:02}" for index in range(50)])
    result = fuse_saved_bm25("original?", original, "translation?", translated)
    assert len(result.union) == 100 and len(result.ranked) == 50
    assert [h["chunk_id"] for h in result.ranked[:4]] == ["a00", "b00", "a01", "b01"]
    assert result.ranked[0]["rrf_score"] == 1 / 61
    assert result.independent_lists == 2


def test_shared_chunk_is_deduplicated_and_gets_both_rank_contributions() -> None:
    original = hits(["a", "b"])
    translated = [{**original[1], "rank": 1}, {**original[0], "rank": 2}]
    result = fuse_saved_bm25("original", original, "translated", translated)
    assert len(result.union) == 2
    assert result.ranked[0]["rrf_score"] == 1 / 61 + 1 / 62
    assert [h["chunk_id"] for h in result.ranked] == ["a", "b"]
    assert result.ranked[0]["component_ranks"] == {"original": 1, "cached_translation": 2}


@pytest.mark.parametrize("field,value", [("text", "changed"), ("physical_page_number", 999)])
def test_same_id_with_different_text_or_provenance_is_rejected(field, value) -> None:
    original = hits(["a"])
    with pytest.raises(FinragError, match="disagree"):
        fuse_saved_bm25("original", original, "translated", [{**original[0], field: value}])


def test_identical_query_counts_once_and_preserves_ranking() -> None:
    original = hits(["z", "a"])
    result = fuse_saved_bm25("same", original, "same", original)
    assert result.independent_lists == 1
    assert [h["chunk_id"] for h in result.ranked] == ["z", "a"]
    assert result.ranked[0]["rrf_score"] == 1 / 61
    with pytest.raises(FinragError, match="different rankings"):
        fuse_saved_bm25("same", original, "same", [{**original[1], "rank": 1}])


def test_duplicate_ids_and_invalid_saved_ranks_are_rejected() -> None:
    original = hits(["a"])
    for bad in ([*original, {**original[0], "rank": 2}], [{**original[0], "rank": 0}]):
        with pytest.raises(FinragError, match="contiguous"):
            fuse_saved_bm25("original", bad, "translated", [])


def test_multi_evidence_distinguishes_missing_eliminated_and_poorly_ranked() -> None:
    chunks = synthetic_chunks(["operand one", "operand two"])
    question = evaluation_question(chunks).model_copy(
        update={
            "relevant_pages": [c.physical_page_number for c in chunks],
            "evidence": [
                Evidence(physical_page_number=c.physical_page_number, text=c.text) for c in chunks
            ],
        }
    )
    passages = [c.model_dump() for c in chunks]
    assert evidence_present(question, passages) == [True, True]
    assert (
        evidence_diagnosis(question, passages[:1], passages[:1])["question_category"]
        == "incomplete_union"
    )
    lost = evidence_diagnosis(question, passages, passages[:1])
    assert lost["question_category"] == "complete_union_lost_in_rrf50"
    assert lost["per_reference_excerpt"] == ["in_rrf_top5", "eliminated_from_rrf50"]
    unrelated = {**passages[0], "text": "other text", "physical_page_number": 999}
    ranked = [passages[0], *[unrelated.copy() for _ in range(5)], passages[1]]
    metrics = evaluate_candidates(question, ranked, True)
    assert metrics["hit@5"] and not metrics["complete_excerpt_coverage@5"]
    assert metrics["complete_excerpt_coverage@50"]
    assert (
        evidence_diagnosis(question, passages, ranked)["question_category"]
        == "complete_rrf50_below_top5"
    )


def test_evaluation_excludes_test_and_unanswerable() -> None:
    chunks = synthetic_chunks(["proof"])
    question = evaluation_question(chunks)
    for changes in ({"split": "test"}, {"answerable": False}):
        with pytest.raises(FinragError, match="dev"):
            evidence_present(question.model_copy(update=changes), [])


def test_union_can_complete_evidence_missing_from_each_component() -> None:
    chunks = synthetic_chunks(["operand one", "operand two"])
    question = evaluation_question(chunks).model_copy(
        update={
            "relevant_pages": [c.physical_page_number for c in chunks],
            "evidence": [
                Evidence(physical_page_number=c.physical_page_number, text=c.text) for c in chunks
            ],
        }
    )
    original = [{**chunks[0].model_dump(), "rank": 1}]
    translated = [{**chunks[1].model_dump(), "rank": 1}]
    assert not all(evidence_present(question, original))
    assert not all(evidence_present(question, translated))
    result = fuse_saved_bm25("original", original, "translated", translated)
    assert all(evidence_present(question, result.union))
    assert evaluate_candidates(question, result.ranked, True)["complete_excerpt_coverage@5"]


def test_fusion_does_not_propagate_annotations_or_depend_on_them() -> None:
    original = hits(["a"])
    annotated = [{**original[0], "expected_document_id": "secret", "reference_answer": "secret"}]
    assert fuse_saved_bm25("original", original, "translated", original) == fuse_saved_bm25(
        "original", annotated, "translated", original
    )


def test_existing_output_is_protected_before_any_input_read(tmp_path) -> None:
    output = tmp_path / "artifacts/private/occupied"
    output.mkdir(parents=True)
    sentinel = output / "existing.json"
    sentinel.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FinragError, match="overwrite refused"):
        run(tmp_path, tmp_path / "missing-source", output)
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
