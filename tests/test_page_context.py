import pytest
from pydantic import ValidationError
from test_bm25 import synthetic_chunks
from test_retrieval_metrics import evaluation_question

from finrag.config import FinragError
from finrag.domain.documents import PageRecord
from finrag.domain.evaluation import Evidence
from finrag.evaluation.page_context_experiment import context_coverage, context_summary, run
from finrag.retrieval.page_context import CONTEXT_CHARACTER_LIMIT, ContextPassage, build_context


def fixture(text: str, number: int = 1):
    chunk = synthetic_chunks(["seed"])[0]
    identity = {
        key: getattr(chunk, key) for key in ("document_id", "source_filename", "source_sha256")
    }
    page = PageRecord(
        **identity,
        physical_page_number=number,
        text=text,
        is_blank=False,
        extraction_status="extracted",
    )
    passage = ContextPassage(
        **identity, physical_page_number=number, text=text[:4], chunk_id=f"chunk_{number}"
    )
    return page, passage


def test_pages_keep_exact_text_and_first_bm25_appearance_not_numeric_order() -> None:
    page2, p2 = fixture("seed\nsecond \u2028page", 2)
    page1, p1 = fixture("seed first page", 1)
    pages = {(p.document_id, p.physical_page_number): p for p in (page1, page2)}
    overlap = p2.model_copy(update={"chunk_id": "another", "text": "second"})
    result = build_context([p2, overlap, p1], pages, "pages")
    assert [u["physical_page_number"] for u in result.units] == [2, 1]
    assert result.units[0]["text"] == page2.text
    assert result.text == page2.text + "\n\n" + page1.text
    assert result.represented_chunk_ids == [p2.chunk_id, "another", p1.chunk_id]
    assert len(context_summary(result)["pages"]) == 2


@pytest.mark.parametrize(
    "field,value", [("text", "not in page"), ("source_filename", "different.pdf")]
)
def test_provenance_or_text_mismatch_is_explicit(field, value) -> None:
    page, passage = fixture("seed page")
    with pytest.raises(FinragError, match="mismatch"):
        build_context(
            [passage.model_copy(update={field: value})], {(page.document_id, 1): page}, "pages"
        )


def test_budget_stops_without_cutting_or_skipping_next_page() -> None:
    a, pa = fixture("a" * 19_999, 1)
    b, pb = fixture("bb", 2)
    c, pc = fixture("c", 3)
    pages = {(p.document_id, p.physical_page_number): p for p in (a, b, c)}
    context = build_context([pa, pb, pc], pages, "pages")
    assert context.text == a.text and len(context.text) == 19_999
    assert [u["physical_page_number"] for u in context.omitted_units] == [2, 3]
    assert context.omitted_units[1]["reason"] == "not_added_after_budget_stop"
    assert not context.first_unit_exceeds_limit
    exact, passage = fixture("x" * CONTEXT_CHARACTER_LIMIT)
    assert len(build_context([passage], {(exact.document_id, 1): exact}, "pages").text) == 20_000


def test_oversized_first_page_produces_empty_context_not_truncated_text() -> None:
    page, passage = fixture("x" * 20_001)
    context = build_context([passage], {(page.document_id, 1): page}, "pages")
    assert context.text == "" and context.first_unit_exceeds_limit
    assert context.omitted_units[0]["text_characters"] == 20_001


def test_passages_use_same_budget_and_keep_overlaps_as_separate_units() -> None:
    page, passage = fixture("x" * 20_000)
    first = passage.model_copy(update={"text": "x" * 10_000})
    second = passage.model_copy(update={"chunk_id": "second", "text": "x" * 10_000})
    context = build_context([first, second], {(page.document_id, 1): page}, "passages")
    assert len(context.text) == 10_000 and len(context.units) == 1
    assert context.stop_reason


def test_no_annotations_and_no_neighbor_pages() -> None:
    page, passage = fixture("seed page")
    neighbor, _ = fixture("neighbor", 2)
    with pytest.raises(ValidationError):
        ContextPassage.model_validate({**passage.model_dump(), "relevant_pages": [2]})
    context = build_context(
        [passage], {(page.document_id, 1): page, (neighbor.document_id, 2): neighbor}, "pages"
    )
    assert [u["physical_page_number"] for u in context.units] == [1]


def test_expansion_can_recover_exact_evidence_without_certifying_financial_answer() -> None:
    page, passage = fixture("seed and omitted evidence")
    question = evaluation_question(synthetic_chunks(["seed"])).model_copy(
        update={
            "document_id": page.document_id,
            "evidence": [Evidence(physical_page_number=1, text="omitted evidence")],
        }
    )
    pages = {(page.document_id, 1): page}
    assert not context_coverage(question, build_context([passage], pages, "passages"))[
        "complete_excerpt_coverage"
    ]
    assert context_coverage(question, build_context([passage], pages, "pages"))[
        "complete_excerpt_coverage"
    ]
    with pytest.raises(FinragError, match="dev"):
        context_coverage(
            question.model_copy(update={"split": "test"}), build_context([passage], pages, "pages")
        )


def test_existing_output_is_not_overwritten(tmp_path) -> None:
    output = tmp_path / "artifacts/private/existing"
    output.mkdir(parents=True)
    with pytest.raises(FinragError, match="overwrite"):
        run(tmp_path, tmp_path / "missing", tmp_path / "missing", output)
