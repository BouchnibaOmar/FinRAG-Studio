from finrag.config import FinragError
from finrag.domain.evaluation import EvaluationQuestion
from finrag.evaluation.dev_routing_diagnostics import distinct_pages, page_key

TOP_K = (1, 5, 10, 50)


def evaluate_candidates(
    question: EvaluationQuestion, hits: list[dict], exhaustive_candidates: bool
) -> dict:
    if question.split != "dev" or not question.answerable:
        raise FinragError("Document oracle metrics require an answerable dev question.")
    expected = {(question.document_id, page) for page in question.relevant_pages}
    page_ranks = {
        page: next(
            (
                rank
                for rank, hit in enumerate(hits, 1)
                if page_key(hit) == (question.document_id, page)
            ),
            None,
        )
        for page in question.relevant_pages
    }
    evidence_ranks = [
        next(
            (
                rank
                for rank, hit in enumerate(hits, 1)
                if page_key(hit) == (question.document_id, evidence.physical_page_number)
                and evidence.text in hit["text"]
            ),
            None,
        )
        for evidence in question.evidence
    ]
    observed_pages = [rank for rank in page_ranks.values() if rank is not None]
    observed_evidence = [rank for rank in evidence_ranks if rank is not None]
    pages = distinct_pages(hits)
    result = {
        "candidate_depth": len(hits),
        "distinct_page_depth": len(pages),
        "exhaustive_candidates": exhaustive_candidates,
        "first_reference_segment_rank": min(observed_pages, default=None),
        "first_reference_distinct_page_rank": next(
            (rank for rank, hit in enumerate(pages, 1) if page_key(hit) in expected), None
        ),
        "first_supporting_excerpt_rank": min(observed_evidence, default=None),
        "all_reference_pages_rank": max(observed_pages)
        if len(observed_pages) == len(expected)
        else None,
        "all_reference_excerpts_rank": max(observed_evidence)
        if len(observed_evidence) == len(question.evidence)
        else None,
        "reference_page_ranks": page_ranks,
        "reference_excerpt_ranks": evidence_ranks,
        "reference_page_count": len(expected),
        "reference_excerpt_count": len(question.evidence),
    }
    for k in TOP_K:
        available = exhaustive_candidates or len(hits) >= k
        found_pages = sum(rank is not None and rank <= k for rank in page_ranks.values())
        found_excerpts = sum(rank is not None and rank <= k for rank in evidence_ranks)
        result[f"hit@{k}"] = bool(found_pages) if available else None
        result[f"reference_pages_found@{k}"] = found_pages
        result[f"reference_excerpts_found@{k}"] = found_excerpts
        result[f"complete_page_coverage@{k}"] = found_pages == len(expected) if available else None
        result[f"complete_excerpt_coverage@{k}"] = (
            found_excerpts == len(question.evidence) if available else None
        )
    result["distinct_page_hit@5"] = (
        any(page_key(hit) in expected for hit in pages[:5])
        if exhaustive_candidates or len(pages) >= 5
        else None
    )
    return result


def aggregate_metrics(rows: list[dict]) -> dict:
    keys = [
        f"{metric}@{k}"
        for metric in ("hit", "complete_page_coverage", "complete_excerpt_coverage")
        for k in TOP_K
    ]
    keys.append("distinct_page_hit@5")
    result = {}
    for key in keys:
        available = [row[key] for row in rows if row[key] is not None]
        result[key] = {
            "numerator": sum(available),
            "denominator": len(available),
            "unavailable": len(rows) - len(available),
        }
    return result
