from dataclasses import dataclass

from finrag.config import FinragError

PASSAGE_FIELDS = (
    "chunk_id",
    "document_id",
    "physical_page_number",
    "source_filename",
    "source_sha256",
    "text",
)


@dataclass(frozen=True)
class SavedFusion:
    union: list[dict]
    ranked: list[dict]
    independent_lists: int


def fuse_saved_bm25(
    original_query: str,
    original_hits: list[dict],
    translated_query: str,
    translated_hits: list[dict],
) -> SavedFusion:
    passages, component_ranks = {}, {}
    for name, hits in (("original", original_hits), ("cached_translation", translated_hits)):
        if len(hits) > 50:
            raise FinragError("Saved BM25 fusion accepts at most 50 hits per component.")
        seen = set()
        for rank, hit in enumerate(hits, 1):
            identifier = hit["chunk_id"]
            if identifier in seen or hit["rank"] != rank:
                raise FinragError("Saved BM25 ranks must be contiguous with unique chunk IDs.")
            seen.add(identifier)
            passage = {key: hit[key] for key in PASSAGE_FIELDS}
            if identifier in passages and passages[identifier] != passage:
                raise FinragError(f"Saved BM25 components disagree on passage {identifier}.")
            passages[identifier] = passage
            component_ranks.setdefault(identifier, {})[name] = rank
    same_query = original_query == translated_query
    if same_query and [h["chunk_id"] for h in original_hits] != [
        h["chunk_id"] for h in translated_hits
    ]:
        raise FinragError(
            "Identical saved queries have different rankings; inspect their provenance."
        )
    scores = {
        identifier: sum(
            1 / (60 + rank) for name, rank in ranks.items() if not same_query or name == "original"
        )
        for identifier, ranks in component_ranks.items()
    }
    identifiers = sorted(passages, key=lambda key: (-scores[key], key))[:50]
    ranked = [
        {
            **passages[identifier],
            "rank": rank,
            "rrf_score": scores[identifier],
            "component_ranks": component_ranks[identifier],
        }
        for rank, identifier in enumerate(identifiers, 1)
    ]
    return SavedFusion(
        union=[passages[key] for key in sorted(passages)],
        ranked=ranked,
        independent_lists=1 if same_query else 2,
    )
