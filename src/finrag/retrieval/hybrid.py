from collections import defaultdict

from finrag.config import FinragError
from finrag.domain.dense import DenseHit


def reciprocal_rank_fusion(
    bm25: list[DenseHit], dense: list[DenseHit], top_k: int = 10
) -> list[DenseHit]:
    if top_k <= 0:
        raise FinragError("Hybrid top_k must be positive.")
    scores = defaultdict(float)
    ranks, sources = {}, {}
    for name, candidates in (("bm25", bm25[:50]), ("dense", dense[:50])):
        seen = set()
        for hit in candidates:
            if hit.chunk_id in seen:
                continue
            if hit.chunk_id in sources and (
                sources[hit.chunk_id].document_id != hit.document_id
                or sources[hit.chunk_id].physical_page_number != hit.physical_page_number
                or sources[hit.chunk_id].text != hit.text
            ):
                raise FinragError("Hybrid components disagree on chunk provenance.")
            seen.add(hit.chunk_id)
            scores[hit.chunk_id] += 1 / (60 + hit.rank)
            ranks.setdefault(hit.chunk_id, {})[name] = hit.rank
            sources[hit.chunk_id] = hit
    identifiers = sorted(scores, key=lambda identifier: (-scores[identifier], identifier))[:top_k]
    return [
        DenseHit.model_validate(
            {
                **sources[identifier].model_dump(),
                "rank": rank,
                "score": scores[identifier],
                "configuration_id": "hybrid-rrf60-v1",
                "component_ranks": ranks[identifier],
            }
        )
        for rank, identifier in enumerate(identifiers, start=1)
    ]
