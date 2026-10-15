import math

import pytest

from finrag.chunking.page_chunker import chunk_pages
from finrag.config import FinragError
from finrag.domain.chunks import ChunkingConfig
from finrag.domain.documents import PageRecord
from finrag.domain.retrieval import LexicalConfig
from finrag.ingestion.hashing import hash_bytes
from finrag.retrieval.bm25 import BM25, tokenize


def lexical_config(name: str = "A", maximum_size: int = 1000, overlap: int = 100) -> LexicalConfig:
    return LexicalConfig(
        configuration_id=name, chunking=ChunkingConfig(maximum_size=maximum_size, overlap=overlap)
    )


def synthetic_chunks(texts: list[str]):
    digest = hash_bytes(b"fictional-report")
    pages = [
        PageRecord(
            document_id=f"sha256:{digest}",
            source_sha256=digest,
            source_filename="fictional.pdf",
            physical_page_number=page,
            text=text,
            is_blank=not text.strip(),
            extraction_status="extracted" if text.strip() else "blank",
        )
        for page, text in enumerate(texts, start=1)
    ]
    return chunk_pages(pages, lexical_config().chunking)


def test_known_ranking_and_score_formula() -> None:
    chunks = synthetic_chunks(["profit profit", "profit unrelated unrelated unrelated", "nothing"])
    retriever = BM25(chunks, lexical_config())
    hits = retriever.search("profit")
    assert [hit.physical_page_number for hit in hits] == [1, 2]
    expected = (
        math.log1p((3 - 2 + 0.5) / (2 + 0.5)) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 2 / (7 / 3)))
    )
    assert hits[0].lexical_score == pytest.approx(expected)
    assert hits[0].rank == 1 and hits[1].rank == 2
    assert hits[0].lexical_score > hits[1].lexical_score > 0


def test_unicode_accents_numbers_abbreviations_and_punctuation() -> None:
    assert tokenize("ÉTÉ crédit CET1 ROTE 2025 12,5% 3,760 EUR/USD") == [
        "ete",
        "credit",
        "cet1",
        "rote",
        "2025",
        "12,5",
        "3,760",
        "eur",
        "usd",
    ]
    assert tokenize("cre\u0301dit") == tokenize("crédit") == tokenize("CREDIT")
    assert tokenize("０１２ ＣＥＴ１") == ["012", "cet1"]


def test_ties_use_full_chunk_id_independent_of_input_order() -> None:
    chunks = synthetic_chunks(["same words", "same words", "same words"])
    left = BM25(chunks, lexical_config()).search("words")
    right = BM25(list(reversed(chunks)), lexical_config()).search("words")
    assert left == right
    assert [hit.chunk_id for hit in left] == sorted(chunk.chunk_id for chunk in chunks)


@pytest.mark.parametrize("query", ["", "  ", "!!!", "absenttoken"])
def test_no_matching_terms_returns_no_arbitrary_hits(query: str) -> None:
    assert BM25(synthetic_chunks(["revenue"]), lexical_config()).search(query) == []


def test_empty_corpus_duplicate_chunks_and_invalid_k_are_rejected() -> None:
    with pytest.raises(FinragError, match="nonempty"):
        BM25([], lexical_config())
    chunks = synthetic_chunks(["profit"])
    with pytest.raises(FinragError, match="duplicate"):
        BM25(chunks * 2, lexical_config())
    with pytest.raises(FinragError, match="positive"):
        BM25(chunks, lexical_config()).search("profit", 0)
