from pathlib import Path

import pytest
from pydantic import ValidationError

from finrag.chunking.page_chunker import chunk_pages
from finrag.config import FinragError
from finrag.domain.chunks import ChunkingConfig
from finrag.domain.documents import PageRecord
from finrag.ingestion.hashing import hash_bytes
from finrag.ingestion.pdf_loader import load_pdf


def test_chunks_preserve_page_and_document_metadata(synthetic_pdf: Path) -> None:
    _, pages = load_pdf(synthetic_pdf)
    config = ChunkingConfig(maximum_size=8, overlap=2)
    chunks = chunk_pages(pages, config)
    assert chunks == chunk_pages(pages, config)
    assert {chunk.physical_page_number for chunk in chunks} == {1, 3}
    assert [(chunk.physical_page_number, chunk.chunk_index_on_page) for chunk in chunks] == sorted(
        (chunk.physical_page_number, chunk.chunk_index_on_page) for chunk in chunks
    )
    for page in (pages[0], pages[2]):
        local = [
            chunk for chunk in chunks if chunk.physical_page_number == page.physical_page_number
        ]
        assert local[0].text + "".join(chunk.text[2:] for chunk in local[1:]) == page.text
        for chunk in local:
            assert chunk.document_id == page.document_id
            assert chunk.source_sha256 == page.source_sha256
            assert chunk.source_filename == page.source_filename
            assert chunk.content_sha256 == hash_bytes(chunk.text.encode("utf-8"))
            assert chunk.chunking_config == config


def test_configuration_changes_identity_even_for_short_page(synthetic_pdf: Path) -> None:
    _, pages = load_pdf(synthetic_pdf)
    original = chunk_pages(pages, ChunkingConfig(maximum_size=100))[0]
    changed = chunk_pages(pages, ChunkingConfig(maximum_size=101))[0]
    overlap = chunk_pages(pages, ChunkingConfig(maximum_size=100, overlap=1))[0]
    assert original.text == changed.text == overlap.text
    assert len({original.chunk_id, changed.chunk_id, overlap.chunk_id}) == 3


def test_unicode_character_windows_preserve_exact_text() -> None:
    digest = hash_bytes(b"synthetic")
    page = PageRecord(
        document_id=f"sha256:{digest}",
        source_sha256=digest,
        source_filename="fiction.pdf",
        physical_page_number=1,
        text="€été\n🙂value",
        is_blank=False,
        extraction_status="extracted",
    )
    chunks = chunk_pages([page], ChunkingConfig(maximum_size=3))
    assert "".join(chunk.text for chunk in chunks) == page.text
    changed = PageRecord(**{**page.model_dump(), "text": "€été\n🙂other"})
    assert (
        chunk_pages([changed], ChunkingConfig(maximum_size=3))[-1].chunk_id != chunks[-1].chunk_id
    )


@pytest.mark.parametrize(
    "values",
    [
        {"maximum_size": 0},
        {"maximum_size": 5, "overlap": 5},
        {"overlap": -1},
        {"unit": "mpnet_tokens"},
        {"maximum_size": "5"},
    ],
)
def test_invalid_chunking_config(values: dict) -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(**values)


def test_whitespace_windows_preserve_page_text() -> None:
    digest = hash_bytes(b"synthetic")
    page = PageRecord(
        document_id=f"sha256:{digest}",
        source_sha256=digest,
        source_filename="fiction.pdf",
        physical_page_number=1,
        text="start" + " " * 20 + "end",
        is_blank=False,
        extraction_status="extracted",
    )
    chunks = chunk_pages([page], ChunkingConfig(maximum_size=5))
    assert any(not chunk.text.strip() for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == page.text


def test_missing_blank_or_reordered_pages_are_rejected(synthetic_pdf: Path) -> None:
    _, pages = load_pdf(synthetic_pdf)
    for records in ([pages[0], pages[2]], list(reversed(pages)), []):
        with pytest.raises(FinragError, match="physical page"):
            chunk_pages(records, ChunkingConfig())
