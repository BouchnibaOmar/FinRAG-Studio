import io
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import pdf_bytes
from pypdf import PdfReader, PdfWriter

from finrag.config import FinragError
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.pdf_loader import load_pdf


def test_blank_pages_keep_physical_positions(synthetic_pdf: Path) -> None:
    manifest, pages = load_pdf(synthetic_pdf)
    assert [page.physical_page_number for page in pages] == [1, 2, 3]
    assert [page.is_blank for page in pages] == [False, True, False]
    assert manifest.physical_page_count == 3
    assert manifest.text_page_count == 2
    assert manifest.blank_page_count == 1
    assert manifest.document_id == f"sha256:{hash_file(synthetic_pdf)}"
    assert all(page.source_filename == synthetic_pdf.name for page in pages)
    assert ":\\" not in pages[0].model_dump_json()
    assert pages == load_pdf(synthetic_pdf)[1]


@pytest.mark.parametrize("content", [b"not PDF", b"%PDF-1.4\ninvalid"])
def test_invalid_pdf_is_rejected(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(content)
    with pytest.raises(FinragError, match="PDF"):
        load_pdf(path)


@pytest.mark.parametrize("page_count,reason", [(0, "zero physical"), (2, "OCR")])
def test_empty_and_textless_pdfs_are_rejected(tmp_path: Path, page_count: int, reason: str) -> None:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=100, height=100)
    path = tmp_path / "unusable.pdf"
    writer.write(path)
    with pytest.raises(FinragError, match=reason):
        load_pdf(path)


@pytest.mark.parametrize("password,readable", [("private-password", False), ("", True)])
def test_encrypted_pdf_behaviour(tmp_path: Path, password: str, readable: bool) -> None:
    reader = PdfReader(io.BytesIO(pdf_bytes(["Synthetic encrypted text"])))
    writer = PdfWriter()
    writer.append(reader)
    writer.encrypt(password)
    path = tmp_path / "encrypted.pdf"
    writer.write(path)
    if readable:
        assert load_pdf(path)[0].text_page_count == 1
    else:
        with pytest.raises(FinragError, match="encrypted PDF"):
            load_pdf(path)


def test_extraction_failure_is_not_silently_blank(synthetic_pdf: Path) -> None:
    with patch("pypdf._page.PageObject.extract_text", side_effect=RuntimeError("corruption")):
        with pytest.raises(FinragError, match="physical page 1"):
            load_pdf(synthetic_pdf)


def test_unsupported_encryption_has_clear_reason(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(pdf_bytes(["Synthetic text"]))))
    writer.encrypt("")
    path = tmp_path / "unsupported_encryption.pdf"
    writer.write(path)
    with patch(
        "finrag.ingestion.pdf_loader.PdfReader.decrypt", side_effect=RuntimeError("unsupported")
    ):
        with pytest.raises(FinragError, match="encrypted PDF cannot be decrypted"):
            load_pdf(path)
