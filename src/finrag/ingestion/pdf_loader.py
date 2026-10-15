import io
from datetime import UTC, datetime
from pathlib import Path

import pypdf
from pypdf import PdfReader

from finrag.config import FinragError
from finrag.domain.documents import PageRecord
from finrag.domain.manifests import DocumentManifest
from finrag.ingestion.hashing import document_id, hash_bytes


def read_pdf(content: bytes, source_filename: str) -> PdfReader:
    if not content.startswith(b"%PDF-"):
        raise FinragError(f"{source_filename}: invalid PDF header.")
    try:
        reader = PdfReader(io.BytesIO(content), strict=True)
        if reader.is_encrypted:
            try:
                readable = reader.decrypt("") != 0
            except Exception as error:
                raise FinragError(
                    f"{source_filename}: encrypted PDF cannot be decrypted locally."
                ) from error
            if not readable:
                raise FinragError(f"{source_filename}: encrypted PDF requires a password.")
        if not reader.pages:
            raise FinragError(f"{source_filename}: PDF has zero physical pages.")
        return reader
    except FinragError:
        raise
    except Exception as error:
        raise FinragError(f"{source_filename}: PDF cannot be read locally.") from error


def extract_page_texts(content: bytes, source_filename: str) -> list[str]:
    reader = read_pdf(content, source_filename)
    texts = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            texts.append(page.extract_text() or "")
        except Exception as error:
            raise FinragError(
                f"{source_filename}: extraction failed on physical page {number}."
            ) from error
    return texts


def load_pdf(path: Path) -> tuple[DocumentManifest, list[PageRecord]]:
    content = path.read_bytes()
    digest = hash_bytes(content)
    identity = {
        "document_id": document_id(digest),
        "source_filename": path.name,
        "source_sha256": digest,
    }
    texts = extract_page_texts(content, path.name)
    pages = [
        PageRecord(
            **identity,
            physical_page_number=number,
            text=text,
            is_blank=not text.strip(),
            extraction_status="extracted" if text.strip() else "blank",
        )
        for number, text in enumerate(texts, start=1)
    ]
    text_count = sum(not page.is_blank for page in pages)
    if text_count == 0:
        raise FinragError(f"{path.name}: PDF has no extractable text; OCR is not implemented.")
    manifest = DocumentManifest(
        **identity,
        byte_size=len(content),
        physical_page_count=len(pages),
        text_page_count=text_count,
        blank_page_count=len(pages) - text_count,
        extractor_name="pypdf",
        extractor_version=pypdf.__version__,
        extraction_timestamp=datetime.now(UTC),
        extraction_status="extracted",
    )
    return manifest, pages
