import tempfile
from pathlib import Path
from statistics import mean, median

import pypdf

from finrag.chunking.page_chunker import chunk_pages
from finrag.config import FinragError, private_output
from finrag.corpus.import_corpus import ImportedReport, validate_corpus
from finrag.domain.chunks import ChunkRecord
from finrag.domain.documents import PageRecord
from finrag.domain.manifests import DocumentManifest
from finrag.domain.retrieval import LexicalConfig
from finrag.ingestion.hashing import hash_bytes, hash_file
from finrag.ingestion.manifest import (
    json_bytes,
    jsonl_bytes,
    publish_directory,
    read_json,
    read_jsonl,
    save_page_extraction,
    validate_page_sequence,
)
from finrag.ingestion.pdf_loader import load_pdf


def extraction_directory(root: Path, report: ImportedReport) -> Path:
    return root / Path(report.source_filename).stem


def validate_extraction(
    directory: Path, report: ImportedReport
) -> tuple[DocumentManifest, list[PageRecord]]:
    cache = read_json(directory / "extraction_cache.json")
    if (
        cache.get("source_sha256") != report.source_sha256
        or cache.get("extractor_version") != pypdf.__version__
        or cache.get("pipeline_version") != "finrag-pages-v1"
        or cache.get("pages_sha256") != hash_file(directory / "pages.jsonl")
        or cache.get("manifest_sha256") != hash_file(directory / "document_manifest.json")
    ):
        raise FinragError("Extraction cache has mismatched inputs, versions or output hashes.")
    manifest = DocumentManifest.model_validate_json(
        (directory / "document_manifest.json").read_bytes()
    )
    pages = [PageRecord.model_validate(row) for row in read_jsonl(directory / "pages.jsonl")]
    validate_page_sequence(pages)
    if (
        manifest.document_id != report.document_id
        or manifest.source_filename != report.source_filename
        or manifest.source_sha256 != report.source_sha256
        or manifest.byte_size != report.byte_size
        or manifest.physical_page_count != len(pages)
        or manifest.text_page_count != sum(not page.is_blank for page in pages)
        or any(
            page.document_id != report.document_id or page.source_filename != report.source_filename
            for page in pages
        )
    ):
        raise FinragError("Extraction manifest and physical page records disagree.")
    return manifest, pages


def prepare_extraction(
    corpus_manifest: Path,
    output_root: Path,
    project_root: Path,
    filename: str | None = None,
) -> list[dict]:
    corpus = validate_corpus(corpus_manifest)
    output_root = private_output(output_root, project_root)
    reports = [
        report
        for report in corpus.reports
        if filename is None or report.source_filename == filename
    ]
    if not reports:
        raise FinragError("Requested extraction document is not in the private corpus manifest.")
    output_root.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for report in reports:
        destination = extraction_directory(output_root, report)
        reused = destination.exists()
        if not reused:
            manifest, pages = load_pdf(Path(report.destination_path))
            if manifest.source_sha256 != report.source_sha256:
                raise FinragError("PDF identity changed before extraction.")
            with tempfile.TemporaryDirectory(prefix=".finrag-", dir=output_root) as temporary:
                staged = Path(temporary) / "extraction"
                save_page_extraction(manifest, pages, staged)
                cache = {
                    "source_sha256": report.source_sha256,
                    "extractor_version": pypdf.__version__,
                    "pipeline_version": "finrag-pages-v1",
                    "pages_sha256": hash_file(staged / "pages.jsonl"),
                    "manifest_sha256": hash_file(staged / "document_manifest.json"),
                }
                (staged / "extraction_cache.json").write_bytes(json_bytes(cache))
                validate_extraction(staged, report)
                publish_directory(staged, destination, False)
        manifest, _ = validate_extraction(destination, report)
        diagnostics.append(
            {
                "source_filename": report.source_filename,
                "document_id": report.document_id,
                "physical_pages": manifest.physical_page_count,
                "text_pages": manifest.text_page_count,
                "blank_pages": manifest.blank_page_count,
                "reused": reused,
                "historical_physical_pages": report.historical_physical_pages,
                "historical_text_pages": report.historical_text_pages,
                "historical_counts_match": (
                    manifest.physical_page_count == report.historical_physical_pages
                    and manifest.text_page_count == report.historical_text_pages
                ),
            }
        )
    return diagnostics


def validate_chunk_records(
    pages: list[PageRecord], chunks: list[ChunkRecord], config: LexicalConfig
) -> None:
    validate_page_sequence(pages)
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise FinragError("Duplicate chunk IDs in lexical corpus.")
    config_hash = hash_bytes(json_bytes(config.chunking.model_dump(mode="json")))
    position = 0
    for page in pages:
        if page.is_blank:
            continue
        start, index = 0, 0
        while start < len(page.text):
            if position >= len(chunks):
                raise FinragError("Chunk output loses nonblank page text.")
            chunk = chunks[position]
            end = min(start + config.chunking.maximum_size, len(page.text))
            text = page.text[start:end]
            digest = hash_bytes(text.encode("utf-8"))
            expected_id = (
                f"{page.document_id}:p{page.physical_page_number}:c{index}"
                f":{config.chunking.version}:{config_hash}:{digest}"
            )
            if (
                chunk.chunk_id != expected_id
                or chunk.text != text
                or chunk.content_sha256 != digest
                or chunk.document_id != page.document_id
                or chunk.source_filename != page.source_filename
                or chunk.source_sha256 != page.source_sha256
                or chunk.physical_page_number != page.physical_page_number
                or chunk.chunk_index_on_page != index
                or chunk.chunking_config != config.chunking
            ):
                raise FinragError(
                    "Chunk identity, text coverage or page provenance is inconsistent."
                )
            position += 1
            if end == len(page.text):
                break
            start += config.chunking.maximum_size - config.chunking.overlap
            index += 1
    if position != len(chunks):
        raise FinragError("Chunk output includes extra or blank-page chunks.")


def load_chunk_cache(
    directory: Path,
    pages: list[PageRecord],
    pages_hash: str,
    config: LexicalConfig,
) -> tuple[list[ChunkRecord], dict]:
    manifest = read_json(directory / "chunk_manifest.json")
    if (
        manifest.get("configuration") != config.model_dump(mode="json")
        or manifest.get("input_pages_sha256") != pages_hash
        or manifest.get("output_chunks_sha256") != hash_file(directory / "chunks.jsonl")
    ):
        raise FinragError("Chunk cache has mismatched configuration or input/output hashes.")
    chunks = [ChunkRecord.model_validate(row) for row in read_jsonl(directory / "chunks.jsonl")]
    validate_chunk_records(pages, chunks, config)
    if (
        manifest.get("chunk_count") != len(chunks)
        or manifest.get("document_id") != pages[0].document_id
        or manifest.get("source_sha256") != pages[0].source_sha256
        or manifest.get("source_filename") != pages[0].source_filename
        or manifest.get("nonblank_page_characters")
        != sum(len(page.text) for page in pages if not page.is_blank)
        or manifest.get("chunk_characters_with_overlap") != sum(len(chunk.text) for chunk in chunks)
        or manifest.get("blank_pages")
        != [page.physical_page_number for page in pages if page.is_blank]
        or manifest.get("size_distribution") != size_distribution(chunks)
    ):
        raise FinragError("Chunk manifest statistics do not match validated output.")
    return chunks, manifest


def size_distribution(chunks: list[ChunkRecord]) -> dict:
    lengths = [len(chunk.text) for chunk in chunks]
    if not lengths:
        return {"minimum": 0, "maximum": 0, "mean": 0.0, "median": 0.0}
    return {
        "minimum": min(lengths),
        "maximum": max(lengths),
        "mean": mean(lengths),
        "median": median(lengths),
    }


def prepare_chunks(
    corpus_manifest: Path,
    extraction_root: Path,
    output_root: Path,
    config: LexicalConfig,
    project_root: Path,
) -> list[dict]:
    corpus = validate_corpus(corpus_manifest)
    output_root = private_output(output_root, project_root) / config.configuration_id
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for report in corpus.reports:
        source = extraction_directory(extraction_root, report)
        _, pages = validate_extraction(source, report)
        pages_hash = hash_file(source / "pages.jsonl")
        destination = extraction_directory(output_root, report)
        if not destination.exists():
            chunks = chunk_pages(pages, config.chunking)
            validate_chunk_records(pages, chunks, config)
            with tempfile.TemporaryDirectory(prefix=".finrag-", dir=output_root) as temporary:
                staged = Path(temporary) / "chunks"
                staged.mkdir()
                (staged / "chunks.jsonl").write_bytes(
                    jsonl_bytes([chunk.model_dump(mode="json") for chunk in chunks])
                )
                manifest = {
                    "configuration": config.model_dump(mode="json"),
                    "document_id": report.document_id,
                    "source_filename": report.source_filename,
                    "source_sha256": report.source_sha256,
                    "input_pages_sha256": pages_hash,
                    "output_chunks_sha256": hash_file(staged / "chunks.jsonl"),
                    "chunk_count": len(chunks),
                    "size_distribution": size_distribution(chunks),
                    "blank_pages": [page.physical_page_number for page in pages if page.is_blank],
                    "nonblank_page_characters": sum(
                        len(page.text) for page in pages if not page.is_blank
                    ),
                    "chunk_characters_with_overlap": sum(len(chunk.text) for chunk in chunks),
                }
                (staged / "chunk_manifest.json").write_bytes(json_bytes(manifest))
                load_chunk_cache(staged, pages, pages_hash, config)
                publish_directory(staged, destination, False)
        _, manifest = load_chunk_cache(destination, pages, pages_hash, config)
        summaries.append(manifest)
    return summaries
