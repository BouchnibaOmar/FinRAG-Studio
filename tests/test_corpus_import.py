from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import pdf_bytes
from test_bm25 import lexical_config

from finrag.config import FinragError
from finrag.corpus.import_corpus import ReportSpec, import_corpus, validate_corpus
from finrag.corpus.prepare_corpus import (
    load_chunk_cache,
    prepare_chunks,
    prepare_extraction,
    validate_extraction,
)
from finrag.ingestion.hashing import hash_bytes, hash_file


@pytest.fixture
def corpus(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    pdf = pdf_bytes(["Fictional revenue profit CET1", "", "Fictional audited earnings"])
    (source / "fictional.pdf").write_bytes(pdf)
    digest = hash_bytes(pdf)
    spec = ReportSpec(
        document_id=f"sha256:{digest}",
        source_sha256=digest,
        source_filename="fictional.pdf",
        company="Fictional bank",
        historical_physical_pages=3,
        historical_text_pages=2,
    )
    output = tmp_path / "data/private/corpus/originals"
    manifest = import_corpus(source, output, tmp_path, reports=[spec])
    return source, output / "corpus_manifest.json", manifest.reports[0]


def test_corpus_import_hashes_and_overwrite(corpus, tmp_path: Path) -> None:
    source, manifest_path, report = corpus
    assert (
        hash_file(source / "fictional.pdf")
        == hash_file(Path(report.destination_path))
        == report.source_sha256
    )
    before = (source / "fictional.pdf").read_bytes()
    assert validate_corpus(manifest_path).reports[0] == report
    with pytest.raises(FinragError, match="overwrite"):
        import_corpus(source, manifest_path.parent, tmp_path, reports=[report])
    assert (source / "fictional.pdf").read_bytes() == before


def test_source_hash_mismatch_has_no_destination(corpus, tmp_path: Path) -> None:
    source, _, report = corpus
    (source / "fictional.pdf").write_bytes(b"changed")
    destination = tmp_path / "data/private/corpus/rejected"
    with pytest.raises(FinragError, match="source hash mismatch"):
        import_corpus(source, destination, tmp_path, reports=[report])
    assert not destination.exists()


def test_native_extraction_and_chunk_cache_integrity(corpus, tmp_path: Path) -> None:
    _, manifest_path, report = corpus
    extracted = tmp_path / "data/private/corpus/extracted"
    chunk_root = tmp_path / "data/private/corpus/chunks"
    first = prepare_extraction(manifest_path, extracted, tmp_path)
    assert first[0]["physical_pages"] == 3 and first[0]["blank_pages"] == 1
    assert first[0]["historical_counts_match"]
    with patch("finrag.corpus.prepare_corpus.load_pdf", side_effect=AssertionError("must reuse")):
        assert prepare_extraction(manifest_path, extracted, tmp_path)[0]["reused"]
    config = lexical_config(maximum_size=12, overlap=2)
    summaries = prepare_chunks(manifest_path, extracted, chunk_root, config, tmp_path)
    with patch(
        "finrag.corpus.prepare_corpus.chunk_pages", side_effect=AssertionError("must reuse")
    ):
        assert prepare_chunks(manifest_path, extracted, chunk_root, config, tmp_path) == summaries
    _, pages = validate_extraction(extracted / "fictional", report)
    chunks, stats = load_chunk_cache(
        chunk_root / "A/fictional", pages, hash_file(extracted / "fictional/pages.jsonl"), config
    )
    assert stats["blank_pages"] == [2]
    assert {chunk.physical_page_number for chunk in chunks} == {1, 3}
    with (chunk_root / "A/fictional/chunks.jsonl").open("ab") as output:
        output.write(b"\n")
    with pytest.raises(FinragError, match="hashes"):
        prepare_chunks(manifest_path, extracted, chunk_root, config, tmp_path)


def test_corrupt_extraction_is_not_recomputed(corpus, tmp_path: Path) -> None:
    _, manifest_path, _ = corpus
    extracted = tmp_path / "data/private/corpus/extracted"
    prepare_extraction(manifest_path, extracted, tmp_path)
    (extracted / "fictional/pages.jsonl").write_bytes(b"altered")
    with pytest.raises(FinragError, match="hashes"):
        prepare_extraction(manifest_path, extracted, tmp_path)
