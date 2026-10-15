import copy
import http.client
import io

import pytest
from pypdf import PdfWriter
from test_bm25 import lexical_config, synthetic_chunks
from test_candidate_reranking import fixture_input

from finrag.config import FileOperationError, FinragError
from finrag.evaluation.question_translation import OpenRouterTranslator
from finrag.ingestion.hashing import hash_bytes
from finrag.retrieval.candidate_reranking import OpenRouterReranker, input_identity
from finrag.retrieval.company_router import CatalogueDocument
from finrag.retrieval.local_reranker import LocalReranker, rank_scores
from finrag.retrieval.routed_bm25 import CachedQueryTranslation, RoutedBM25
from finrag.ui.resources import (
    FreeSearch,
    assemble_example,
    authorized_pdf_path,
    load_free_search,
    read_source_pdf,
)


def forbidden(*args, **kwargs):
    raise AssertionError("No network, translation, inference or extra mode loader permitted")


@pytest.fixture(autouse=True)
def no_model_or_api(monkeypatch):
    for cls in (OpenRouterTranslator, OpenRouterReranker, LocalReranker):
        monkeypatch.setattr(cls, "__init__", forbidden)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", forbidden)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", forbidden)


def search_fixture():
    chunks = synthetic_chunks(["net income profit", "capital capital", "unrelated"])
    document = CatalogueDocument(
        document_id=chunks[0].document_id,
        source_sha256=chunks[0].source_sha256,
        source_filename=chunks[0].source_filename,
        company_family="BNP Paribas",
        reporting_entity="BNP Paribas",
        language="en",
    )
    translation = CachedQueryTranslation(
        original_question="Quel résultat BNP Paribas ?",
        translated_question="net income profit",
        target_language="en",
        provenance="local_correction",
    )
    return FreeSearch(RoutedBM25(chunks, [document], lexical_config(), [translation]), []), document


def saved_fixture():
    source, mapping, hits, _, document = fixture_input(50)
    baseline = {
        "question_id": "synthetic_dev",
        "input_id": input_identity(source),
        "candidate_id_to_chunk_id": mapping,
        "hits": hits,
        "reference_answer": "DO NOT CONSULT",
        "metrics": object(),
    }
    route = {
        "query_used": source.question,
        "original_question": source.question,
        "hits": hits,
        "route": {"companies": ["BNP Paribas"]},
    }
    record = {
        "input_id": input_identity(source),
        "status": "success",
        "configuration": {"model": "saved-only"},
        "ranking": rank_scores(source, list(map(float, range(50)))),
    }
    return baseline, route, source, record, document


def test_exact_translation_and_synthetic_search_preserve_provenance() -> None:
    search, document = search_fixture()
    result = search.search("Quel résultat BNP Paribas ?")
    assert result.query_used == "net income profit"
    assert result.translation_status == "local_correction"
    hit = result.hits[0]
    chunk = search.search_engine.chunks[0]
    assert (hit.document_id, hit.physical_page_number, hit.text, hit.chunk_id) == (
        document.document_id,
        chunk.physical_page_number,
        chunk.text,
        chunk.chunk_id,
    )
    # Even a trailing space prevents reuse of the cached translation.
    changed = search.search("Quel résultat BNP Paribas ? ")
    assert changed.query_used == "Quel résultat BNP Paribas ? "
    incompatible = search.search_engine.translations[0].model_copy(update={"target_language": "fr"})
    search.search_engine.translations = [incompatible]
    assert search.search("Quel résultat BNP Paribas ?").query_used == "Quel résultat BNP Paribas ?"


def test_empty_question_and_no_results_are_different_from_missing_files(tmp_path) -> None:
    search, _ = search_fixture()
    with pytest.raises(FinragError, match="Saisissez"):
        search.search(" \n ")
    assert search.search("zzzzzzunknown").hits == []
    with pytest.raises(FileOperationError):
        load_free_search(tmp_path)


def test_precomputed_is_read_only_and_keeps_exact_passages() -> None:
    baseline, route, source, record, _ = saved_fixture()
    original = copy.deepcopy(baseline["hits"])
    example = assemble_example(baseline, route, source, record, [])
    assert example.query_used == source.question
    assert example.bm25 == original
    assert baseline["hits"] == original
    assert {h["chunk_id"] for h in example.bge} == {h["chunk_id"] for h in original}
    for hit in example.bge:
        old = next(h for h in original if h["chunk_id"] == hit["chunk_id"])
        assert all(hit[k] == old[k] for k in ("text", "document_id", "physical_page_number"))


@pytest.mark.parametrize("damage", ["text", "page", "query", "duplicate"])
def test_precomputed_rejects_inconsistent_saved_data(damage) -> None:
    baseline, route, source, record, _ = saved_fixture()
    if damage == "text":
        baseline["hits"][0]["text"] = "changed"
    elif damage == "page":
        baseline["hits"][0]["physical_page_number"] += 1
    elif damage == "query":
        route["query_used"] = "another query"
    else:
        record["ranking"][1] = record["ranking"][0]
    with pytest.raises(FinragError):
        assemble_example(baseline, route, source, record, [])


def pdf_fixture(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    data = buffer.getvalue()
    _, document = search_fixture()
    digest = hash_bytes(data)
    document = document.model_copy(
        update={
            "document_id": "sha256:" + digest,
            "source_sha256": digest,
            "source_filename": "bank.pdf",
        }
    )
    directory = tmp_path / "data/private/corpus/originals"
    directory.mkdir(parents=True)
    (directory / "bank.pdf").write_bytes(data)
    return document, data


def test_only_authorized_pdf_with_valid_hash_and_physical_page(tmp_path) -> None:
    document, data = pdf_fixture(tmp_path)
    assert read_source_pdf(tmp_path, [document], document.document_id, 2) == (data, "bank.pdf")
    with pytest.raises(FinragError, match="catalogue"):
        read_source_pdf(tmp_path, [document], "../../secret.pdf", 1)
    for page in (0, -1, True, 3):
        with pytest.raises(FinragError):
            read_source_pdf(tmp_path, [document], document.document_id, page)
    path = authorized_pdf_path(tmp_path, [document], document.document_id)
    path.write_bytes(b"different bytes")
    with pytest.raises(FinragError, match="empreinte"):
        read_source_pdf(tmp_path, [document], document.document_id, 1)
    path.unlink()
    with pytest.raises(FileOperationError):
        read_source_pdf(tmp_path, [document], document.document_id, 1)


@pytest.mark.parametrize(
    "filename",
    [
        "../bank.pdf",
        "..\\bank.pdf",
        "/tmp/bank.pdf",
        "C:\\private.pdf",
        "bank.pdf:secret",
        "file.txt",
    ],
)
def test_catalogue_cannot_authorize_arbitrary_paths(tmp_path, filename) -> None:
    _, document = search_fixture()
    document = document.model_copy(update={"source_filename": filename})
    with pytest.raises(FinragError, match="non autorisé"):
        authorized_pdf_path(tmp_path, [document], document.document_id)


def test_streamlit_modes_and_explicit_search(monkeypatch, tmp_path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from finrag.ui import app

    search, document = search_fixture()
    calls = []
    monkeypatch.setattr(app, "document_catalogue", lambda _: [document])
    monkeypatch.setattr(app, "search_resource", lambda _: calls.append("search") or search)
    monkeypatch.setattr(app, "saved_examples", forbidden)
    script = tmp_path / "ui_runner.py"
    script.write_text("from finrag.ui.app import main\nmain()", encoding="utf-8")
    ui = AppTest.from_file(str(script), default_timeout=10).run()
    assert not ui.exception and not calls
    ui.button(key="free_search_submit").click().run()
    assert ui.warning and not calls
    ui.text_area[0].input("net income BNP Paribas").run()
    assert not calls
    ui.button(key="free_search_submit").click().run()
    assert not ui.exception and calls == ["search"]
    assert any(t.value == "net income profit" for t in ui.text)
    baseline, route, source, record, _ = saved_fixture()
    example = assemble_example(baseline, route, source, record, [])
    monkeypatch.setattr(app, "saved_examples", lambda _: [example])
    monkeypatch.setattr(app, "search_resource", forbidden)
    ui.sidebar.radio[0].set_value("Explorer les exemples").run()
    assert not ui.exception
    assert any(i.value == app.PRECOMPUTED_NOTICE for i in ui.info)
    ui.radio(key="saved_method").set_value("BGE").run()
    assert not ui.exception
    assert any("BGE — logit" in c.value for c in ui.caption)


def test_streamlit_missing_files_are_technical_errors(monkeypatch, tmp_path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from finrag.ui import app

    def unavailable(_):
        raise FileNotFoundError("Catalogue local absent")

    monkeypatch.setattr(app, "document_catalogue", unavailable)
    script = tmp_path / "ui_runner.py"
    script.write_text("from finrag.ui.app import main\nmain()", encoding="utf-8")
    ui = AppTest.from_file(str(script), default_timeout=10).run()
    assert not ui.exception
    assert any("Erreur technique" in error.value for error in ui.error)


def test_streamlit_source_viewer_uses_only_local_pdf_bytes(monkeypatch, tmp_path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from finrag.ui import app

    document, _ = pdf_fixture(tmp_path)
    monkeypatch.setattr(app, "PROJECT", tmp_path)
    monkeypatch.setattr(app, "document_catalogue", lambda _: [document])
    script = tmp_path / "pdf_runner.py"
    script.write_text(
        "from finrag.ui import app\n"
        "catalogue = app.document_catalogue(str(app.PROJECT))\n"
        "doc = catalogue[0]\n"
        "app.source_viewer([{'document_id':doc.document_id, "
        "'source_filename':doc.source_filename, 'physical_page_number':2}], catalogue, 'test')\n",
        encoding="utf-8",
    )
    ui = AppTest.from_file(str(script), default_timeout=10).run()
    ui.button(key="test_open").click().run()
    assert not ui.exception and not ui.error
    assert any("Page physique du PDF : 2" in c.value for c in ui.caption)
    assert ui.get("download_button")
