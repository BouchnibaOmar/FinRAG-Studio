import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_dense_retrieval import model_chunks

from finrag.config import FinragError
from finrag.ingestion.manifest import json_bytes, read_json
from finrag.retrieval.faiss_process import search_in_separate_process
from finrag.retrieval.vector_index import DenseIndex, normalize_vectors


def inputs():
    chunks = model_chunks(["fictional revenue", "fictional profit"])
    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[0, 0], vectors[1, 1] = 1, 1
    query = normalize_vectors(np.ones((1, 384), dtype=np.float32), 384)
    return chunks, vectors, query


def test_real_worker_stable_search_persistence_and_no_torch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chunks, vectors, query = inputs()
    root = tmp_path / "artifacts/private/synthetic"
    # The parent may contain Torch; it never imports FAISS for this operation.
    monkeypatch.setitem(sys.modules, "torch", object())
    hits = search_in_separate_process(root, chunks, vectors, query, {"revision": "synthetic"})
    assert [hit.chunk_id for hit in hits[0]] == sorted(chunk.chunk_id for chunk in chunks)
    assert hits[0][0].score == pytest.approx(1 / np.sqrt(384))
    result = read_json(root / "faiss_exchange/results.json")
    assert result["worker_torch_loaded"] is False and result["threads"] == 4
    assert (
        search_in_separate_process(root, chunks, vectors, query, {"revision": "synthetic"}) == hits
    )
    with pytest.raises(FinragError, match="fingerprint"):
        search_in_separate_process(root, chunks, vectors, query, {"revision": "changed"})


def test_exchange_rejects_nonfinite_vectors_and_modified_arrays(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chunks, vectors, query = inputs()
    root = tmp_path / "artifacts/private/synthetic"
    corrupt = vectors.copy()
    corrupt[0, 0] = float("nan")
    with pytest.raises(FinragError, match="finite"):
        search_in_separate_process(root, chunks, corrupt, query, {})
    search_in_separate_process(root, chunks, vectors, query, {})
    with (root / "faiss_exchange/passages.npy").open("wb") as output:
        np.save(output, vectors[::-1], allow_pickle=False)
    with pytest.raises(FinragError, match="differs"):
        search_in_separate_process(root, chunks, vectors, query, {})


def test_dense_index_refuses_torch_in_process_before_native_call(monkeypatch):
    chunks, vectors, _ = inputs()
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(FinragError, match="isolated worker"):
        DenseIndex(chunks, vectors)


def test_encoder_refuses_faiss_in_process_before_loading_model(monkeypatch, tmp_path):
    from finrag.retrieval.e5 import E5Encoder

    monkeypatch.setitem(sys.modules, "faiss", object())
    with pytest.raises(FinragError, match="separately"):
        E5Encoder(tmp_path)


def test_worker_error_is_explicit_and_recorded(tmp_path, monkeypatch):
    from finrag.retrieval import faiss_process

    monkeypatch.chdir(tmp_path)
    chunks, vectors, query = inputs()
    monkeypatch.setattr(
        faiss_process.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=3, stdout="", stderr="synthetic worker failure"
        ),
    )
    root = tmp_path / "artifacts/private/synthetic"
    with pytest.raises(FinragError, match="code 3.*synthetic worker failure"):
        search_in_separate_process(root, chunks, vectors, query, {})
    assert "synthetic worker failure" in (root / "faiss_exchange/worker.log").read_text()


def test_worker_rejects_modified_input_hash(tmp_path, monkeypatch):
    from finrag.retrieval.faiss_worker import run_search

    monkeypatch.chdir(tmp_path)
    chunks, vectors, query = inputs()
    root = tmp_path / "artifacts/private/synthetic"
    search_in_separate_process(root, chunks, vectors, query, {})
    path = root / "faiss_exchange/request.json"
    request = read_json(path)
    request["inputs"]["passages"]["sha256"] = "0" * 64
    path.write_bytes(json_bytes(request))
    with pytest.raises(FinragError, match="SHA-256 changed"):
        run_search(path)


def test_worker_rejects_public_result_destination(tmp_path, monkeypatch):
    from finrag.retrieval.faiss_worker import run_search

    monkeypatch.chdir(tmp_path)
    chunks, vectors, query = inputs()
    root = tmp_path / "artifacts/private/synthetic"
    search_in_separate_process(root, chunks, vectors, query, {})
    public_request = tmp_path / "public-request.json"
    public_request.write_bytes((root / "faiss_exchange/request.json").read_bytes())
    with pytest.raises(FinragError, match="inside the private experiment root"):
        run_search(public_request)
    assert not (tmp_path / "results.json").exists()
