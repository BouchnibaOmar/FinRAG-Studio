import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from finrag.config import FinragError, filesystem_operation, private_output
from finrag.domain.dense import DenseHit, FaissRequest, ModelChunk
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes, read_json, write_atomic
from finrag.retrieval.embedding_cache import save_vectors
from finrag.retrieval.vector_index import validate_vectors


def _save_or_validate_array(path: Path, vectors: np.ndarray) -> None:
    if path.exists():
        with filesystem_operation("read FAISS exchange array", path):
            existing = np.load(path, allow_pickle=False)
        validate_vectors(existing, len(vectors), 384)
        if not np.array_equal(existing, vectors):
            raise FinragError("FAISS exchange array differs from validated cached vectors.")
    else:
        save_vectors(path, vectors)


def search_in_separate_process(
    root: Path,
    chunks: list[ModelChunk],
    passages: np.ndarray,
    queries: np.ndarray,
    fingerprint: dict,
) -> list[list[DenseHit]]:
    root = private_output(root, Path.cwd())
    validate_vectors(passages, len(chunks), 384)
    validate_vectors(queries, len(queries), 384)
    exchange = root / "faiss_exchange"
    exchange.mkdir(parents=True, exist_ok=True)
    chunks_path = exchange / "chunks.jsonl"
    content = jsonl_bytes([chunk.model_dump(mode="json") for chunk in chunks])
    if chunks_path.exists():
        if chunks_path.read_bytes() != content:
            raise FinragError("FAISS exchange chunk metadata changed.")
    else:
        write_atomic(chunks_path, content)
    passage_path, query_path = exchange / "passages.npy", exchange / "queries.npy"
    _save_or_validate_array(passage_path, passages)
    _save_or_validate_array(query_path, queries)
    inputs = {
        name: {"path": str(path), "sha256": hash_file(path)}
        for name, path in (
            ("chunks", chunks_path),
            ("passages", passage_path),
            ("queries", query_path),
        )
    }
    request = {
        "root": str(root),
        "fingerprint": fingerprint,
        "inputs": inputs,
        "query_count": len(queries),
        "top_k": 50,
        "threads": 4,
    }
    request = FaissRequest.model_validate(request).model_dump(mode="json")
    request_path = exchange / "request.json"
    content = json_bytes(request)
    if request_path.exists():
        if request_path.read_bytes() != content:
            raise FinragError("FAISS exchange request fingerprint changed.")
    else:
        write_atomic(request_path, content)
    # A fresh interpreter keeps FAISS's LLVM runtime out of the Torch process.
    command = [
        sys.executable,
        "-m",
        "finrag.retrieval.faiss_worker",
        "--request",
        str(request_path),
    ]
    try:
        with filesystem_operation("run isolated FAISS search", request_path):
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=120, check=False
            )
    except subprocess.TimeoutExpired as error:
        raise FinragError("Isolated FAISS search exceeded its 120-second timeout.") from error
    write_atomic(
        exchange / "worker.log",
        (completed.stdout + completed.stderr).encode("utf-8"),
        overwrite=True,
    )
    if completed.returncode != 0:
        raise FinragError(
            f"Isolated FAISS search exited with code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        receipt = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise FinragError("FAISS worker did not return a JSON receipt.") from error
    result_path = exchange / "results.json"
    if hash_file(result_path) != receipt["results_sha256"]:
        raise FinragError("FAISS result hash differs from the worker receipt.")
    result = read_json(result_path)
    if (
        result["request_sha256"] != hash_file(request_path)
        or result["query_count"] != len(queries)
        or result["worker_torch_loaded"]
    ):
        raise FinragError("FAISS result request/count/process isolation validation failed.")
    rows = [[DenseHit.model_validate(hit) for hit in hits] for hits in result["hits"]]
    if len(rows) != len(queries):
        raise FinragError("FAISS result row count differs from query count.")
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for hits in rows:
        if len(hits) != min(50, len(chunks)) or len({hit.chunk_id for hit in hits}) != len(hits):
            raise FinragError("FAISS result contains missing or duplicated chunk IDs.")
        for rank, hit in enumerate(hits, 1):
            chunk = by_id.get(hit.chunk_id)
            if (
                chunk is None
                or hit.rank != rank
                or any(
                    getattr(hit, field) != getattr(chunk, field)
                    for field in (
                        "document_id",
                        "physical_page_number",
                        "source_filename",
                        "source_sha256",
                        "text",
                    )
                )
            ):
                raise FinragError(
                    "FAISS result rank or source provenance differs from input chunks."
                )
    return rows
