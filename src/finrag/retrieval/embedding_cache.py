import io
from collections.abc import Callable
from pathlib import Path

import numpy as np

from finrag.config import FinragError, filesystem_operation
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, read_json, write_atomic
from finrag.retrieval.vector_index import validate_vectors


def save_vectors(path: Path, vectors: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, vectors, allow_pickle=False)
    write_atomic(path, buffer.getvalue())


class EmbeddingCache:
    def __init__(self, root: Path, fingerprint: dict, rows: int, dimensions: int = 384):
        self.root, self.rows, self.dimensions = root, rows, dimensions
        with filesystem_operation("create embedding cache directory", root):
            root.mkdir(parents=True, exist_ok=True)
        path = root / "cache_manifest.json"
        expected = {
            "fingerprint": fingerprint,
            "rows": rows,
            "dimensions": dimensions,
            "schema_version": "1.0",
        }
        if path.exists() and read_json(path) != expected:
            raise FinragError("Embedding cache input/model/encoding fingerprint mismatch.")
        if not path.exists():
            if any(root.iterdir()):
                raise FinragError("Embedding cache contains unregistered files.")
            write_atomic(path, json_bytes(expected))
        self.completed = self._read_completed()

    def _read_completed(self) -> dict[int, np.ndarray]:
        completed = {}
        for record in sorted(self.root.glob("batch-*.json")):
            metadata = read_json(record)
            path = self.root / metadata["file"]
            if (
                path.parent != self.root
                or path.suffix != ".npy"
                or hash_file(path) != metadata["sha256"]
            ):
                raise FinragError("Checkpoint file path or hash is invalid.")
            with filesystem_operation("read embedding checkpoint", path):
                vectors = np.load(path, allow_pickle=False)
            validate_vectors(vectors, len(metadata["row_indices"]), self.dimensions)
            for row, vector in zip(metadata["row_indices"], vectors, strict=True):
                if not isinstance(row, int) or row < 0 or row >= self.rows or row in completed:
                    raise FinragError("Checkpoint contains duplicate or invalid row indices.")
                completed[row] = vector
        registered = {read_json(path)["file"] for path in self.root.glob("batch-*.json")}
        if {path.name for path in self.root.glob("*.npy")} != registered:
            raise FinragError("Uncommitted embedding checkpoint found; inspect before resuming.")
        return completed

    def store(self, row_indices: list[int], vectors: np.ndarray) -> None:
        if len(set(row_indices)) != len(row_indices) or any(
            row in self.completed or row < 0 or row >= self.rows for row in row_indices
        ):
            raise FinragError("Embedding checkpoint would overwrite completed or invalid rows.")
        validate_vectors(vectors, len(row_indices), self.dimensions)
        name = f"batch-{len(list(self.root.glob('batch-*.json'))):06d}"
        path = self.root / (name + ".npy")
        save_vectors(path, vectors)
        write_atomic(
            self.root / (name + ".json"),
            json_bytes({"file": path.name, "row_indices": row_indices, "sha256": hash_file(path)}),
        )
        for row, vector in zip(row_indices, vectors, strict=True):
            self.completed[row] = vector.copy()

    def resume(self, texts: list[str], encode: Callable, batch_size: int, log: Path) -> np.ndarray:
        if len(texts) != self.rows or batch_size <= 0:
            raise FinragError("Cache rows or encoding batch size are inconsistent.")
        remaining = [row for row in range(self.rows) if row not in self.completed]
        with filesystem_operation("create progress log directory", log.parent):
            log.parent.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(remaining), batch_size):
            rows = remaining[start : start + batch_size]
            vectors = encode([texts[row] for row in rows])
            self.store(rows, vectors)
            with (
                filesystem_operation("append encoding progress log", log),
                log.open("a", encoding="utf-8") as output,
            ):
                output.write(f"completed={len(self.completed)} total={self.rows}\n")
                output.flush()
        return np.stack([self.completed[row] for row in range(self.rows)]).astype(np.float32)
