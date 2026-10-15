import sys
from pathlib import Path

import numpy as np

from finrag.config import FinragError
from finrag.domain.dense import DenseHit, ModelChunk
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes, read_json, read_jsonl, write_atomic


def normalize_vectors(vectors: np.ndarray, dimensions: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != dimensions or not np.isfinite(vectors).all():
        raise FinragError("Embedding vectors have invalid dimensions or nonfinite values.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise FinragError("Zero embedding vectors cannot represent cosine similarity.")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def validate_vectors(vectors: np.ndarray, rows: int, dimensions: int) -> None:
    if (
        vectors.dtype != np.float32
        or vectors.shape != (rows, dimensions)
        or not np.isfinite(vectors).all()
    ):
        raise FinragError("Embedding count, dimensions, dtype or finite-value contract failed.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise FinragError("Embedding vectors must be L2-normalised.")


class DenseIndex:
    def __init__(self, chunks: list[ModelChunk], vectors: np.ndarray, dimensions: int = 384):
        if "torch" in sys.modules:
            raise FinragError("FAISS must run in its isolated worker when Torch is loaded.")
        import faiss

        if not chunks or len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise FinragError("Dense index requires nonempty unique chunk IDs.")
        validate_vectors(vectors, len(chunks), dimensions)
        order = sorted(range(len(chunks)), key=lambda row: chunks[row].chunk_id)
        self.chunks = [chunks[row] for row in order]
        self.index = faiss.IndexFlatIP(dimensions)
        self.index.add(np.ascontiguousarray(vectors[order]))

    def search(self, query: np.ndarray, top_k: int = 50) -> list[DenseHit]:
        if top_k <= 0:
            raise FinragError("Dense top_k must be positive.")
        validate_vectors(query, 1, self.index.d)
        # Score every row so ties at the candidate cutoff use the same chunk-ID rule.
        scores, rows = self.index.search(query, self.index.ntotal)
        ranked = sorted(
            zip(scores[0], rows[0], strict=True),
            key=lambda pair: (-float(pair[0]), self.chunks[int(pair[1])].chunk_id),
        )[:top_k]
        return [
            DenseHit(
                rank=rank,
                score=float(score),
                chunk_id=self.chunks[int(row)].chunk_id,
                document_id=self.chunks[int(row)].document_id,
                source_filename=self.chunks[int(row)].source_filename,
                source_sha256=self.chunks[int(row)].source_sha256,
                physical_page_number=self.chunks[int(row)].physical_page_number,
                text=self.chunks[int(row)].text,
                configuration_id="dense-e5-v1",
            )
            for rank, (score, row) in enumerate(ranked, start=1)
        ]

    def save(self, root: Path, fingerprint: dict) -> None:
        import faiss

        if root.exists():
            raise FinragError("Dense index output already exists; overwrite refused.")
        root.mkdir(parents=True)
        faiss.write_index(self.index, str(root / "index.faiss"))
        write_atomic(
            root / "rows.jsonl",
            jsonl_bytes([chunk.model_dump(mode="json") for chunk in self.chunks]),
        )
        write_atomic(
            root / "index_manifest.json",
            json_bytes(
                {
                    "fingerprint": fingerprint,
                    "dimensions": self.index.d,
                    "rows": self.index.ntotal,
                    "index_sha256": hash_file(root / "index.faiss"),
                    "metadata_sha256": hash_file(root / "rows.jsonl"),
                    "storage": "FAISS IndexFlatIP + validated JSON; no pickle",
                }
            ),
        )

    @classmethod
    def load(cls, root: Path, fingerprint: dict) -> "DenseIndex":
        if "torch" in sys.modules:
            raise FinragError("FAISS must run in its isolated worker when Torch is loaded.")
        import faiss

        manifest = read_json(root / "index_manifest.json")
        if (
            manifest["fingerprint"] != fingerprint
            or hash_file(root / "index.faiss") != manifest["index_sha256"]
            or hash_file(root / "rows.jsonl") != manifest["metadata_sha256"]
        ):
            raise FinragError("Dense index input or native-storage hashes changed.")
        chunks = [ModelChunk.model_validate(row) for row in read_jsonl(root / "rows.jsonl")]
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks) or [
            chunk.chunk_id for chunk in chunks
        ] != sorted(chunk.chunk_id for chunk in chunks):
            raise FinragError("Dense row mapping is duplicated or not deterministically ordered.")
        index = faiss.read_index(str(root / "index.faiss"))
        if (
            not isinstance(index, faiss.IndexFlatIP)
            or index.ntotal != len(chunks)
            or index.d != manifest["dimensions"]
            or index.ntotal != manifest["rows"]
        ):
            raise FinragError("FAISS index type, vector count or dimensions do not match metadata.")
        validate_vectors(index.reconstruct_n(0, index.ntotal), index.ntotal, index.d)
        instance = cls.__new__(cls)
        instance.chunks, instance.index = chunks, index
        return instance
