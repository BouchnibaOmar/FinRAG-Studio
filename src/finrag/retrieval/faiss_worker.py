import argparse
import json
import sys
from pathlib import Path

import numpy as np

from finrag.config import FinragError, filesystem_operation, private_output
from finrag.domain.dense import FaissRequest, ModelChunk
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, read_json, read_jsonl, write_atomic
from finrag.retrieval.vector_index import DenseIndex, validate_vectors


def run_search(request_path: Path) -> dict:
    if "torch" in sys.modules:
        raise FinragError("FAISS worker must run without importing Torch.")
    import faiss

    request = FaissRequest.model_validate(read_json(request_path)).model_dump(mode="json")
    root = private_output(Path(request["root"]), Path.cwd())
    if not request_path.resolve().is_relative_to(root):
        raise FinragError(
            "FAISS request and results must remain inside the private experiment root."
        )
    arrays = {}
    for name in ("chunks", "passages", "queries"):
        record = request["inputs"][name]
        path = Path(record["path"])
        if not path.resolve().is_relative_to(root) or hash_file(path) != record["sha256"]:
            raise FinragError(f"FAISS {name} input path or SHA-256 changed.")
        if name == "chunks":
            chunks = [ModelChunk.model_validate(row) for row in read_jsonl(path)]
        else:
            with filesystem_operation("read FAISS input array", path):
                arrays[name] = np.load(path, allow_pickle=False)
    validate_vectors(arrays["passages"], len(chunks), 384)
    validate_vectors(arrays["queries"], request["query_count"], 384)
    if request["top_k"] != 50 or request["threads"] != 4:
        raise FinragError("FAISS worker requires fixed top-50 retrieval and four threads.")
    faiss.omp_set_num_threads(4)
    index_root = root / "index"
    if index_root.exists():
        index = DenseIndex.load(index_root, request["fingerprint"])
        if {chunk.chunk_id: chunk for chunk in index.chunks} != {
            chunk.chunk_id: chunk for chunk in chunks
        }:
            raise FinragError("Persisted FAISS row metadata differs from input chunks.")
    else:
        index = DenseIndex(chunks, arrays["passages"])
        index.save(index_root, request["fingerprint"])
    hits = [index.search(query.reshape(1, -1), 50) for query in arrays["queries"]]
    result = {
        "request_sha256": hash_file(request_path),
        "query_count": len(hits),
        "worker_torch_loaded": "torch" in sys.modules,
        "faiss_version": faiss.__version__,
        "threads": 4,
        "hits": [[hit.model_dump(mode="json") for hit in rows] for rows in hits],
    }
    result_path = request_path.parent / "results.json"
    content = json_bytes(result)
    if result_path.exists():
        if result_path.read_bytes() != content:
            raise FinragError("Existing FAISS results differ from the validated request.")
    else:
        write_atomic(result_path, content)
    return {"results_sha256": hash_file(result_path), "worker_torch_loaded": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated exact FAISS search")
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run_search(args.request)
    except (FinragError, OSError, ValueError, RuntimeError) as error:
        print(f"FAISS worker failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
