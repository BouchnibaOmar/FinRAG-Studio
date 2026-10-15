import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from finrag.config import FinragError, private_output
from finrag.domain.dense import ModelChunk
from finrag.domain.evaluation import EvaluationQuestion
from finrag.evaluation.document_oracle_metrics import evaluate_candidates
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, read_json, read_jsonl, write_atomic
from finrag.retrieval.vector_index import DenseIndex, validate_vectors


def document_rows(chunks: list[ModelChunk]) -> dict[str, list[int]]:
    rows = defaultdict(list)
    for row, chunk in enumerate(chunks):
        rows[chunk.document_id].append(row)
    return dict(rows)


def run_worker(request_path: Path) -> dict:
    if "torch" in sys.modules:
        raise FinragError("Document oracle FAISS worker must not import Torch.")
    started = perf_counter()
    job = read_json(request_path)
    output = private_output(Path(job["output"]), Path.cwd())
    input_root = private_output(Path(job["input_root"]), Path.cwd())
    if not request_path.resolve().is_relative_to(output):
        raise FinragError("Oracle request must remain inside its new private output.")
    inputs = {}
    for name, record in job["inputs"].items():
        path = Path(record["path"])
        if not path.resolve().is_relative_to(input_root) or hash_file(path) != record["sha256"]:
            raise FinragError(f"Document oracle input {name} path/hash mismatch.")
        inputs[name] = path
    chunks = [ModelChunk.model_validate(row) for row in read_jsonl(inputs["chunks"])]
    passages = np.load(inputs["passages"], allow_pickle=False)
    queries = np.load(inputs["queries"], allow_pickle=False)
    validate_vectors(passages, len(chunks), 384)
    validate_vectors(queries, job["cached_query_count"], 384)
    import faiss

    faiss.omp_set_num_threads(4)
    grouped_rows = document_rows(chunks)
    indexes = {}
    results = []
    for item in job["questions"]:
        question = EvaluationQuestion.model_validate(item["question"])
        if not question.answerable or question.split != "dev":
            raise FinragError("Oracle worker accepts only answerable dev questions.")
        row = item["query_row"]
        if not isinstance(row, int) or not 0 <= row < len(queries):
            raise FinragError("Oracle query row is outside the validated cache.")
        selected = grouped_rows.get(question.document_id)
        if not selected:
            raise FinragError("Reference document has no cached passages.")
        if question.document_id not in indexes:
            indexes[question.document_id] = DenseIndex(
                [chunks[position] for position in selected], passages[selected]
            )
        index = indexes[question.document_id]
        ranked = index.search(queries[row : row + 1], top_k=len(selected))
        hits = [hit.model_dump(mode="json") for hit in ranked]
        results.append(
            {
                "question_id": question.question_id,
                "document_passages_searched": len(selected),
                "hits": hits[:50],
                "metrics": evaluate_candidates(question, hits, True),
            }
        )
    result_path = output / "dense_worker_results.json"
    write_atomic(
        result_path,
        json_bytes(
            {
                "request_sha256": hash_file(request_path),
                "worker_torch_loaded": False,
                "faiss_version": faiss.__version__,
                "threads": 4,
                "runtime_seconds": perf_counter() - started,
                "results": results,
            }
        ),
    )
    return {"results_sha256": hash_file(result_path), "worker_torch_loaded": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated cached-vector document oracle")
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_worker(args.request)))


if __name__ == "__main__":
    main()
