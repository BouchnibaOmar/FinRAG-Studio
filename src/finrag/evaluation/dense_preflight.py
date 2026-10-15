import importlib.metadata
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from finrag.chunking.model_chunker import split_for_model, validate_model_chunks
from finrag.config import FinragError, private_output
from finrag.domain.chunks import ChunkRecord
from finrag.domain.dense import ModelChunk
from finrag.evaluation.local_model_smoke import run_local_model_smoke
from finrag.evaluation.run_lexical_experiment import experiment_inputs, load_config
from finrag.evaluation.split_guard import load_dev_questions
from finrag.ingestion.hashing import hash_bytes, hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes, read_json, read_jsonl, write_atomic
from finrag.retrieval.e5 import PREFIXES, E5Encoder, require_token_limit, token_count
from finrag.retrieval.embedding_cache import EmbeddingCache
from finrag.retrieval.model_assets import REVISION

DEFAULT_MODEL = Path("models/multilingual-e5-small") / REVISION
DEFAULT_ROOT = Path("artifacts/private/phase2b")
PROTOCOL = Path("experiments/phase2b/protocol.json")


def length_summary(lengths: list[int]) -> dict:
    ordered = sorted(lengths)
    return {
        "count": len(lengths),
        "minimum": min(lengths),
        "maximum": max(lengths),
        "mean": mean(lengths),
        "median": median(lengths),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


def original_inputs() -> tuple[list[ChunkRecord], dict, dict]:
    return experiment_inputs(
        Path("data/private/corpus/originals/corpus_manifest.json"),
        Path("data/private/corpus/extracted"),
        Path("data/private/corpus/chunks"),
        load_config(Path("experiments/phase2a/configs/B.json")),
    )


def deterministic_sample(chunks: list[ModelChunk]) -> list[int]:
    documents = defaultdict(list)
    for row, chunk in enumerate(chunks):
        documents[chunk.document_id].append(row)
    return sorted(
        {
            rows[position]
            for rows in documents.values()
            for position in (0, len(rows) // 2, len(rows) - 1)
        }
    )


def prepare_dense_inputs(encoder: E5Encoder, root: Path) -> tuple[list[ModelChunk], dict]:
    parents, _, original_hashes = original_inputs()
    questions = load_dev_questions(Path("data/private/evaluation/finrag_v1"))
    protocol = read_json(PROTOCOL)
    fixed = {
        "prefixes": PREFIXES,
        "child_overlap_tokens": 32,
        "split": "dev",
        "dimensions": 384,
        "test_evaluation": False,
        "selection_priority": ["Hit@5", "MRR@10", "Hit@1"],
        "tie_simplicity_order": ["bm25", "dense", "hybrid"],
        "hybrid": {
            "method": "reciprocal_rank_fusion",
            "weights": [1.0, 1.0],
            "constant": 60,
            "candidates_per_retriever": 50,
        },
    }
    if any(protocol.get(key) != value for key, value in fixed.items()):
        raise FinragError("Protocol differs from the implemented fixed comparison.")
    if protocol["model_revision"] != REVISION or protocol["maximum_length"] != encoder.limit:
        raise FinragError("Local model limits/revision differ from the predeclared protocol.")
    input_hash = hash_bytes(json_bytes([chunk.model_dump(mode="json") for chunk in parents]))
    lengths = [token_count(encoder.tokenizer, chunk.text, "passage") for chunk in parents]
    query_lengths = require_token_limit(
        encoder.tokenizer, [question.question for question in questions], "query", encoder.limit
    )
    chunks_path = root / "model_chunks.jsonl"
    if chunks_path.exists():
        children = [ModelChunk.model_validate(row) for row in read_jsonl(chunks_path)]
        validate_model_chunks(
            parents,
            children,
            encoder.tokenizer,
            REVISION,
            encoder.limit,
            protocol["child_overlap_tokens"],
        )
    else:
        children = split_for_model(
            parents, encoder.tokenizer, REVISION, encoder.limit, protocol["child_overlap_tokens"]
        )
        write_atomic(
            chunks_path, jsonl_bytes([chunk.model_dump(mode="json") for chunk in children])
        )
    fingerprint = {
        "model": encoder.manifest,
        "prefixes": PREFIXES,
        "pooling": "masked-mean-v1",
        "tokenizer_revision": REVISION,
        "effective_limit": encoder.limit,
        "original_B_input_sha256": input_hash,
        "original_B_fingerprints": original_hashes,
        "model_chunks_sha256": hash_file(chunks_path),
        "protocol_sha256": hash_file(PROTOCOL),
    }
    preflight = {
        "effective_limit": encoder.limit,
        "tokenizer_limit": encoder.tokenizer.model_max_length,
        "model_position_limit": encoder.model.config.max_position_embeddings,
        "original_B_count": len(parents),
        "original_B_over_limit": sum(length > encoder.limit for length in lengths),
        "original_B_lengths": length_summary(lengths),
        "model_chunk_count": len(children),
        "model_chunk_lengths": length_summary([chunk.token_count for chunk in children]),
        "dev_query_lengths": length_summary(query_lengths),
        "fingerprint": fingerprint,
        "B_identities_preserved": all(
            child.chunk_id == parent.chunk_id
            for child, parent in zip(children, parents, strict=False)
        )
        and len(children) == len(parents),
        "character_coverage_verified": True,
        "silent_truncation": False,
    }
    path = root / "token_preflight.json"
    if path.exists() and read_json(path) != preflight:
        raise FinragError("Token preflight cache differs from current model/chunk inputs.")
    if not path.exists():
        write_atomic(path, json_bytes(preflight))
    return children, preflight


def encoding_fingerprint(encoder: E5Encoder, preflight: dict, batch_size: int) -> dict:
    return {
        **preflight["fingerprint"],
        "encoding": {
            "device": encoder.device,
            "precision": "float32",
            "batch_size": batch_size,
            "threads": encoder.threads,
            "deterministic_algorithms": True,
            "seed": 20261013,
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "tokenizers",
                    "numpy",
                    "faiss-cpu",
                    "safetensors",
                )
            },
        },
    }


def run_preflight(
    model: Path = DEFAULT_MODEL, root: Path = DEFAULT_ROOT, device: str = "cpu", batch_size: int = 4
) -> dict:
    import torch

    root = private_output(root, Path.cwd())
    root.mkdir(parents=True, exist_ok=True)
    encoder = E5Encoder(model, device)
    chunks, tokens = prepare_dense_inputs(encoder, root)
    fingerprint = encoding_fingerprint(encoder, tokens, batch_size)
    cache = EmbeddingCache(root / "embeddings", fingerprint, len(chunks))
    path = root / "sample_benchmark.json"
    if path.exists():
        benchmark = read_json(path)
        if benchmark["fingerprint"] != fingerprint:
            raise FinragError("Sample benchmark encoding configuration changed.")
        return benchmark
    rows = deterministic_sample(chunks)
    if any(row in cache.completed for row in rows):
        raise FinragError(
            "A sample checkpoint exists without its benchmark; inspect before resuming."
        )
    smoke = run_local_model_smoke(encoder)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        cache.store(batch, encoder.encode([chunks[row].text for row in batch], "passage"))
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = perf_counter() - started
    throughput = len(rows) / seconds
    estimate = (len(chunks) - len(rows)) / throughput * 1.25
    benchmark = {
        "fingerprint": fingerprint,
        "sample_rows": rows,
        "sample_count": len(rows),
        "sample_seconds": seconds,
        "passages_per_second": throughput,
        "remaining_encoding_estimate_seconds_with_25pct_margin": estimate,
        "requires_manual_execution": estimate > 300,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if device == "cuda"
        else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
        "cpu_peak_working_set_bytes": _peak_working_set(),
        "real_model_smoke": smoke,
        "token_preflight": {key: value for key, value in tokens.items() if key != "fingerprint"},
    }
    write_atomic(path, json_bytes(benchmark))
    return benchmark


def _peak_working_set() -> int | None:
    import ctypes
    from ctypes import wintypes

    if not hasattr(ctypes, "windll"):
        return None

    class MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            *[
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ],
        ]

    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize)
