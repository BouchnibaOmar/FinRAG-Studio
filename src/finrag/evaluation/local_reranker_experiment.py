import argparse
import ctypes
import json
import math
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from finrag.config import FinragError, private_output
from finrag.domain.evaluation import EvaluationQuestion
from finrag.evaluation.candidate_reranking_experiment import score_cached_outputs, sealed_inputs
from finrag.evaluation.document_oracle_metrics import evaluate_candidates
from finrag.ingestion.hashing import hash_bytes, hash_file
from finrag.ingestion.manifest import json_bytes, read_json, read_jsonl, write_atomic
from finrag.retrieval.candidate_reranking import RerankInput, RerankSettings, input_identity
from finrag.retrieval.local_reranker import (
    LocalReranker,
    count_pairs,
    hardware_sample,
    rank_scores,
    validate_local_record,
)
from finrag.retrieval.local_reranker_assets import validate_model

METRICS = ("hit@1", "hit@5", "complete_excerpt_coverage@5")


def resident_memory() -> dict:
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
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

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"rss_bytes": counters.WorkingSetSize, "peak_rss_bytes": counters.PeakWorkingSetSize}
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"rss_bytes": None, "peak_rss_bytes": peak * (1 if sys.platform == "darwin" else 1024)}


def technical_failure(error: Exception, stage: str, context: dict) -> dict:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "stage": stage,
        "original_exception_type": f"{type(error).__module__}.{type(error).__name__}",
        "errno": getattr(error, "errno", None),
        "winerror": getattr(error, "winerror", None),
        "original_message_logged": False,
        "context": context,
    }


def configuration(directory: Path, device: str, dtype: str) -> dict:
    from importlib.metadata import version

    required = {
        "torch": "2.7.1",
        "transformers": "4.51.3",
        "tokenizers": "0.21.4",
        "huggingface-hub": "0.30.2",
        "safetensors": "0.5.3",
        "numpy": "1.26.4",
    }
    actual = {name: version(name) for name in required}
    if any(actual[name].split("+")[0] != pinned for name, pinned in required.items()):
        raise FinragError("Local reranker dependency versions differ from the tested pins.")
    return {
        "protocol": "local-bge-pair-v1",
        "manifest": validate_model(directory),
        "device": device,
        "dtype": dtype,
        "batch_size": 1,
        "versions": actual,
    }


def preflight(experiment: Path, directory: Path, device: str, dtype: str) -> tuple[dict, list]:
    config = configuration(directory, device, dtype)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    _, inputs = sealed_inputs(experiment)
    if any(len(source.candidates) != 50 for source in inputs):
        raise FinragError("Local preflight requires the unchanged 50 candidates per dev input.")
    start = time.perf_counter()
    counts = count_pairs(inputs, tokenizer)
    return {
        "configuration": config,
        "input_ids": [input_identity(s) for s in inputs],
        "tokenizer_seconds": time.perf_counter() - start,
        **counts,
    }, inputs


def smoke(
    inputs: list[RerankInput],
    counts: dict,
    backend: LocalReranker,
    memory: Callable[[], dict] = resident_memory,
) -> dict:
    if counts["overflows"]:
        raise FinragError("Pair overflows detected; hardware inference blocked.")
    sample = hardware_sample(counts["pairs"])
    texts = {
        (input_identity(s), c.candidate_id): (s.question, c.text)
        for s in inputs
        for c in s.candidates
    }
    before = memory()
    backend.gpu_memory(reset=True)
    backend.score_pairs([texts[(sample[0]["input_id"], sample[0]["candidate_id"])]])
    backend.synchronize()
    start = time.perf_counter()
    measured = []
    for pair in sample:
        one = time.perf_counter()
        value = backend.score_pairs([texts[(pair["input_id"], pair["candidate_id"])]])
        backend.synchronize()
        if len(value) != 1 or not math.isfinite(value[0]):
            raise FinragError("Hardware smoke returned an invalid score.")
        measured.append({**pair, "seconds": time.perf_counter() - one, "score": value[0]})
    duration = time.perf_counter() - start
    return {
        "status": "success",
        "configuration": counts["configuration"],
        "input_ids": counts["input_ids"],
        "sample": measured,
        "warmup_pairs": 1,
        "measured_pairs": len(sample),
        "seconds": duration,
        "pairs_per_second": len(sample) / duration,
        "ram_before": before,
        "ram_after": memory(),
        "gpu_memory": backend.gpu_memory(),
        "naive_full_inference_seconds": duration / len(sample) * len(counts["pairs"]),
        "estimate_is_not_guarantee": True,
        "sample_rule": "three shortest, three median, three longest pairs; "
        "ties by input/candidate ID",
    }


def local_key(source: RerankInput, config: dict) -> str:
    return hash_bytes(json_bytes({"input_id": input_identity(source), "configuration": config}))


def execute(inputs: list[RerankInput], counts: dict, backend: LocalReranker, cache: Path) -> dict:
    if counts["overflows"]:
        raise FinragError("Pair overflows detected; full inference blocked.")
    statuses = []
    for source in inputs:
        key = local_key(source, counts["configuration"])
        path = cache / f"{key}.json"
        if path.exists():
            record = read_json(path)
            if record["status"] == "success":
                validate_local_record(source, record, counts["configuration"])
            else:
                raise FinragError("Previous local failure remains recorded; no automatic retry.")
            statuses.append(
                {"input_id": input_identity(source), "status": record["status"], "reused": True}
            )
            continue
        start = time.perf_counter()
        try:
            values = []
            for candidate in source.candidates:
                pair_scores = backend.score_pairs([(source.question, candidate.text)])
                if len(pair_scores) != 1 or not math.isfinite(pair_scores[0]):
                    raise FinragError("Local scorer must return exactly one finite score per pair.")
                values.extend(pair_scores)
            backend.synchronize()
            ranking = rank_scores(source, values)
            record = {
                "status": "success",
                "input_id": input_identity(source),
                "configuration": counts["configuration"],
                "ranking": ranking,
                "seconds": time.perf_counter() - start,
                "ram": resident_memory(),
                "gpu_memory": backend.gpu_memory(),
            }
        except Exception as error:
            record = {
                "status": "failed",
                "input_id": input_identity(source),
                "configuration": counts["configuration"],
                "error": technical_failure(
                    error,
                    "score_input",
                    {
                        "input_id": input_identity(source),
                        "candidate_id": candidate.candidate_id,
                        "device": counts["configuration"]["device"],
                    },
                ),
                "seconds": time.perf_counter() - start,
            }
            write_atomic(path, json_bytes(record))
            raise FinragError(
                f"Local scoring failed for input {input_identity(source)}; inspect technical error."
            ) from error
        write_atomic(path, json_bytes(record))
        statuses.append({"input_id": input_identity(source), "status": "success", "reused": False})
    return {"inputs": statuses}


def evaluate(experiment: Path, cache: Path, config: dict, mistral_run: Path) -> dict:
    _, inputs = sealed_inputs(experiment)
    baseline = read_jsonl(experiment / "evaluation/baseline.jsonl")
    reference = (
        Path.cwd() / "artifacts/private/bm25_query_translation_dev/evaluation/references.jsonl"
    )
    routed = read_json(Path("artifacts/private/bm25_company_routing_dev/run_1/comparison.json"))
    if hash_file(reference) != routed["input_hashes"].get(str(reference)):
        raise FinragError("Sealed dev v1 reference hash differs.")
    questions = [EvaluationQuestion.model_validate(q) for q in read_jsonl(reference)]
    if len(questions) != 28 or any(q.split != "dev" or not q.answerable for q in questions):
        raise FinragError("Local evaluation requires 28 answerable dev references.")
    by_question = {q.question_id: q for q in questions}
    if len(by_question) != 28 or set(by_question) != {row["question_id"] for row in baseline}:
        raise FinragError("Local references and baseline question identities differ.")
    sources = {input_identity(s): s for s in inputs}
    rows = []
    for base in baseline:
        source = sources[base["input_id"]]
        path = cache / f"{local_key(source, config)}.json"
        record = read_json(path) if path.exists() else {"status": "not_attempted"}
        status, hits = record["status"], []
        if status == "success":
            verified = validate_local_record(source, record, config)
            by_chunk = {h["chunk_id"]: h for h in base["hits"]}
            hits = [
                {**by_chunk[base["candidate_id_to_chunk_id"][r["candidate_id"]]], "rank": r["rank"]}
                for r in verified
            ]
        metrics = (
            evaluate_candidates(by_question[base["question_id"]], hits, True)
            if hits
            else {m: False for m in METRICS}
        )
        rows.append(
            {
                "question_id": base["question_id"],
                "status": status,
                "metrics": {m: metrics[m] for m in METRICS},
                "baseline": {m: base["metrics"][m] for m in METRICS},
                "changes": {
                    m: {
                        "gain": metrics[m] and not base["metrics"][m],
                        "regression": base["metrics"][m] and not metrics[m],
                    }
                    for m in METRICS
                },
                "seconds": record.get("seconds"),
            }
        )
    mistral = score_cached_outputs(
        questions,
        baseline,
        inputs,
        mistral_run / "cache",
        RerankSettings.model_validate(read_json(mistral_run / "settings.json")),
    )
    matched = {r["question_id"] for r in mistral["predictions"] if r["status"] == "success"}
    audited = read_json(mistral_run / "analysis_run_1/analysis.json")
    expected_matched = {
        row["question_id"] for row in audited["question_metrics"] if row["status"] == "success"
    }
    if len(matched) != 12 or matched != expected_matched:
        raise FinragError("Mistral paired subset changed from the expected 12 valid outputs.")

    def aggregate(items, field):
        return {
            m: {"numerator": sum(r[field][m] for r in items), "denominator": len(items)}
            for m in METRICS
        }

    paired = [r for r in rows if r["question_id"] in matched]
    return {
        "questions": 28,
        "failures_remain_in_denominator": True,
        "rows": rows,
        "local_28": aggregate(rows, "metrics"),
        "baseline_28": aggregate(rows, "baseline"),
        "matched_mistral_question_ids": sorted(matched),
        "local_matched12": aggregate(paired, "metrics"),
        "baseline_matched12": aggregate(paired, "baseline"),
        "mistral_matched12": aggregate(
            [r for r in mistral["predictions"] if r["question_id"] in matched], "metrics"
        ),
        "local_missing_outputs_are_execution_failures": True,
        "generalization_from12_to28": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinned local reranker, no network inference")
    parser.add_argument("command", choices=("preflight", "smoke", "run", "evaluate"))
    parser.add_argument(
        "--experiment", type=Path, default=Path("artifacts/private/bm25_candidate_reranking_dev")
    )
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dtype", choices=("float32", "float16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument(
        "--mistral-run",
        type=Path,
        default=Path(
            "artifacts/private/bm25_candidate_reranking_dev/models/mistral_large_4_0/full_dev_v2_run_1"
        ),
    )
    args = parser.parse_args()
    output = private_output(args.output, Path.cwd())
    if output.exists():
        raise FinragError("Local output exists; overwrite refused.")
    stage = "preflight"
    try:
        if args.command == "evaluate":
            if args.cache is None:
                raise FinragError("Evaluation requires --cache.")
            report = evaluate(
                args.experiment,
                private_output(args.cache, Path.cwd()),
                configuration(args.model_directory, args.device, args.dtype),
                args.mistral_run,
            )
        else:
            counts, inputs = preflight(
                args.experiment, args.model_directory, args.device, args.dtype
            )
            if args.command == "preflight":
                report = counts
            else:
                if counts["overflows"]:
                    raise FinragError(
                        "Pair overflows detected; inference blocked. Run preflight for IDs/counts."
                    )
                if args.command == "run":
                    if args.cache is None or args.smoke_report is None:
                        raise FinragError("Run requires --cache and --smoke-report.")
                    prior = read_json(args.smoke_report)
                    if (
                        prior.get("status") != "success"
                        or prior["configuration"] != counts["configuration"]
                        or prior["input_ids"] != counts["input_ids"]
                    ):
                        raise FinragError(
                            "A matching successful hardware smoke report is required."
                        )
                stage = "model_load"
                start = time.perf_counter()
                backend = LocalReranker(args.model_directory, args.device, args.dtype)
                load_seconds = time.perf_counter() - start
                stage = args.command
                report = (
                    smoke(inputs, counts, backend)
                    if args.command == "smoke"
                    else execute(inputs, counts, backend, private_output(args.cache, Path.cwd()))
                )
                report["model_load_seconds"] = load_seconds
        write_atomic(output, json_bytes(report))
        print(
            json.dumps(
                {
                    "output": str(output),
                    "command": args.command,
                    "overflows": len(report.get("overflows", [])),
                }
            )
        )
        if report.get("overflows"):
            raise SystemExit(2)
    except Exception as error:
        diagnostic = technical_failure(error, stage, {"device": args.device, "dtype": args.dtype})
        write_atomic(output, json_bytes({"status": "failed", "error": diagnostic}))
        print(
            f"Local reranker failed during {stage}: {type(error).__name__}; inspect {output}.",
            file=sys.stderr,
        )
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
