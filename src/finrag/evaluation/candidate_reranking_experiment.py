import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from finrag.config import FinragError, filesystem_operation, private_output
from finrag.evaluation.document_oracle_metrics import aggregate_metrics, evaluate_candidates
from finrag.evaluation.reranking_cache import (
    cached_rerank,
    check_estimated_spend,
    context_check,
    estimated_cost,
    load_model_snapshot,
    reusable_record,
)
from finrag.evaluation.split_guard import load_dev_questions
from finrag.ingestion.hashing import hash_bytes, hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes, read_json, read_jsonl, write_atomic
from finrag.retrieval.candidate_reranking import (
    OpenRouterReranker,
    PublicCandidate,
    RerankInput,
    RerankSettings,
    input_identity,
    request_identity,
    token_reservation,
    validate_order,
)
from finrag.retrieval.company_router import CatalogueDocument

ORDER_METRICS = ("hit@1", "hit@5", "complete_excerpt_coverage@5")


def project_candidates(
    question: str, hits: list[dict], catalogue: list[CatalogueDocument]
) -> tuple[RerankInput, dict]:
    by_document = {doc.document_id: doc for doc in catalogue}
    candidates, mapping = [], {}
    for hit in hits:
        candidate_id = (
            "c_"
            + hash_bytes(
                json_bytes({"document_id": hit["document_id"], "chunk_id": hit["chunk_id"]})
            )[:20]
        )
        if candidate_id in mapping:
            raise FinragError("Duplicate or colliding candidate identity; no candidates removed.")
        doc = by_document.get(hit["document_id"])
        if doc is None or (doc.source_filename, doc.source_sha256) != (
            hit["source_filename"],
            hit["source_sha256"],
        ):
            raise FinragError("Candidate source identity does not match the public catalogue.")
        candidates.append(
            PublicCandidate(
                candidate_id=candidate_id,
                text=hit["text"],
                document=doc,
                physical_page_number=hit["physical_page_number"],
            )
        )
        mapping[candidate_id] = hit["chunk_id"]
    candidates.sort(key=lambda row: row.candidate_id)
    return RerankInput(question=question, candidates=candidates), mapping


def prepare(project: Path, source_directory: Path, catalogue_path: Path, experiment: Path) -> dict:
    experiment = private_output(experiment, project)
    if experiment.exists():
        raise FinragError("Reranking preparation requires a new directory; overwrite refused.")
    predictions_path, report_path = (
        source_directory / "predictions_top50.jsonl",
        source_directory / "comparison.json",
    )
    paths = (predictions_path, report_path, catalogue_path)
    hashes = {str(path.resolve()): hash_file(path) for path in paths}
    report = read_json(report_path)
    if (report["split"], report["reference_version"], report["questions"]) != ("dev", "v1", 28):
        raise FinragError("Reranking baseline must contain 28 answerable dev questions under v1.")
    rows = read_jsonl(predictions_path)
    if len(rows) != 28 or len({row["question_id"] for row in rows}) != 28:
        raise FinragError("Reranking baseline must contain 28 unique question IDs.")
    catalogue = [CatalogueDocument.model_validate(row) for row in read_json(catalogue_path)]
    sources, baseline = [], []
    for row in rows:
        selected = row["cached_translation"]
        if not 1 <= len(selected["hits"]) <= 50:
            raise FinragError(
                "Saved candidate count must be between 1 and 50; no truncation allowed."
            )
        source, mapping = project_candidates(selected["query_used"], selected["hits"], catalogue)
        sources.append(source)
        baseline.append(
            {
                "question_id": row["question_id"],
                "input_id": input_identity(source),
                "candidate_id_to_chunk_id": mapping,
                "hits": selected["hits"],
                "metrics": selected["metrics"],
            }
        )
    if len({input_identity(source) for source in sources}) != 28:
        raise FinragError("Duplicate reranking inputs require review before preparing a pilot.")
    aggregate = aggregate_metrics([row["metrics"] for row in baseline])
    if aggregate != report["aggregate"]["cached_translation"]:
        raise FinragError("Saved cached_translation metrics differ from its report.")
    if (
        aggregate["complete_excerpt_coverage@5"]["numerator"],
        aggregate["complete_excerpt_coverage@50"]["numerator"],
    ) != (11, 20):
        raise FinragError(
            "Saved candidate baseline must reproduce excerpt coverage 11/28 and 20/28."
        )
    sources.sort(key=input_identity)
    settings = RerankSettings(
        model=os.environ.get("OPENROUTER_RERANK_MODEL")
        or os.environ.get("OPENROUTER_MODEL")
        or None
    )
    with filesystem_operation("reserve reranking experiment directory", experiment):
        experiment.mkdir(parents=True, exist_ok=False)
    write_atomic(
        experiment / "reranker/inputs.jsonl",
        jsonl_bytes([source.model_dump(mode="json") for source in sources]),
    )
    write_atomic(
        experiment / "reranker/settings.json", json_bytes(settings.model_dump(mode="json"))
    )
    write_atomic(experiment / "evaluation/baseline.jsonl", jsonl_bytes(baseline))
    prepared = {
        str(path.relative_to(experiment)): hash_file(path)
        for path in experiment.rglob("*")
        if path.is_file()
    }
    if any(hash_file(Path(path)) != digest for path, digest in hashes.items()):
        raise FinragError("A historical input changed during preparation.")
    write_atomic(
        experiment / "manifest.json",
        json_bytes(
            {
                "source_hashes": hashes,
                "prepared_hashes": prepared,
                "variant": "cached_translation",
                "split": "dev",
                "reference_version": "v1",
                "baseline_metrics": aggregate,
                "frozen_input_hashes": report["input_hashes"],
                "pilot_input_ids": [input_identity(source) for source in sources[:3]],
                "pilot_rule": "three smallest SHA-256 of reference-free reranking inputs",
                "input_order_rule": (
                    "candidate IDs sorted lexicographically; IDs hash document and chunk identity"
                ),
            }
        ),
    )
    return {
        "questions": 28,
        "variant": "cached_translation",
        "pilot_input_ids": [input_identity(s) for s in sources[:3]],
    }


def sealed_inputs(experiment: Path) -> tuple[dict, list[RerankInput]]:
    manifest = read_json(experiment / "manifest.json")
    for name, digest in manifest["prepared_hashes"].items():
        if hash_file(experiment / name) != digest:
            raise FinragError("Prepared reranking input changed; operation refused.")
    for name, digest in manifest["source_hashes"].items():
        if hash_file(Path(name)) != digest:
            raise FinragError("Historical BM25 input changed; operation refused.")
    inputs = [
        RerankInput.model_validate(row) for row in read_jsonl(experiment / "reranker/inputs.jsonl")
    ]
    if (
        len(inputs) != 28
        or [input_identity(source) for source in inputs[:3]] != manifest["pilot_input_ids"]
    ):
        raise FinragError("Prepared question count or pilot selection changed.")
    return manifest, inputs


def select_pilot_inputs(
    experiment: Path,
    inputs: list[RerankInput],
    question_id: str | None,
    all_questions: bool = False,
) -> list[RerankInput]:
    if all_questions:
        if question_id is not None:
            raise FinragError("Select one question or all prepared questions, not both.")
        return inputs
    if question_id is None:
        return inputs[:3]
    matching = [
        row["input_id"]
        for row in read_jsonl(experiment / "evaluation/baseline.jsonl")
        if row["question_id"] == question_id
    ]
    selected = [source for source in inputs if input_identity(source) in matching]
    if len(matching) != 1 or len(selected) != 1:
        raise FinragError("Selected question must identify exactly one prepared dev input.")
    return selected


def dry_run(
    experiment: Path,
    settings: RerankSettings,
    output: Path,
    metadata: dict | None,
    question_id: str | None = None,
    all_questions: bool = False,
    cache: Path | None = None,
    max_estimated_cost_usd: str | None = None,
) -> dict:
    output = private_output(output, Path.cwd())
    _, inputs = sealed_inputs(experiment)
    selected = select_pilot_inputs(experiment, inputs, question_id, all_questions)
    check_estimated_spend(selected, settings, metadata, max_estimated_cost_usd)
    attempted = (
        {read_json(path)["request_identity"] for path in (cache / "attempts").glob("*.json")}
        if cache is not None
        else set()
    )
    baseline = {
        row["input_id"]: row for row in read_jsonl(experiment / "evaluation/baseline.jsonl")
    }
    rows = []
    for source in selected if question_id is not None else inputs:
        prompt, reserved = token_reservation(source, settings)
        record = reusable_record(cache, source, settings) if cache is not None else None
        cache_status = (
            "reusable"
            if record is not None
            else (
                "previous_attempt_not_replayed"
                if request_identity(source, settings) in attempted
                else "new_call_required"
            )
        )
        rows.append(
            {
                "question_id": baseline[input_identity(source)]["question_id"],
                "input_id": input_identity(source),
                "candidate_count": len(source.candidates),
                "input_utf8_bytes": len(json_bytes(source.model_dump(mode="json"))),
                "candidate_text_characters": sum(len(row.text) for row in source.candidates),
                "estimated_prompt_tokens": prompt,
                "output_token_budget": settings.max_completion_tokens,
                "reserved_total_tokens": reserved,
                "context_compatible": context_check(source, settings, metadata),
                "candidate_input_order": [row.candidate_id for row in source.candidates],
                "pilot_selected": source in selected,
                "cache_status": cache_status,
            }
        )
    report = {
        "model": settings.model,
        "model_configured": settings.model is not None,
        "output_schema_version": settings.output_schema_version,
        "selected_question_id": question_id,
        "all_questions": all_questions,
        "max_estimated_cost_usd": max_estimated_cost_usd,
        "estimation_method": (
            "one token per serialized UTF-8 payload byte plus 512 framing tokens; "
            "conservative admission estimate, not native counts"
        ),
        "max_calls": settings.max_calls,
        "max_total_tokens": settings.max_total_tokens,
        "pilot_reserved_tokens": sum(token_reservation(source, settings)[1] for source in selected),
        "pilot_budget_sufficient": settings.max_calls >= len(selected)
        and sum(token_reservation(s, settings)[1] for s in selected) <= settings.max_total_tokens,
        "pilot_cost_estimate": estimated_cost(selected, settings, metadata),
        "reusable_results": sum(
            row["cache_status"] == "reusable" and row["pilot_selected"] for row in rows
        ),
        "new_calls": sum(
            row["cache_status"] == "new_call_required" and row["pilot_selected"] for row in rows
        ),
        "new_reserved_tokens": sum(
            row["reserved_total_tokens"]
            for row in rows
            if row["cache_status"] == "new_call_required" and row["pilot_selected"]
        ),
        "questions": rows,
        "api_calls": 0,
        "candidate_truncation": False,
        "billing_limit_guaranteed": False,
    }
    write_atomic(output, json_bytes(report))
    return {key: value for key, value in report.items() if key != "questions"}


def pilot(
    experiment: Path,
    settings: RerankSettings,
    metadata: dict,
    cache: Path | None = None,
    question_id: str | None = None,
    all_questions: bool = False,
    max_estimated_cost_usd: str | None = None,
) -> dict:
    _, inputs = sealed_inputs(experiment)
    selected = select_pilot_inputs(experiment, inputs, question_id, all_questions)
    check_estimated_spend(selected, settings, metadata, max_estimated_cost_usd)
    if (
        settings.max_calls < len(selected)
        or sum(token_reservation(source, settings)[1] for source in selected)
        > settings.max_total_tokens
    ):
        raise FinragError(
            "Selected pilot exceeds configured call or token limit; no API call sent."
        )
    if any(context_check(source, settings, metadata) is not True for source in selected):
        raise FinragError("Pilot context check failed; no API call sent.")
    transport = OpenRouterReranker(settings)
    statuses = []
    cache = cache or experiment / "reranker/cache"
    for source in selected:
        record_path = cache / "records" / f"{request_identity(source, settings)}.json"
        reused = record_path.exists()
        try:
            if reused:
                reusable_record(cache, source, settings)
            record = cached_rerank(cache, source, settings, transport, metadata)
            statuses.append(
                {
                    "input_id": input_identity(source),
                    "status": "success",
                    "cache_reused": reused,
                    "requested_model": record["requested_model"],
                    "returned_model": record["returned_model"],
                    "reported_cost": record["reported_cost"],
                    "usage": record["usage"],
                    "latency_seconds": record["latency_seconds"],
                }
            )
        except FinragError as error:
            statuses.append(
                {
                    "input_id": input_identity(source),
                    "status": "failed_or_not_sent",
                    "reason": str(error),
                }
            )
    return {"pilot": statuses, "global_gain_claim": False}


def score_cached_outputs(
    questions: list,
    baseline: list[dict],
    inputs: list[RerankInput],
    cache: Path,
    settings: RerankSettings,
) -> dict:
    by_question = {question.question_id: question for question in questions}
    sources = {input_identity(source): source for source in inputs}
    if len(by_question) != len(questions) or set(by_question) != {
        row["question_id"] for row in baseline
    }:
        raise FinragError("Reranking evaluation questions and baseline identities differ.")
    attempts = {
        read_json(path)["request_identity"]: path for path in (cache / "attempts").glob("*.json")
    }
    rows = []
    for row in baseline:
        question = by_question[row["question_id"]]
        if question.split != "dev" or not question.answerable:
            raise FinragError("Reranking evaluation excludes test and unanswerable questions.")
        source = sources[row["input_id"]]
        pool_metrics = evaluate_candidates(question, row["hits"], True)
        for metric in (*ORDER_METRICS, "complete_excerpt_coverage@50"):
            if pool_metrics[metric] != row["metrics"][metric]:
                raise FinragError("Frozen v1 evidence differs from the saved BM25 baseline.")
        key = request_identity(source, settings)
        path = cache / "records" / f"{key}.json"
        status, hits = "not_attempted", []
        consumption = {}
        if path.exists():
            try:
                record = reusable_record(cache, source, settings)
                order = validate_order(source, record["ordered_candidate_ids"])
                by_chunk = {hit["chunk_id"]: hit for hit in row["hits"]}
                hits = [
                    {**by_chunk[row["candidate_id_to_chunk_id"][candidate]], "rank": rank}
                    for rank, candidate in enumerate(order, 1)
                ]
                status = "success"
                consumption = record
            except (FinragError, KeyError, TypeError):
                status = "failed_validation"
        elif key in attempts:
            outcome = cache / "outcomes" / attempts[key].name
            consumption = read_json(outcome) if outcome.exists() else {}
            status = (
                read_json(outcome)["status"] if outcome.exists() else "pending_or_uncertain_api"
            )
            if status == "success":
                status = "failed_cache_missing"
        metrics = (
            evaluate_candidates(question, hits, True)
            if status == "success"
            else {metric: False for metric in ORDER_METRICS}
        )
        if (
            status == "success"
            and metrics["complete_excerpt_coverage@50"]
            != pool_metrics["complete_excerpt_coverage@50"]
        ):
            raise FinragError("Reordering changed the candidate evidence coverage.")
        rows.append(
            {
                "question_id": question.question_id,
                "status": status,
                "usage": consumption.get("usage"),
                "reported_cost": consumption.get(
                    "reported_cost", (consumption.get("usage") or {}).get("cost")
                ),
                "latency_seconds": consumption.get("latency_seconds"),
                "hits": hits,
                "metrics": {metric: bool(metrics[metric]) for metric in ORDER_METRICS},
                "candidate_complete_excerpt_coverage@50": pool_metrics[
                    "complete_excerpt_coverage@50"
                ],
                "changes": {
                    metric: {
                        "gain": not row["metrics"][metric] and bool(metrics[metric]),
                        "regression": bool(row["metrics"][metric]) and not metrics[metric],
                    }
                    for metric in ORDER_METRICS
                },
            }
        )
    total = len(rows)
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in {row["status"] for row in rows}
    }
    return {
        "questions": total,
        "statuses": counts,
        "predictions": rows,
        "reranked_metrics_failure_as_miss": {
            metric: {"numerator": sum(row["metrics"][metric] for row in rows), "denominator": total}
            for metric in ORDER_METRICS
        },
        "candidate_complete_excerpt_coverage@50": {
            "numerator": sum(row["candidate_complete_excerpt_coverage@50"] for row in rows),
            "denominator": total,
        },
        "failures_and_not_attempted_in_denominator": True,
        "full_experiment_completed": counts.get("success", 0) == total,
        "consumption": {
            "reported_cost_sum_usd": sum(
                row["reported_cost"] for row in rows if row["reported_cost"] is not None
            ),
            "attempted_questions_without_reported_cost": sum(
                row["reported_cost"] is None and row["status"] != "not_attempted" for row in rows
            ),
            "latency_sum_seconds": sum(
                row["latency_seconds"] for row in rows if row["latency_seconds"] is not None
            ),
            "attempted_questions_without_latency": sum(
                row["latency_seconds"] is None and row["status"] != "not_attempted" for row in rows
            ),
            "includes_reused_attempts": True,
        },
    }


def evaluate(
    experiment: Path, settings: RerankSettings, output: Path, cache: Path | None = None
) -> dict:
    output = private_output(output, Path.cwd())
    if output.exists():
        raise FinragError("Reranking evaluation output exists; overwrite refused.")
    if not settings.model:
        raise FinragError("Cached reranking evaluation requires the exact request model.")
    manifest, inputs = sealed_inputs(experiment)
    questions = [
        q
        for q in load_dev_questions(Path.cwd() / "data/private/evaluation/finrag_v1")
        if q.answerable
    ]
    for name, digest in manifest["frozen_input_hashes"].items():
        if (
            Path(name).name in ("dev.jsonl", "dataset_manifest.json")
            and hash_file(Path(name)) != digest
        ):
            raise FinragError("Frozen dev input differs from the routed BM25 snapshot.")
    report = score_cached_outputs(
        questions,
        read_jsonl(experiment / "evaluation/baseline.jsonl"),
        inputs,
        cache or experiment / "reranker/cache",
        settings,
    )
    if report["questions"] != 28:
        raise FinragError("Reranking evaluation requires all 28 answerable dev questions.")
    report["baseline_metrics"] = manifest["baseline_metrics"]
    sealed_inputs(experiment)
    with filesystem_operation("reserve reranking evaluation output", output):
        output.mkdir(parents=True, exist_ok=False)
    write_atomic(output / "evaluation.json", json_bytes(report))
    write_atomic(output / "rapport.md", render_evaluation(report).encode("utf-8"))
    return {key: value for key, value in report.items() if key != "predictions"}


def render_evaluation(report: dict) -> str:
    lines = [
        "# Reclassement du pool BM25 sauvegardé",
        "",
        "Dev v1, variante cached_translation uniquement. Échecs et questions non tentées "
        "restent au dénominateur et sont comptés comme des misses d'exécution. La couverture "
        "du pool @50 est distincte du succès du reclasseur.",
        "",
        "| Variante | Hit@1 | Hit@5 | Extraits complets @5 |",
        "| --- | --- | --- | --- |",
    ]
    for name, metrics in (
        ("BM25", report["baseline_metrics"]),
        ("Reclassement, échecs inclus", report["reranked_metrics_failure_as_miss"]),
    ):
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(
                f"{metrics[key]['numerator']}/{metrics[key]['denominator']}"
                for key in ORDER_METRICS
            )
            + " |"
        )
    pool = report["candidate_complete_excerpt_coverage@50"]
    lines.extend(
        [
            "",
            "Couverture complète du pool @50 inchangée : "
            f"{pool['numerator']}/{pool['denominator']}.",
            "",
            "États : " + json.dumps(report["statuses"], ensure_ascii=False),
            "",
            "Consommation des tentatives du cache sélectionné, résultat réutilisé inclus : "
            + json.dumps(report["consumption"], ensure_ascii=False)
            + ". Les sommes portent uniquement sur les coûts et latences fournis ; "
            "les valeurs manquantes restent inconnues. Les anciens pilotes non importés "
            "ne sont pas inclus dans cette somme.",
            "",
        ]
    )
    if not report["full_experiment_completed"]:
        lines.append(
            "Expérience incomplète : les résultats d'un pilote ne permettent "
            "aucune annonce de gain global."
        )
    lines.extend(
        [
            "",
            "| ID | État | Gains | Régressions / misses d'exécution |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in report["predictions"]:
        gains = [key for key, value in row["changes"].items() if value["gain"]]
        regressions = [key for key, value in row["changes"].items() if value["regression"]]
        lines.append(
            f"| {row['question_id']} | {row['status']} | "
            f"{', '.join(gains) or 'aucun'} | {', '.join(regressions) or 'aucune'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Cached BM25 candidate reranking, dev only")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument(
        "--source", type=Path, default=Path("artifacts/private/bm25_company_routing_dev/run_1")
    )
    prep.add_argument(
        "--catalogue", type=Path, default=Path("experiments/company_routing/corpus_catalog.json")
    )
    prep.add_argument("--experiment", type=Path, required=True)
    for name in ("dry-run", "pilot", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--experiment", type=Path, required=True)
        command.add_argument("--model")
        command.add_argument("--max-calls", type=int)
        command.add_argument("--max-total-tokens", type=int)
        command.add_argument("--max-completion-tokens", type=int)
        command.add_argument("--reasoning-effort", choices=("none", "high"))
        command.add_argument("--output-schema-version", choices=("v1", "v2"))
        command.add_argument("--omit-temperature", action="store_true")
        if name in ("dry-run", "evaluate"):
            command.add_argument("--output", type=Path, required=True)
        if name in ("dry-run", "pilot"):
            command.add_argument("--model-snapshot", type=Path, required=name == "pilot")
            selection = command.add_mutually_exclusive_group()
            selection.add_argument("--question-id")
            selection.add_argument("--all-questions", action="store_true")
            command.add_argument("--max-estimated-cost-usd")
        command.add_argument("--cache-directory", type=Path)
    args = parser.parse_args()
    try:
        experiment = private_output(args.experiment, Path.cwd())
        if args.command == "prepare":
            result = prepare(Path.cwd(), args.source, args.catalogue, experiment)
        else:
            values = read_json(experiment / "reranker/settings.json")
            values["model"] = (
                args.model
                or os.environ.get("OPENROUTER_RERANK_MODEL")
                or os.environ.get("OPENROUTER_MODEL")
                or values["model"]
            )
            for key in (
                "max_calls",
                "max_total_tokens",
                "max_completion_tokens",
                "reasoning_effort",
                "output_schema_version",
            ):
                if getattr(args, key) is not None:
                    values[key] = getattr(args, key)
            if args.omit_temperature:
                values["temperature_zero"] = False
            settings = RerankSettings.model_validate(values)
            metadata = (
                load_model_snapshot(args.model_snapshot, settings)
                if getattr(args, "model_snapshot", None)
                else None
            )
            cache_arg = getattr(args, "cache_directory", None)
            cache = private_output(cache_arg, Path.cwd()) if cache_arg else None
            if args.command == "dry-run":
                result = dry_run(
                    experiment,
                    settings,
                    args.output,
                    metadata,
                    args.question_id,
                    args.all_questions,
                    cache,
                    args.max_estimated_cost_usd,
                )
            elif args.command == "pilot":
                result = pilot(
                    experiment,
                    settings,
                    metadata,
                    cache,
                    args.question_id,
                    args.all_questions,
                    args.max_estimated_cost_usd,
                )
            else:
                result = evaluate(experiment, settings, args.output, cache)
        print(json.dumps(result))
    except ValidationError:
        print("Candidate reranking configuration or prepared fields are invalid.", file=sys.stderr)
        raise SystemExit(2) from None
    except (FinragError, OSError) as error:
        print(f"Candidate reranking experiment failed: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
