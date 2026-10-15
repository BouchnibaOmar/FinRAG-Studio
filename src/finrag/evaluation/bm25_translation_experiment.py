import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from finrag.config import FinragError, private_output
from finrag.domain.dense import ModelChunk
from finrag.domain.evaluation import EvaluationQuestion
from finrag.domain.retrieval import LexicalConfig
from finrag.evaluation.document_oracle_metrics import aggregate_metrics, evaluate_candidates
from finrag.evaluation.question_translation import (
    OpenRouterTranslator,
    TranslationInput,
    TranslationSettings,
    fetch_official_model_metadata,
    input_identity,
    official_price_estimate,
    request_payload,
    timestamp,
    translation_checks,
)
from finrag.evaluation.split_guard import load_dev_questions
from finrag.evaluation.translation_cache import (
    attempted_without_cache,
    cached_translation,
    import_corrections,
    load_translation_set,
    private_translation_cache,
    reserved_tokens,
    save_translation_set,
)
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes, read_json, read_jsonl, write_atomic
from finrag.retrieval.bm25 import BM25

REPORT_LANGUAGES = {
    "bnp_urd_2025_en.pdf": "en",
    "casa_urd_2025_en.pdf": "en",
    "sg_urd_2026_fy2025_fr.pdf": "fr",
}
EXPECTED_COHORTS = {"fr->en": (6, 15), "fr->fr": (7, 7), "en->en": (3, 4), "en->fr": (0, 2)}


def translation_plan(
    questions: list[EvaluationQuestion], names: dict[str, str]
) -> tuple[list, list]:
    inputs, identities = [], []
    for question in questions:
        if question.split != "dev":
            raise FinragError("Translation plan excludes the test split.")
        if not question.answerable:
            continue
        target = REPORT_LANGUAGES[names[question.document_id]]
        source = TranslationInput(
            question=question.question, source_language=question.language, target_language=target
        )
        needs_translation = question.language != target
        identities.append(
            {
                "question_id": question.question_id,
                "input_id": input_identity(source),
                "needs_translation": needs_translation,
                "target_language": target,
            }
        )
        if needs_translation:
            inputs.append(source.model_dump(mode="json"))
    return inputs, identities


def load_control(
    project_root: Path,
) -> tuple[list[ModelChunk], LexicalConfig, list[EvaluationQuestion], dict[str, BM25]]:
    questions = [
        q
        for q in load_dev_questions(project_root / "data/private/evaluation/finrag_v1")
        if q.answerable
    ]
    phase = project_root / "artifacts/private/phase2b"
    request = read_json(phase / "faiss_exchange/request.json")
    path = phase / "model_chunks.jsonl"
    if hash_file(path) != request["inputs"]["chunks"]["sha256"]:
        raise FinragError("Phase 2B control passage hash changed.")
    chunks = [ModelChunk.model_validate(row) for row in read_jsonl(path)]
    if len(chunks) != 8325 or len({c.chunk_id for c in chunks}) != 8325:
        raise FinragError("Translation control requires the same 8325 unique Phase 2B passages.")
    config = LexicalConfig.model_validate(
        read_json(project_root / "experiments/phase2a/configs/B.json")
    )
    if (config.k1, config.b, config.tokenizer_version) != (1.2, 0.75, "unicode-financial-v1"):
        raise FinragError("Translation control BM25 configuration changed.")
    grouped = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.document_id].append(chunk)
    indexes = {doc: BM25(rows, config) for doc, rows in grouped.items()}
    return chunks, config, questions, indexes


def original_results(
    questions: list[EvaluationQuestion], indexes: dict[str, BM25], previous: list[dict]
) -> list[dict]:
    by_id = {row["question_id"]: row for row in previous}
    if len(previous) != 28 or set(by_id) != {q.question_id for q in questions}:
        raise FinragError("Original oracle must match all 28 answerable dev questions.")
    output = []
    for question in questions:
        index = indexes[question.document_id]
        hits = [
            h.model_dump(mode="json") for h in index.search(question.question, len(index.chunks))
        ]
        metrics = evaluate_candidates(question, hits, True)
        stored = by_id[question.question_id]
        if hits[:50] != stored["hits"] or json.loads(json_bytes(metrics)) != stored["metrics"]:
            raise FinragError(f"Original oracle ranking/metrics differ for {question.question_id}.")
        output.append({"question_id": question.question_id, "hits": hits[:50], "metrics": metrics})
    return output


def cohort_results(questions: list[EvaluationQuestion], rows: list[dict], names: dict) -> dict:
    by_id = {row["question_id"]: row for row in rows}
    pairs, banks = defaultdict(list), defaultdict(list)
    for question in questions:
        pair = question.language + "->" + REPORT_LANGUAGES[names[question.document_id]]
        pairs[pair].append(by_id[question.question_id]["metrics"])
        banks[question.company].append(by_id[question.question_id]["metrics"])
    return {
        "overall": aggregate_metrics([row["metrics"] for row in rows]),
        "language_pairs": {key: aggregate_metrics(group) for key, group in pairs.items()},
        "banks": {key: aggregate_metrics(group) for key, group in banks.items()},
    }


def prepare(project_root: Path, experiment: Path) -> dict:
    experiment = private_output(experiment, project_root)
    if experiment.exists():
        raise FinragError("Prepare requires a new experiment directory.")
    protected = [
        project_root / "data/private/evaluation/finrag_v1/dev.jsonl",
        project_root / "data/private/evaluation/finrag_v1/dataset_manifest.json",
        project_root / "artifacts/private/phase2b/model_chunks.jsonl",
        project_root / "artifacts/private/phase2b/faiss_exchange/request.json",
        project_root / "artifacts/private/document_oracle_dev/bm25_top50.jsonl",
        project_root / "experiments/phase2a/configs/B.json",
    ]
    before = {str(path): hash_file(path) for path in protected}
    chunks, config, questions, indexes = load_control(project_root)
    previous = read_jsonl(project_root / "artifacts/private/document_oracle_dev/bm25_top50.jsonl")
    originals = original_results(questions, indexes, previous)
    names = {chunk.document_id: chunk.source_filename for chunk in chunks}
    groups = cohort_results(questions, originals, names)
    for pair, (successes, count) in EXPECTED_COHORTS.items():
        actual = groups["language_pairs"][pair]["complete_excerpt_coverage@50"]
        if (actual["numerator"], actual["denominator"]) != (successes, count):
            raise FinragError(f"Original language cohort {pair} differs from the supplied control.")
    inputs, identities = translation_plan(questions, names)
    if (
        len(inputs) != 17
        or len({input_identity(TranslationInput.model_validate(i)) for i in inputs}) != 17
    ):
        raise FinragError("Translation experiment requires 17 distinct cross-language inputs.")
    settings = TranslationSettings(model=os.environ.get("OPENROUTER_MODEL") or None)
    statistics = {
        doc: {
            "passages": len(index.chunks),
            "average_length": index.average_length,
            "total_lexical_tokens": sum(index.lengths),
        }
        for doc, index in indexes.items()
    }
    after = {str(path): hash_file(path) for path in protected}
    if before != after:
        raise FinragError("An original source changed while preparing translation control.")
    write_atomic(experiment / "translator/inputs.jsonl", jsonl_bytes(inputs))
    write_atomic(
        experiment / "translator/inputs_digest.json",
        json_bytes({"sha256": hash_file(experiment / "translator/inputs.jsonl")}),
    )
    write_atomic(
        experiment / "translator/settings.json", json_bytes(settings.model_dump(mode="json"))
    )
    write_atomic(
        experiment / "evaluation/references.jsonl",
        jsonl_bytes([q.model_dump(mode="json") for q in questions]),
    )
    write_atomic(experiment / "evaluation/identities.json", json_bytes(identities))
    write_atomic(experiment / "evaluation/original_top50.jsonl", jsonl_bytes(originals))
    manifest = {
        "timestamp": timestamp(),
        "source_hashes": before,
        "source_hashes_after": after,
        "document_statistics": statistics,
        "bm25_config": config.model_dump(mode="json"),
        "groups_original": groups,
        "translated_inputs": 17,
        "unchanged_queries": 11,
        "test_read": False,
        "paid_calls": 0,
        "translation_results": None,
    }
    prepared = {
        str(path.relative_to(experiment)): hash_file(path)
        for path in experiment.rglob("*")
        if path.is_file()
    }
    manifest["prepared_hashes"] = prepared
    write_atomic(experiment / "experiment.json", json_bytes(manifest))
    write_atomic(
        experiment / "rapport_preparation.md",
        (
            "# Expérience BM25 : préparation hors ligne\n\n"
            "Le contrôle original est reproduit exactement, candidats et métriques "
            "par question.\n\n"
            "| Langues | Couverture complète des extraits v1 @50 |\n| --- | --- |\n"
            "| FR→EN | 6/15 |\n| FR→FR | 7/7 |\n| EN→EN | 3/4 |\n"
            "| EN→FR | 0/2 |\n| Total | 16/28 |\n\n"
            "17 requêtes à traduire ; 11 questions inchangées. Le traducteur lit uniquement "
            "translator/inputs.jsonl et sa configuration. Les références, IDs et résultats sont "
            "séparés dans evaluation/. Aucun appel payant, traduction réelle ou résultat traduit "
            "n'est disponible. Le document attendu reste un oracle diagnostique.\n"
        ).encode(),
    )
    create_review_sheet(experiment, None)
    return {
        "experiment": str(experiment),
        "original_complete_excerpt50": "16/28",
        "translation_inputs": 17,
        "unchanged": 11,
        "paid_calls": 0,
    }


def translator_inputs(experiment: Path) -> list[TranslationInput]:
    if (
        hash_file(experiment / "translator/inputs.jsonl")
        != read_json(experiment / "translator/inputs_digest.json")["sha256"]
    ):
        raise FinragError("Translator input questions changed.")
    inputs = [
        TranslationInput.model_validate(row)
        for row in read_jsonl(experiment / "translator/inputs.jsonl")
    ]
    if len(inputs) != 17 or any(row.source_language == row.target_language for row in inputs):
        raise FinragError("Translator inputs must contain exactly the 17 cross-language questions.")
    return inputs


def dry_run(
    experiment: Path, settings: TranslationSettings, output: Path, pricing: Path | None
) -> dict:
    output = private_output(output, Path.cwd())
    inputs = translator_inputs(experiment)
    cache_root = private_translation_cache(experiment)
    # No translator instance or HTTP connection is constructed by this command.
    requests = [
        {"input_id": input_identity(source), "payload": request_payload(source, settings)}
        for source in inputs
    ]
    for planned in requests:
        estimated_prompt = (
            sum(
                len(message["content"].encode("utf-8"))
                for message in planned["payload"]["messages"]
            )
            + 256
        )
        planned["prompt_admission_estimate"] = estimated_prompt
        planned["within_prompt_allowance"] = estimated_prompt <= settings.prompt_token_allowance
    cost = None
    if pricing is not None:
        snapshot = read_json(pricing)
        cost = official_price_estimate(snapshot["model"], settings, len(inputs))
    report = {
        "requests": requests,
        "model_configured": settings.model is not None,
        "key_configured": bool(os.environ.get("OPENROUTER_API_KEY")),
        "max_calls": settings.max_calls,
        "max_total_tokens": settings.max_total_tokens,
        "reserved_tokens_per_call": reserved_tokens(settings),
        "total_reservation_if_no_cache": len(inputs) * reserved_tokens(settings),
        "budget_sufficient_without_cache": len(inputs) <= settings.max_calls
        and len(inputs) * reserved_tokens(settings) <= settings.max_total_tokens,
        "all_prompt_estimates_within_allowance": all(
            row["within_prompt_allowance"] for row in requests
        ),
        "cost_estimate": cost,
        "paid_calls": 0,
        "token_limit_scope": "admission reservations, supplier usage overrun stops later calls; "
        "not a guarantee of native tokenizer counts or supplier billing",
        "cache": str(cache_root),
    }
    write_atomic(output, json_bytes(report))
    return {key: value for key, value in report.items() if key != "requests"}


def translate(experiment: Path, settings: TranslationSettings, resume: bool = False) -> Path:
    inputs = translator_inputs(experiment)
    translator = OpenRouterTranslator(settings)
    root = private_translation_cache(experiment)
    identities = {
        row["input_id"]: row["question_id"]
        for row in read_json(experiment / "evaluation/identities.json")
    }
    entries, missing = [], []
    for source in inputs:
        identity = input_identity(source)
        if resume and attempted_without_cache(root, source, settings):
            missing.append(identity)
            continue
        try:
            entries.append(cached_translation(root, source, settings, translator))
        except FinragError as error:
            raise FinragError(f"{identities[identity]}: {error}") from error
    result = save_translation_set(root, entries, missing_input_ids=missing)
    create_review_sheet(experiment, result)
    if missing:
        print(
            f"Incomplete translation set: {len(entries)}/{len(inputs)} cached; "
            "previous unsuccessful attempts were not replayed: "
            + ", ".join(identities[identity] for identity in missing),
            file=sys.stderr,
        )
    return result


def create_review_sheet(experiment: Path, translation_set: Path | None) -> Path:
    inputs = {input_identity(source): source for source in translator_inputs(experiment)}
    identities = read_json(experiment / "evaluation/identities.json")
    cached = (
        load_translation_set(private_translation_cache(experiment), translation_set)
        if translation_set
        else {}
    )
    rows = []
    for item in identities:
        if not item["needs_translation"]:
            continue
        source = inputs[item["input_id"]]
        record = cached.get(item["input_id"])
        rows.append(
            {
                "question_id": item["question_id"],
                "input_id": item["input_id"],
                "original": source.question,
                "source_language": source.source_language,
                "target_language": source.target_language,
                "translation": record["translation"] if record else None,
                "checks": record["checks"] if record else None,
                "requested_model": record["requested_model"] if record else None,
                "returned_model": record["returned_model"] if record else None,
                "review_status": "candidate_not_validated" if record else "not_translated",
            }
        )
    name = translation_set.stem if translation_set else "pending"
    path = experiment / "review" / f"pairs-{name}.json"
    write_atomic(path, json_bytes(rows))
    return path


def selected_query(
    question: EvaluationQuestion, target: str, translations: dict
) -> tuple[str, str]:
    if question.language == target:
        return question.question, "unchanged"
    source = TranslationInput(
        question=question.question, source_language=question.language, target_language=target
    )
    record = translations.get(input_identity(source))
    if record is None or record["input"] != source.model_dump(mode="json"):
        raise FinragError("Missing or stale cached translation for an answerable dev query.")
    checks = translation_checks(source, record["translation"])
    if (
        not checks["numbers_preserved"]
        or not checks["numeric_years_preserved"]
        or not checks["numeric_dates_preserved_literally"]
    ):
        raise FinragError(
            "Numeric/date checks failed; import a local correction before evaluation."
        )
    return record["translation"], record["source"]


def gain_regression(original: dict, translated: dict) -> dict:
    keys = [key for key in original if key.startswith(("hit@", "complete_excerpt_coverage@"))]
    return {
        key: {
            "gain": not original[key] and bool(translated[key]),
            "regression": bool(original[key]) and not translated[key],
        }
        for key in keys
    }


def evaluate(experiment: Path, translation_set: Path, output: Path) -> dict:
    output = private_output(output, Path.cwd())
    if output.exists():
        raise FinragError("Evaluation requires a new result directory.")
    manifest = read_json(experiment / "experiment.json")
    for name, digest in manifest["prepared_hashes"].items():
        if hash_file(experiment / name) != digest:
            raise FinragError("Prepared experiment inputs changed.")
    for name, digest in manifest["source_hashes"].items():
        if hash_file(Path(name)) != digest:
            raise FinragError("Frozen v1/control/oracle inputs changed.")
    chunks, config, questions, indexes = load_control(Path.cwd())
    names = {chunk.document_id: chunk.source_filename for chunk in chunks}
    originals = original_results(
        questions, indexes, read_jsonl(experiment / "evaluation/original_top50.jsonl")
    )
    original_by_id = {row["question_id"]: row for row in originals}
    translations = load_translation_set(private_translation_cache(experiment), translation_set)
    if len({record["requested_model"] for record in translations.values()}) > 1:
        raise FinragError("One explicitly requested translation model is required per comparison.")
    required = {
        row["input_id"]
        for row in read_json(experiment / "evaluation/identities.json")
        if row["needs_translation"]
    }
    if set(translations) != required:
        raise FinragError(
            "Evaluation requires exactly the 17 cached translations, not a partial subset."
        )
    rows, comparisons = [], []
    for question in questions:
        target = REPORT_LANGUAGES[names[question.document_id]]
        query, origin = selected_query(question, target, translations)
        index = indexes[question.document_id]
        hits = [hit.model_dump(mode="json") for hit in index.search(query, len(index.chunks))]
        row = {
            "question_id": question.question_id,
            "query_used": query,
            "translation_source": origin,
            "hits": hits[:50],
            "metrics": evaluate_candidates(question, hits, True),
        }
        original = original_by_id[question.question_id]
        if origin == "unchanged" and (
            row["hits"] != original["hits"] or row["metrics"] != original["metrics"]
        ):
            raise FinragError("An untranslated question changed its ranking or metrics.")
        rows.append(row)
        comparisons.append(
            {
                "question_id": question.question_id,
                "language_pair": question.language + "->" + target,
                "bank": question.company,
                "original": original["metrics"],
                "translated": row["metrics"],
                "changes": gain_regression(original["metrics"], row["metrics"]),
            }
        )
    statistics = {
        doc: {
            "passages": len(index.chunks),
            "average_length": index.average_length,
            "total_lexical_tokens": sum(index.lengths),
        }
        for doc, index in indexes.items()
    }
    if statistics != manifest["document_statistics"]:
        raise FinragError("Document-local BM25 statistics changed.")
    after = {name: hash_file(Path(name)) for name in manifest["source_hashes"]}
    if after != manifest["source_hashes"]:
        raise FinragError("An original source changed during translation evaluation.")
    report = {
        "original": cohort_results(questions, originals, names),
        "translated": cohort_results(questions, rows, names),
        "comparison": comparisons,
        "translation_set_sha256": hash_file(translation_set),
        "translations_automatically_validated": False,
        "oracle_document_from_annotations": True,
        "unchanged_queries_verified": sum(row["translation_source"] == "unchanged" for row in rows),
        "requested_model": next(
            iter({record["requested_model"] for record in translations.values()})
        ),
        "input_hashes_before": manifest["source_hashes"],
        "input_hashes_after": after,
    }
    write_atomic(output / "translated_top50.jsonl", jsonl_bytes(rows))
    write_atomic(output / "comparison.json", json_bytes(report))
    write_atomic(output / "rapport.md", render_comparison(report).encode("utf-8"))
    return {
        "report": str(output / "rapport.md"),
        "unchanged_queries_verified": report["unchanged_queries_verified"],
    }


def metric_fraction(metrics: dict, key: str) -> str:
    value = metrics[key]
    return f"{value['numerator']}/{value['denominator']}"


def render_comparison(report: dict) -> str:
    lines = [
        "# BM25 oracle : question originale et traduction",
        "",
        "Dev uniquement, références v1. Document attendu issu des annotations : "
        "diagnostic oracle, pas routage déployable. Traductions candidates, "
        "non validées automatiquement.",
        "",
        "| Requête | Hit1 | Hit5 | Hit10 | Hit50 | Hit5 pages | Extraits complets @5/@10/@50 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label in ("original", "translated"):
        metrics = report[label]["overall"]
        counts = [
            metric_fraction(metrics, key)
            for key in ("hit@1", "hit@5", "hit@10", "hit@50", "distinct_page_hit@5")
        ]
        covers = [metric_fraction(metrics, f"complete_excerpt_coverage@{k}") for k in (5, 10, 50)]
        lines.append(f"| {label} | {' | '.join(counts)} | {' / '.join(covers)} |")
    lines.extend(
        [
            "",
            "| Groupe | Extraits complets @50 original | Extraits complets @50 traduction |",
            "| --- | --- | --- |",
        ]
    )
    for grouping in ("language_pairs", "banks"):
        for name, metrics in report["original"][grouping].items():
            translated = report["translated"][grouping][name]
            lines.append(
                f"| {name} | {metric_fraction(metrics, 'complete_excerpt_coverage@50')} | "
                f"{metric_fraction(translated, 'complete_excerpt_coverage@50')} |"
            )
    lines.extend(
        [
            "",
            "| ID | Hit5 original→traduction | Extraits complets50 original→traduction | "
            "Première page (segment) original→traduction | Premier extrait original→traduction |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["comparison"]:
        original, translated = row["original"], row["translated"]
        columns = [
            str(original[key]) + "→" + str(translated[key])
            for key in (
                "hit@5",
                "complete_excerpt_coverage@50",
                "first_reference_segment_rank",
                "first_supporting_excerpt_rank",
            )
        ]
        lines.append(f"| {row['question_id']} | {' | '.join(columns)} |")
    lines.extend(
        [
            "",
            "Les autres Hit@k, couvertures, ventilations par banque, rangs de pages/extraits "
            "et gains/régressions par question sont dans comparison.json. Les 11 questions "
            "sans traduction conservent exactement leurs résultats. Aucun mélange de requêtes "
            "originales et traduites, référence alternative ou changement de paramètres.",
            "",
        ]
    )
    return "\n".join(lines)


def run_command() -> None:
    parser = argparse.ArgumentParser(description="Controlled dev BM25 query translation experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--experiment", type=Path, required=True)
    pricing = commands.add_parser("pricing")
    pricing.add_argument("--model", required=True)
    pricing.add_argument("--output", type=Path, required=True)
    for name in ("dry-run", "translate", "evaluate", "import-corrections", "review"):
        command = commands.add_parser(name)
        command.add_argument("--experiment", type=Path, required=True)
        if name in ("dry-run", "translate"):
            command.add_argument("--model")
            command.add_argument("--omit-temperature", action="store_true")
            command.add_argument("--max-calls", type=int)
            command.add_argument("--max-total-tokens", type=int)
        if name == "translate":
            command.add_argument(
                "--resume",
                action="store_true",
                help="Reuse valid caches and skip previously unsuccessful attempts; no replay.",
            )
        if name in ("evaluate", "review", "import-corrections"):
            command.add_argument("--translation-set", type=Path, required=True)
        if name in ("evaluate", "dry-run"):
            command.add_argument("--output", type=Path, required=True)
        if name == "dry-run":
            command.add_argument("--pricing-snapshot", type=Path)
        if name == "import-corrections":
            command.add_argument("--corrections", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pricing":
        output = private_output(args.output, Path.cwd())
        if output.exists():
            raise FinragError("Official pricing snapshot output already exists.")
        write_atomic(output, json_bytes(fetch_official_model_metadata(args.model)))
        print(json.dumps({"pricing_snapshot": str(output), "generation_calls": 0}))
        return
    experiment = private_output(args.experiment, Path.cwd())
    if args.command == "prepare":
        result = prepare(Path.cwd(), experiment)
    elif args.command in ("dry-run", "translate"):
        values = read_json(experiment / "translator/settings.json")
        values["model"] = args.model or os.environ.get("OPENROUTER_MODEL") or values["model"]
        if args.omit_temperature:
            values["temperature_zero"] = False
        for field in ("max_calls", "max_total_tokens"):
            value = getattr(args, field)
            if value is not None:
                values[field] = value
        settings = TranslationSettings.model_validate(values)
        result = (
            dry_run(experiment, settings, args.output, args.pricing_snapshot)
            if args.command == "dry-run"
            else str(translate(experiment, settings, args.resume))
        )
    elif args.command == "evaluate":
        result = evaluate(experiment, args.translation_set, args.output)
    elif args.command == "review":
        result = str(create_review_sheet(experiment, args.translation_set))
    else:
        new_set = import_corrections(
            private_translation_cache(experiment),
            args.translation_set,
            args.corrections,
            missing_inputs={
                input_identity(source): source for source in translator_inputs(experiment)
            },
        )
        create_review_sheet(experiment, new_set)
        result = str(new_set)
    print(json.dumps(result))


def main() -> None:
    try:
        run_command()
    except FinragError as error:
        print(f"Translation experiment failed: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
