import argparse
import json
import sys
from pathlib import Path

from finrag.config import FinragError, filesystem_operation, private_output
from finrag.evaluation.bm25_translation_experiment import load_control
from finrag.evaluation.document_oracle_metrics import aggregate_metrics, evaluate_candidates
from finrag.evaluation.question_translation import TranslationInput, translation_checks
from finrag.evaluation.translation_cache import load_translation_set
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes, read_json, read_jsonl, write_atomic
from finrag.retrieval.company_router import CatalogueDocument
from finrag.retrieval.routed_bm25 import CachedQueryTranslation, RoutedBM25

METRICS = (
    "hit@1",
    "hit@5",
    "hit@10",
    "hit@50",
    "distinct_page_hit@5",
    "complete_excerpt_coverage@5",
    "complete_excerpt_coverage@10",
    "complete_excerpt_coverage@50",
)


def metric_changes(before: dict, after: dict) -> dict:
    return {
        key: {
            "gain": not before[key] and bool(after[key]),
            "regression": bool(before[key]) and not after[key],
        }
        for key in METRICS
    }


def cached_queries(records: dict) -> list[CachedQueryTranslation]:
    output = []
    for record in records.values():
        source = TranslationInput.model_validate(record["input"])
        checks = translation_checks(source, record["translation"])
        if not all(
            checks[key]
            for key in (
                "numbers_preserved",
                "numeric_years_preserved",
                "numeric_dates_preserved_literally",
            )
        ):
            raise FinragError("Cached query translation failed numeric/date checks.")
        output.append(
            CachedQueryTranslation(
                original_question=source.question,
                translated_question=record["translation"],
                target_language=source.target_language,
                provenance=record["source"],
            )
        )
    return output


def run(project: Path, catalogue_path: Path, translation_set: Path, output: Path) -> dict:
    output = private_output(output, project)
    if output.exists():
        raise FinragError("Company routing output exists; overwrite refused.")
    experiment = project / "artifacts/private/bm25_query_translation_dev"
    oracle_report_path = experiment / "evaluation_run_2/comparison.json"
    translated_oracle_path = experiment / "evaluation_run_2/translated_top50.jsonl"
    original_oracle_path = experiment / "evaluation/original_top50.jsonl"
    manifest = read_json(experiment / "experiment.json")
    protected = {Path(name): digest for name, digest in manifest["source_hashes"].items()}
    for name, digest in manifest["prepared_hashes"].items():
        protected[experiment / name] = digest
    for path in (
        catalogue_path,
        translation_set,
        oracle_report_path,
        translated_oracle_path,
        original_oracle_path,
    ):
        protected[path] = hash_file(path)
    records = load_translation_set(experiment / "translator/cache", translation_set)
    for record in records.values():
        protected[Path(record["cache_entry"]["path"])] = record["cache_entry"]["sha256"]
    for path, digest in protected.items():
        if hash_file(path) != digest:
            raise FinragError(f"Protected routing input changed: {path.name}.")
    oracle_report = read_json(oracle_report_path)
    if oracle_report["translation_set_sha256"] != hash_file(translation_set):
        raise FinragError("Translation set differs from evaluation_run_2.")
    catalogue = [CatalogueDocument.model_validate(row) for row in read_json(catalogue_path)]
    chunks, config, questions, _ = load_control(project)
    if len(records) != 17:
        raise FinragError("Routing comparison requires the complete 17-translation set.")
    search = RoutedBM25(chunks, catalogue, config, cached_queries(records))
    original_oracle = {row["question_id"]: row for row in read_jsonl(original_oracle_path)}
    translated_oracle = {row["question_id"]: row for row in read_jsonl(translated_oracle_path)}
    ids = {question.question_id for question in questions}
    if set(original_oracle) != ids or set(translated_oracle) != ids:
        raise FinragError("Oracle predictions must match the 28 answerable dev questions.")
    rows, comparisons = [], []
    for question in questions:
        # Annotations stay in this evaluator; the search receives only the original text.
        arms = {}
        for label, translated, oracle in (
            ("original", False, original_oracle[question.question_id]),
            ("cached_translation", True, translated_oracle[question.question_id]),
        ):
            ranking = search.search(question.question, use_cached_translation=translated)
            hits = [hit.model_dump(mode="json") for hit in ranking.hits]
            metrics = json.loads(json_bytes(evaluate_candidates(question, hits, True)))
            expected_only = ranking.route.document_ids == [question.document_id]
            oracle_query = oracle.get("query_used", question.question)
            comparable = expected_only and ranking.query_used == oracle_query
            identical = hits[:50] == oracle["hits"] and metrics == oracle["metrics"]
            if comparable and not identical:
                raise FinragError(
                    f"Singleton route differs from its oracle: {question.question_id}."
                )
            arms[label] = {
                "route": ranking.route.model_dump(mode="json"),
                "expected_document_present": question.document_id in ranking.route.document_ids,
                "query_used": ranking.query_used,
                "translation_status": ranking.translation_status,
                "corpus_statistics": ranking.corpus_statistics,
                "hits": hits[:50],
                "metrics": metrics,
                "oracle_identity_comparable": comparable,
                "oracle_top50_and_metrics_identical": identical if comparable else None,
            }
        comparisons.append(
            {
                "question_id": question.question_id,
                "original_vs_original_oracle": metric_changes(
                    original_oracle[question.question_id]["metrics"], arms["original"]["metrics"]
                ),
                "cached_vs_translated_oracle": metric_changes(
                    translated_oracle[question.question_id]["metrics"],
                    arms["cached_translation"]["metrics"],
                ),
                "cached_vs_original": metric_changes(
                    arms["original"]["metrics"], arms["cached_translation"]["metrics"]
                ),
            }
        )
        rows.append({"question_id": question.question_id, **arms})
    for path, digest in protected.items():
        if hash_file(path) != digest:
            raise FinragError(f"Protected routing input changed during execution: {path.name}.")
    aggregate = {
        "original_oracle": aggregate_metrics([row["metrics"] for row in original_oracle.values()]),
        "translated_oracle": aggregate_metrics(
            [row["metrics"] for row in translated_oracle.values()]
        ),
        **{
            label: aggregate_metrics([row[label]["metrics"] for row in rows])
            for label in ("original", "cached_translation")
        },
    }
    report = {
        "aggregate": aggregate,
        "comparisons": comparisons,
        "routing_expected_document_present": sum(
            row["original"]["expected_document_present"] for row in rows
        ),
        "questions": len(rows),
        "split": "dev",
        "reference_version": "v1",
        "cached_translations_are_not_an_autonomous_translation_pipeline": True,
        "input_hashes": {str(path): digest for path, digest in protected.items()},
    }
    # Reserve a new directory only after all read-only checks and comparisons succeeded.
    with filesystem_operation("reserve routing results directory", output):
        output.mkdir(parents=True, exist_ok=False)
    write_atomic(output / "predictions_top50.jsonl", jsonl_bytes(rows))
    write_atomic(output / "comparison.json", json_bytes(report))
    write_atomic(output / "rapport.md", render_report(report, rows).encode("utf-8"))
    return {"report": str(output / "rapport.md"), "aggregate": aggregate}


def render_report(report: dict, rows: list[dict]) -> str:
    lines = [
        "# BM25 avec routage par entreprise",
        "",
        "28 questions dev répondables, références v1. Le routeur reçoit uniquement la question "
        "originale et le catalogue public. L'oracle utilise les annotations. Les traductions "
        "réutilisées incluent des corrections locales : cette expérience ne valide pas une "
        "chaîne autonome de traduction pour des questions nouvelles.",
        "",
        "Statistiques BM25 calculées sur tous les passages sélectionnés : document unique, "
        "union de documents ou corpus complet. Aucun filtrage par année ni validation du "
        "périmètre financier. Une sélection de Crédit Agricole S.A. ne vaut pas preuve Groupe.",
        "",
        "| Variante | " + " | ".join(METRICS) + " |",
        "| --- | " + " | ".join(["---"] * len(METRICS)) + " |",
    ]
    for name, metrics in report["aggregate"].items():
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(
                f"{metrics[key]['numerator']}/{metrics[key]['denominator']}" for key in METRICS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Document attendu présent : {report['routing_expected_document_present']}/28.",
            "",
            "| ID | Route | Documents | Identité oracle A/B | Gains/régressions B vs A |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    by_id = {row["question_id"]: row for row in report["comparisons"]}
    for row in rows:
        a, b = row["original"], row["cached_translation"]
        changes = by_id[row["question_id"]]["cached_vs_original"]
        gains = [key for key, change in changes.items() if change["gain"]]
        losses = [key for key, change in changes.items() if change["regression"]]
        lines.append(
            f"| {row['question_id']} | {a['route']['status']} | "
            f"{len(a['route']['document_ids'])} | "
            f"{a['oracle_top50_and_metrics_identical']}/"
            f"{b['oracle_top50_and_metrics_identical']} | "
            f"gains {', '.join(gains) or 'aucun'} ; régressions {', '.join(losses) or 'aucune'} |"
        )
    lines.extend(
        [
            "",
            "Les décisions, termes déclencheurs, requêtes effectivement utilisées, "
            "rangs complets des références et écarts question par question à chacun des "
            "oracles sont conservés dans predictions_top50.jsonl et comparison.json. "
            "Les listes sauvées sont limitées au top 50 ; les métriques et rangs sont "
            "calculés sur tous les candidats à score positif.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev-only BM25 company routing comparison")
    parser.add_argument(
        "--catalogue", type=Path, default=Path("experiments/company_routing/corpus_catalog.json")
    )
    parser.add_argument("--translation-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(Path.cwd(), args.catalogue, args.translation_set, args.output)))
    except (FinragError, OSError) as error:
        print(f"Company routing experiment failed: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
