import argparse
import json
import sys
from pathlib import Path

from finrag.config import FinragError, private_output
from finrag.domain.evaluation import EvaluationQuestion
from finrag.evaluation.document_oracle_metrics import aggregate_metrics, evaluate_candidates
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes, read_json, write_atomic
from finrag.retrieval.saved_bm25_fusion import fuse_saved_bm25

METRICS = ("hit@5", "complete_excerpt_coverage@5", "complete_excerpt_coverage@50")


def evidence_present(question: EvaluationQuestion, hits: list[dict]) -> list[bool]:
    if question.split != "dev" or not question.answerable or not question.evidence:
        raise FinragError("Fusion evaluation requires answerable dev questions with evidence.")
    return [
        any(
            hit["document_id"] == question.document_id
            and hit["physical_page_number"] == evidence.physical_page_number
            and evidence.text in hit["text"]
            for hit in hits
        )
        for evidence in question.evidence
    ]


def evidence_diagnosis(question: EvaluationQuestion, union: list[dict], rrf: list[dict]) -> dict:
    union_found = evidence_present(question, union)
    retained = evidence_present(question, rrf)
    top5 = evidence_present(question, rrf[:5])
    statuses = [
        "absent_from_both"
        if not available
        else "eliminated_from_rrf50"
        if not kept
        else "retained_below_top5"
        if not early
        else "in_rrf_top5"
        for available, kept, early in zip(union_found, retained, top5, strict=True)
    ]
    category = (
        "incomplete_union"
        if not all(union_found)
        else "complete_union_lost_in_rrf50"
        if not all(retained)
        else "complete_rrf50_below_top5"
        if not all(top5)
        else "complete_rrf_top5"
    )
    return {"per_reference_excerpt": statuses, "question_category": category}


def run(project: Path, source: Path, output: Path) -> dict:
    output = private_output(output, project)
    if output.exists():
        raise FinragError("Fusion output already exists; overwrite refused.")
    comparison_path = source / "comparison.json"
    predictions_path = source / "predictions_top50.jsonl"
    comparison = read_json(comparison_path)
    if comparison["split"] != "dev" or comparison["reference_version"] != "v1":
        raise FinragError("Only saved dev v1 results can be fused.")
    references = (
        project / "artifacts/private/bm25_query_translation_dev/evaluation/references.jsonl"
    )
    expected_hash = comparison["input_hashes"].get(str(references))
    if expected_hash is None or hash_file(references) != expected_hash:
        raise FinragError("Sealed dev references do not match routing run_1.")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    questions = [
        EvaluationQuestion.model_validate(json.loads(line))
        for line in references.read_text(encoding="utf-8").splitlines()
    ]
    if len(questions) != 28 or any(q.split != "dev" or not q.answerable for q in questions):
        raise FinragError("Expected exactly 28 answerable dev references.")
    by_id = {q.question_id: q for q in questions}
    if len(by_id) != 28 or len(rows) != 28 or {r["question_id"] for r in rows} != set(by_id):
        raise FinragError("Saved variants must contain exactly the same 28 dev question IDs.")
    before = {
        str(path): hash_file(path) for path in [*source.iterdir(), references] if path.is_file()
    }
    evaluated, predictions = [], []
    for row in rows:
        question = by_id[row["question_id"]]
        original, translated = row["original"], row["cached_translation"]
        fusion = fuse_saved_bm25(
            original["query_used"],
            original["hits"],
            translated["query_used"],
            translated["hits"],
        )
        metrics = {
            name: evaluate_candidates(question, hits, False)
            for name, hits in (
                ("original", original["hits"]),
                ("cached_translation", translated["hits"]),
                ("rrf50", fusion.ranked),
            )
        }
        for name in ("original", "cached_translation"):
            if any(metrics[name][key] != row[name]["metrics"][key] for key in METRICS):
                raise FinragError(f"Saved baseline metrics differ for {question.question_id}.")
        coverage = evidence_present(question, fusion.union)
        changes = {
            baseline: {
                key: {
                    "gain": not metrics[baseline][key] and metrics["rrf50"][key],
                    "regression": metrics[baseline][key] and not metrics["rrf50"][key],
                }
                for key in METRICS
            }
            for baseline in ("original", "cached_translation")
        }
        evaluated.append(
            {
                "question_id": question.question_id,
                "original_depth": len(original["hits"]),
                "cached_translation_depth": len(translated["hits"]),
                "independent_lists": fusion.independent_lists,
                "union_candidate_count": len(fusion.union),
                "union_complete_excerpt_coverage": all(coverage),
                "union_excerpt_presence": coverage,
                "union_recovers_vs_cached_translation": all(coverage)
                and not metrics["cached_translation"]["complete_excerpt_coverage@50"],
                "metrics": metrics,
                "changes": changes,
                "diagnosis": evidence_diagnosis(question, fusion.union, fusion.ranked),
            }
        )
        predictions.append(
            {
                "question_id": question.question_id,
                "union_order_is_not_relevance_ranking": True,
                "union": fusion.union,
                "rrf50": fusion.ranked,
            }
        )
    aggregates = {
        name: aggregate_metrics([r["metrics"][name] for r in evaluated])
        for name in ("original", "cached_translation", "rrf50")
    }
    for name in ("original", "cached_translation"):
        if any(aggregates[name][k] != comparison["aggregate"][name][k] for k in METRICS):
            raise FinragError("Historical aggregate metrics were not reproduced.")
    report = {
        "split": "dev",
        "reference_version": "v1",
        "questions": 28,
        "rrf_constant": 60,
        "weights": [1, 1],
        "tie_break": "chunk_id",
        "reference_source": "sealed answerable-dev copy hashed by routing run_1",
        "input_hashes": before,
        "aggregate": aggregates,
        "union_complete_excerpt_coverage": {
            "numerator": sum(r["union_complete_excerpt_coverage"] for r in evaluated),
            "denominator": 28,
        },
        "union_recovers_vs_cached_translation": [
            r["question_id"] for r in evaluated if r["union_recovers_vs_cached_translation"]
        ],
        "observations": evaluated,
    }
    write_atomic(output / "predictions.jsonl", jsonl_bytes(predictions))
    write_atomic(output / "comparison.json", json_bytes(report))
    write_atomic(output / "rapport.md", render_report(report).encode("utf-8"))
    if any(hash_file(Path(path)) != digest for path, digest in before.items()):
        raise FinragError("An input changed during offline fusion.")
    return {
        "output": str(output),
        "aggregate": aggregates,
        "union_complete_excerpt_coverage": report["union_complete_excerpt_coverage"],
        "union_recovers_vs_cached_translation": report["union_recovers_vs_cached_translation"],
    }


def render_report(report: dict) -> str:
    lines = [
        "# Fusion hors ligne des classements BM25",
        "",
        "Entrées : run_1/predictions_top50.jsonl, 28 lignes, variantes `original` et "
        "`cached_translation` (traductions corrigées de evaluation_run_2). Les deux listes "
        "contiennent chacune 50 segments pour chaque question. Aucun score lexical ne "
        "participe à la fusion. RRF = somme de 1/(60+rang), rang à partir de 1, poids égaux, "
        "départage par chunk_id. Les requêtes exactement identiques comptent une seule fois.",
        "",
        "Références : copie privée dev v1 scellée, SHA-256 vérifié contre run_1. "
        "Aucune lecture du split test. Fusion et union ne reçoivent aucune annotation. "
        "Toutes les occurrences partagées ont le même texte et la même provenance.",
        "",
        "| Classement | Hit@5 | Extraits complets @5 | Extraits complets @50 |",
        "|---|---|---|---|",
    ]
    for name, metrics in report["aggregate"].items():
        cells = [f"{metrics[k]['numerator']}/{metrics[k]['denominator']}" for k in METRICS]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    union = report["union_complete_excerpt_coverage"]
    recovered = report["union_recovers_vs_cached_translation"]
    counts = [r["union_candidate_count"] for r in report["observations"]]
    lines += [
        "",
        f"Union complète : {union['numerator']}/28 preuves complètes ; "
        f"{min(counts)} à {max(counts)} candidats par question. "
        "Cette couverture ne constitue pas un Recall@50. L'ordre de stockage de "
        "l'union est l'ordre des IDs, sans sens de pertinence.",
        "",
        "Questions récupérées par l'union contre cached_translation @50 : "
        f"{', '.join(recovered) or 'aucune'}.",
        "",
        "| Question | Union | Listes indépendantes | Original H5/E5/E50 | "
        "Traduction H5/E5/E50 | RRF H5/E5/E50 | Diagnostic |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["observations"]:
        cells = [
            "/".join(str(int(row["metrics"][name][k])) for k in METRICS)
            for name in ("original", "cached_translation", "rrf50")
        ]
        lines.append(
            f"| {row['question_id']} | {row['union_candidate_count']} | "
            f"{row['independent_lists']} | {' | '.join(cells)} | "
            f"{row['diagnosis']['question_category']} |"
        )
    lines += ["", "## Gains et régressions RRF", ""]
    for baseline in ("original", "cached_translation"):
        for key in METRICS:
            gains = [
                r["question_id"]
                for r in report["observations"]
                if r["changes"][baseline][key]["gain"]
            ]
            losses = [
                r["question_id"]
                for r in report["observations"]
                if r["changes"][baseline][key]["regression"]
            ]
            lines.append(
                f"- Contre {baseline}, {key} : gains {', '.join(gains) or 'aucun'} ; "
                f"régressions {', '.join(losses) or 'aucune'}."
            )
    lines += [
        "",
        "## Limites",
        "",
        "Le diagnostic des preuves est détaillé par extrait dans comparison.json : "
        "absent des deux listes, éliminé du top 50 RRF, conservé après le rang 5, "
        "ou dans le top 5. Une question multi-preuves peut cumuler plusieurs causes ; "
        "sa catégorie résumée donne priorité à l'incomplétude de l'union, puis à la "
        "perte lors de la limitation à 50, puis au classement après le rang 5.",
        "",
        "Les extraits sont vérifiés par inclusion textuelle exacte sur leur document "
        "et page physique v1, sans référence alternative. Hit@5 mesure une page attendue, "
        "pas nécessairement le passage portant la preuve. Les résultats portent sur "
        "des listes déjà tronquées à 50, jamais sur toute la recherche documentaire. "
        "Les traductions mises en cache comprennent des corrections locales ; ceci "
        "ne valide pas une chaîne autonome pour de nouvelles questions.",
        "",
        "Aucune recherche, API, inférence, encoding, modification des références ou "
        "reclassement par modèle. Les empreintes des trois fichiers run_1 et des "
        "références ont été contrôlées avant et après publication.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline fusion of saved dev BM25 rankings")
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/private/bm25_company_routing_dev/run_1")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(Path.cwd(), args.source, args.output)))
    except (FinragError, OSError, KeyError, ValueError) as error:
        print(f"Saved BM25 fusion failed: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
