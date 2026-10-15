import argparse
import json
from pathlib import Path
from statistics import median

from finrag.config import FinragError, private_output
from finrag.domain.evaluation import EvaluationQuestion
from finrag.evaluation.candidate_reranking_experiment import project_candidates, sealed_inputs
from finrag.evaluation.document_oracle_metrics import evaluate_candidates
from finrag.evaluation.local_reranker_experiment import local_key
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, read_json, read_jsonl, write_atomic
from finrag.retrieval.candidate_reranking import input_identity
from finrag.retrieval.company_router import CatalogueDocument
from finrag.retrieval.local_reranker import validate_local_record

DEPTHS = (1, 5, 10, 20, 50)
CONTEXT_DEPTHS = (5, 10, 20)
BUDGET = 20_000
SEPARATOR = "\n\n"
METADATA_FIELDS = ("document_id", "source_filename", "physical_page_number", "chunk_id")


def depth_metrics(measured: dict, depth: int) -> dict:
    if depth < 0 or depth > measured["candidate_depth"]:
        raise FinragError("Requested depth is outside the saved ranking.")
    pages = list(measured["reference_page_ranks"].values())
    excerpts = measured["reference_excerpt_ranks"]
    if not pages or not excerpts:
        raise FinragError("Depth analysis requires reference pages and excerpts.")
    page_count = sum(rank is not None and rank <= depth for rank in pages)
    excerpt_count = sum(rank is not None and rank <= depth for rank in excerpts)
    return {
        "hit": page_count > 0,
        "complete_pages": page_count == len(pages),
        "complete_excerpts": excerpt_count == len(excerpts),
        "individual_excerpts_found": excerpt_count,
        "individual_excerpts_total": len(excerpts),
    }


def render_passage(hit: dict) -> str:
    metadata = {field: hit[field] for field in METADATA_FIELDS}
    header = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return header + "\n" + hit["text"]


def passage_volume(hits: list[dict]) -> dict:
    raw = sum(len(hit["text"]) for hit in hits)
    units = [render_passage(hit) for hit in hits]
    separators = max(0, len(hits) - 1) * len(SEPARATOR)
    return {
        "passages": len(hits),
        "raw_text_characters": raw,
        "metadata_and_header_newlines_characters": sum(map(len, units)) - raw,
        "separator_characters": separators,
        "serialized_characters": sum(map(len, units)) + separators,
        "distinct_physical_pages": len(
            {(h["document_id"], h["physical_page_number"]) for h in hits}
        ),
    }


def bounded_context(hits: list[dict], budget: int = BUDGET) -> dict:
    if budget < 0:
        raise FinragError("Character budget cannot be negative.")
    units, kept, used = [], [], 0
    for hit in hits:
        unit = render_passage(hit)
        needed = len(unit) + (len(SEPARATOR) if units else 0)
        if used + needed > budget:
            break
        units.append(unit)
        kept.append(hit)
        used += needed
    omitted = [
        {
            "chunk_id": h["chunk_id"],
            "document_id": h["document_id"],
            "physical_page_number": h["physical_page_number"],
            "bge_rank": h["rank"],
            "serialized_unit_characters": len(render_passage(h)),
            "reason": "next_whole_passage_exceeds_budget" if i == 0 else "after_budget_stop",
        }
        for i, h in enumerate(hits[len(kept) :])
    ]
    context = SEPARATOR.join(units)
    assert len(context) == used <= budget
    return {
        "context": context,
        "kept_passages": len(kept),
        "kept_chunk_ids": [h["chunk_id"] for h in kept],
        "omitted_passages": omitted,
        "characters": used,
        "unused_budget": budget - used,
        "distinct_physical_pages": len(
            {(h["document_id"], h["physical_page_number"]) for h in kept}
        ),
        "first_passage_exceeds_budget": bool(hits) and not kept,
    }


def aggregate(rows: list[dict], include_hit: bool = True) -> dict:
    metrics = (
        ("hit", "complete_pages", "complete_excerpts")
        if include_hit
        else ("complete_pages", "complete_excerpts")
    )
    return {
        name: {"numerator": sum(r[name] for r in rows), "denominator": len(rows)}
        for name in metrics
    } | {
        "individual_excerpt_recall": {
            "numerator": sum(r["individual_excerpts_found"] for r in rows),
            "denominator": sum(r["individual_excerpts_total"] for r in rows),
        }
    }


def run(output: Path) -> dict:
    output = private_output(output, Path.cwd())
    if output.exists():
        raise FinragError("Depth/context output exists; overwrite refused.")
    experiment = Path("artifacts/private/bm25_candidate_reranking_dev")
    bge = Path("artifacts/private/bge_reranker_dev")
    routing = Path("artifacts/private/bm25_company_routing_dev/run_1")
    references = (
        Path.cwd() / "artifacts/private/bm25_query_translation_dev/evaluation/references.jsonl"
    )
    catalogue_path = Path("experiments/company_routing/corpus_catalog.json")
    previous_validation = read_json(bge / "diagnostics_run_1/validation.json")
    for path, digest in previous_validation["input_hashes"].items():
        if hash_file(Path(path)) != digest:
            raise FinragError(f"Earlier diagnostic input changed: {path}.")
    paths = [
        p
        for folder in (bge / "cache_cpu", bge / "diagnostics_run_1", experiment, routing)
        for p in folder.rglob("*")
        if p.is_file()
    ]
    paths += [
        bge / "evaluation_cpu.json",
        bge / "run_cpu.json",
        references,
        catalogue_path,
        Path(__file__),
        Path("src/finrag/evaluation/document_oracle_metrics.py"),
    ]
    hashes = {str(p): hash_file(p) for p in paths}
    _, inputs = sealed_inputs(experiment)
    sources = {input_identity(source): source for source in inputs}
    baseline = read_jsonl(experiment / "evaluation/baseline.jsonl")
    routed = {
        r["question_id"]: r["cached_translation"]
        for r in read_jsonl(routing / "predictions_top50.jsonl")
    }
    catalogue = [CatalogueDocument.model_validate(r) for r in read_json(catalogue_path)]
    if hash_file(references) != read_json(routing / "comparison.json")["input_hashes"].get(
        str(references)
    ):
        raise FinragError("Sealed dev v1 reference hash differs.")
    questions = [EvaluationQuestion.model_validate(r) for r in read_jsonl(references)]
    if len(questions) != 28 or any(q.split != "dev" or not q.answerable for q in questions):
        raise FinragError("Depth/context analysis requires 28 answerable dev questions.")
    by_question = {q.question_id: q for q in questions}
    if len(by_question) != 28 or set(by_question) != set(routed):
        raise FinragError("Reference and candidate question IDs differ.")
    saved = read_json(bge / "evaluation_cpu.json")
    config = read_json(next((bge / "cache_cpu").glob("*.json")))["configuration"]
    rows = []
    for base in baseline:
        q = by_question[base["question_id"]]
        source = sources[base["input_id"]]
        route = routed[q.question_id]
        projected, mapping = project_candidates(route["query_used"], route["hits"], catalogue)
        if (
            input_identity(projected) != input_identity(source)
            or mapping != base["candidate_id_to_chunk_id"]
            or route["hits"] != base["hits"]
            or len(source.candidates) != 50
        ):
            raise FinragError("Frozen cached_translation candidates or effective query differ.")
        record = read_json(bge / "cache_cpu" / f"{local_key(source, config)}.json")
        if record["status"] != "success":
            raise FinragError("Expected 28 successful saved BGE outputs.")
        ranking = validate_local_record(source, record, config)
        chunks = {h["chunk_id"]: h for h in base["hits"]}
        hits = [{**chunks[mapping[v["candidate_id"]]], "rank": v["rank"]} for v in ranking]
        measured = {
            "bm25": evaluate_candidates(q, base["hits"], False),
            "bge": evaluate_candidates(q, hits, False),
        }
        saved_row = next(r for r in saved["rows"] if r["question_id"] == q.question_id)
        for label, key in (("bm25", "baseline"), ("bge", "metrics")):
            if any(measured[label][m] != value for m, value in saved_row[key].items()):
                raise FinragError("Saved known metrics were not reproduced.")
        metrics = {
            label: {str(k): depth_metrics(value, k) for k in DEPTHS}
            for label, value in measured.items()
        }
        contexts, volumes = {}, {}
        for k in CONTEXT_DEPTHS:
            selected = hits[:k]
            volumes[str(k)] = passage_volume(selected)
            context = bounded_context(selected)
            context["coverage"] = {
                name: value
                for name, value in depth_metrics(measured["bge"], context["kept_passages"]).items()
                if name != "hit"
            }
            context["lost_reference_excerpt_indices"] = [
                i
                for i, rank in enumerate(measured["bge"]["reference_excerpt_ranks"], 1)
                if rank is not None and context["kept_passages"] < rank <= k
            ]
            # Evidence ranks come from raw passages, never from metadata or joins.
            contexts[str(k)] = context
        rows.append(
            {
                "question_id": q.question_id,
                "input_id": base["input_id"],
                "query_used": source.question,
                "reference_physical_pages": q.relevant_pages,
                "reference_page_ranks": {n: v["reference_page_ranks"] for n, v in measured.items()},
                "reference_excerpt_ranks": {
                    n: v["reference_excerpt_ranks"] for n, v in measured.items()
                },
                "depth_metrics": metrics,
                "volumes": volumes,
                "contexts": contexts,
            }
        )
    if len(rows) != 28 or len({r["question_id"] for r in rows}) != 28:
        raise FinragError("Expected exactly the same 28 questions.")
    depth_aggregate = {
        label: {
            str(k): aggregate([r["depth_metrics"][label][str(k)] for r in rows]) for k in DEPTHS
        }
        for label in ("bm25", "bge")
    }
    context_aggregate = {
        str(k): aggregate([r["contexts"][str(k)]["coverage"] for r in rows], include_hit=False)
        for k in CONTEXT_DEPTHS
    }
    assert depth_aggregate["bm25"]["5"]["complete_excerpts"]["numerator"] == 11
    assert depth_aggregate["bm25"]["50"]["complete_excerpts"]["numerator"] == 20
    assert depth_aggregate["bge"]["5"]["complete_excerpts"]["numerator"] == 16
    volumes = {
        str(k): {
            field: {
                "median": median(r["volumes"][str(k)][field] for r in rows),
                "maximum": max(r["volumes"][str(k)][field] for r in rows),
            }
            for field in ("raw_text_characters", "serialized_characters", "distinct_physical_pages")
        }
        for k in CONTEXT_DEPTHS
    }
    transitions = {
        f"{a}_to_{b}": [
            {
                "question_id": r["question_id"],
                "before": r["depth_metrics"]["bge"][str(a)],
                "after": r["depth_metrics"]["bge"][str(b)],
            }
            for r in rows
            if r["depth_metrics"]["bge"][str(a)] != r["depth_metrics"]["bge"][str(b)]
        ]
        for a, b in ((5, 10), (10, 20))
    }
    report = {
        "split": "dev",
        "reference_version": "v1",
        "questions": 28,
        "successful_bge_outputs": 28,
        "protocol": {
            "character_budget": BUDGET,
            "metadata_fields": METADATA_FIELDS,
            "metadata_format": "compact JSON, ensure_ascii=False, sort_keys=True, then one LF",
            "passage_separator": SEPARATOR,
            "query_outside_context_budget": True,
            "character_measure": "Python len of Unicode strings, not bytes or tokens",
            "stop_at_first_overflow": True,
            "overlap_preserved": True,
            "existing_builder_not_used": "Does not budget metadata and requires extracted pages.",
        },
        "depth_aggregate": depth_aggregate,
        "context_aggregate": context_aggregate,
        "volume_summary": volumes,
        "bounded_volume_summary": {
            str(k): {
                field: {
                    "median": median(r["contexts"][str(k)][field] for r in rows),
                    "maximum": max(r["contexts"][str(k)][field] for r in rows),
                }
                for field in ("characters", "distinct_physical_pages", "kept_passages")
            }
            for k in CONTEXT_DEPTHS
        },
        "transitions": transitions,
        "rows": rows,
        "bge_at20_complete_excerpts": depth_aggregate["bge"]["20"]["complete_excerpts"],
        "pool_incomplete_question_ids": [
            r["question_id"]
            for r in rows
            if not r["depth_metrics"]["bge"]["50"]["complete_excerpts"]
        ],
    }
    write_atomic(output / "comparison.json", json_bytes(report))
    write_atomic(output / "rapport.md", render_report(report).encode("utf-8"))
    if any(hash_file(Path(path)) != digest for path, digest in hashes.items()):
        raise FinragError("An input changed during offline depth/context analysis.")
    write_atomic(
        output / "validation.json",
        json_bytes(
            {
                "input_hashes": hashes,
                "reference_sha256": hash_file(references),
                "checks": {
                    "previous_diagnostic_hashes_valid": True,
                    "cached_translation_candidates_queries_exact": True,
                    "all_28_successful_cache_permutations_and_finite_scores_verified": True,
                    "known_saved_metrics_reproduced": True,
                    "at20_observed_not_forced": report["bge_at20_complete_excerpts"],
                    "metadata_separator_and_text_budget_invariants": True,
                    "no_new_inference_search_api_or_test_access": True,
                },
                "command": ".\\.venv\\Scripts\\python.exe -m finrag.evaluation.bge_depth_context "
                "--output artifacts/private/bge_reranker_dev/depth_context_run_1",
                "test_command": ".\\.venv\\Scripts\\python.exe -m pytest "
                "tests/test_bge_depth_context.py -q -p no:cacheprovider "
                "--basetemp artifacts/private/dc1",
                "output_hashes": {
                    p.name: hash_file(p)
                    for p in (output / "comparison.json", output / "rapport.md")
                },
            }
        ),
    )
    return {
        "output": str(output),
        "depth_aggregate": depth_aggregate,
        "context_aggregate": context_aggregate,
        "volume_summary": volumes,
    }


def render_report(report: dict) -> str:
    lines = [
        "# Profondeur BGE et budget de contexte — dev v1",
        "",
        "28 sorties réussies ; mêmes candidats cached_translation et requêtes scellées. "
        "Les évaluateurs existants fournissent les rangs des pages et extraits ; @20 "
        "est dérivé de ces rangs sans hypothèse de score attendu.",
        "",
        "| Classement | Profondeur | Hit | Pages complètes | "
        "Extraits complets | Extraits individuels |",
        "|---|---|---|---|---|---|",
    ]
    for label, depths in report["depth_aggregate"].items():
        for k, values in depths.items():
            cells = [
                f"{values[m]['numerator']}/{values[m]['denominator']}"
                for m in ("hit", "complete_pages", "complete_excerpts", "individual_excerpt_recall")
            ]
            lines.append(f"| {label} | {k} | {' | '.join(cells)} |")
    lines += [
        "",
        "Le rappel individuel est micro : extraits retrouvés / extraits annotés, "
        "avec document et page physique v1 obligatoires. Une page présente peut manquer "
        "l'extrait. Aucune occurrence dans des métadonnées ou à travers deux passages "
        "n'est comptée comme preuve.",
        "",
        "## Volumes BGE non plafonnés",
        "",
        "| Top | Texte médiane/max | Avec métadonnées et séparateurs médiane/max | "
        "Pages distinctes médiane/max |",
        "|---|---|---|---|",
    ]
    for k, fields in report["volume_summary"].items():
        cells = [
            f"{fields[f]['median']:g} / {fields[f]['maximum']:g}"
            for f in ("raw_text_characters", "serialized_characters", "distinct_physical_pages")
        ]
        lines.append(f"| {k} | {' | '.join(cells)} |")
    lines += [
        "",
        "Le constructeur précédent compte uniquement textes/séparateurs et exige "
        "les extractions : il n'est pas réutilisé ici. Protocole analytique fixé : "
        "en-tête JSON compact {document_id, source_filename, physical_page_number, chunk_id}, "
        "un LF avant le texte intact, deux LF entre passages. Aucun rang, score ni "
        "annotation dans l'en-tête. La question reste hors du contexte mesuré. "
        "Les caractères Unicode sont comptés par len, jamais convertis en tokens.",
        "",
        "## Contextes plafonnés à 20 000 caractères",
        "",
        "Passages entiers, ordre BGE, arrêt au premier dépassement ; aucune expansion, "
        "déduplication de page ou suppression des chevauchements.",
        "",
        "| Top demandé | Preuves complètes sans budget | Preuves complètes du contexte | "
        "Passages conservés médiane/min/max | Contextes arrêtés |",
        "|---|---|---|---|---|",
    ]
    for k in CONTEXT_DEPTHS:
        key = str(k)
        contexts = [r["contexts"][key] for r in report["rows"]]
        counts = [c["kept_passages"] for c in contexts]
        plain = report["depth_aggregate"]["bge"][key]["complete_excerpts"]["numerator"]
        kept = report["context_aggregate"][key]["complete_excerpts"]["numerator"]
        lines.append(
            f"| {k} | {plain}/28 | {kept}/28 | {median(counts):g}/{min(counts)}/{max(counts)} | "
            f"{sum(bool(c['omitted_passages']) for c in contexts)}/28 |"
        )
    lines += [
        "",
        "Ces couvertures de contexte ne sont pas des Hit@k. Les textes construits, "
        "caractères, unités omises, raisons d'arrêt, pages distinctes et indices "
        "d'extraits perdus sont conservés pour chaque question dans comparison.json.",
        "Les médianes et maxima des contextes plafonnés figurent aussi dans "
        "bounded_volume_summary. Le rappel individuel reste défini sur les 32 extraits v1.",
        "",
        "## Changements par question",
        "",
    ]
    for transition, changed in report["transitions"].items():
        lines.append(f"- {transition} :")
        for row in changed:
            a, b = row["before"], row["after"]
            lines.append(
                f"  - {row['question_id']} : "
                f"pages {int(a['complete_pages'])}→{int(b['complete_pages'])}, "
                f"extraits complets {int(a['complete_excerpts'])}→{int(b['complete_excerpts'])}, "
                f"extraits {a['individual_excerpts_found']}→{b['individual_excerpts_found']}. "
            )
    for k in CONTEXT_DEPTHS:
        lost = [
            r["question_id"]
            for r in report["rows"]
            if r["depth_metrics"]["bge"][str(k)]["complete_excerpts"]
            and not r["contexts"][str(k)]["coverage"]["complete_excerpts"]
        ]
        lines.append(
            f"- Top {k}, couverture complète perdue au budget : {', '.join(lost) or 'aucune'}."
        )
    lines += [
        "",
        "Pools v1 incomplets même à 50 : "
        + ", ".join(report["pool_incomplete_question_ids"])
        + ".",
        "",
        "Augmenter la profondeur peut remonter les preuves déjà présentes ; le budget "
        "peut les retirer du contexte. Les résultats dépendent du format d'en-tête "
        "explicitement fixé, qui ajoute un coût mesuré. Une couverture textuelle exacte "
        "ne certifie pas la suffisance financière et ne valide aucune annotation candidate.",
        "",
        "Une seule prochaine action proposée : comparer hors ligne cet en-tête à "
        "un format de provenance plus compact (document/page dans le contexte, "
        "identités complètes conservées dans un index associé), au même budget de "
        "20 000 caractères. Les textes bruts du top 10 tiennent tous sous 20 000 ; "
        "l'en-tête actuel pousse 24 contextes au-delà de la limite. Cette ablation "
        "isolerait le coût des métadonnées, sans changer les candidats ni références. "
        "Elle ne résout pas les huit pools incomplets ni, nécessairement, la preuve "
        "au rang 18. Action non implémentée ; aucune profondeur du moteur n'est changée.",
        "",
        "Aucune nouvelle recherche, inférence, installation, API, lecture du test, "
        "modification des historiques, commit ou push.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline BGE depth and framed-context analysis")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output)))


if __name__ == "__main__":
    main()
