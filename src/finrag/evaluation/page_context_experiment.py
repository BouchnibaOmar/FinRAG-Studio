import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from finrag.config import FinragError, filesystem_operation, private_output
from finrag.domain.documents import PageRecord
from finrag.domain.evaluation import EvaluationQuestion
from finrag.evaluation.document_oracle_metrics import evaluate_candidates
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import (
    json_bytes,
    jsonl_bytes,
    read_json,
    read_jsonl,
    validate_page_sequence,
    write_atomic,
)
from finrag.retrieval.page_context import (
    CONTEXT_CHARACTER_LIMIT,
    ConstructedContext,
    ContextPassage,
    build_context,
)


def load_extracted_pages(root: Path, passages: list[ContextPassage]) -> tuple[dict, list[Path]]:
    identities = {p.source_filename: p for p in passages}
    pages, files = {}, []
    for filename, identity in identities.items():
        directory = root / Path(filename).stem
        cache_path = directory / "extraction_cache.json"
        with filesystem_operation("read existing extraction cache", cache_path):
            cache = read_json(cache_path)
        page_path, manifest_path = directory / "pages.jsonl", directory / "document_manifest.json"
        with filesystem_operation("validate existing extracted pages", page_path):
            if cache["pages_sha256"] != hash_file(page_path):
                raise FinragError("Extracted page hash differs from extraction cache.")
            records = [PageRecord.model_validate(row) for row in read_jsonl(page_path)]
        with filesystem_operation("validate extraction manifest", manifest_path):
            if cache["manifest_sha256"] != hash_file(manifest_path):
                raise FinragError("Extraction manifest hash differs from extraction cache.")
            manifest = read_json(manifest_path)
        validate_page_sequence(records)
        if (
            cache["source_sha256"] != identity.source_sha256
            or manifest["document_id"] != identity.document_id
            or manifest["physical_page_count"] != len(records)
            or any(
                p.document_id != identity.document_id
                or p.source_filename != filename
                or p.source_sha256 != identity.source_sha256
                for p in records
            )
        ):
            raise FinragError("Extraction identity differs from saved candidates.")
        for page in records:
            key = (page.document_id, page.physical_page_number)
            if key in pages:
                raise FinragError("Duplicate physical page in extraction.")
            pages[key] = page
        files.extend([cache_path, page_path, manifest_path])
    return pages, files


def context_coverage(question: EvaluationQuestion, context: ConstructedContext) -> dict:
    if question.split != "dev" or not question.answerable or not question.evidence:
        raise FinragError("Context evaluation requires answerable dev references.")
    found = [
        any(
            unit["document_id"] == question.document_id
            and unit["physical_page_number"] == evidence.physical_page_number
            and evidence.text in unit["text"]
            for unit in context.units
        )
        for evidence in question.evidence
    ]
    return {"excerpt_presence": found, "complete_excerpt_coverage": all(found)}


def context_summary(context: ConstructedContext) -> dict:
    return {
        "characters": len(context.text),
        "unit_count": len(context.units),
        "pages": list(
            dict.fromkeys((u["document_id"], u["physical_page_number"]) for u in context.units)
        ),
        "represented_chunk_ids": context.represented_chunk_ids,
        "represented_passages": len(context.represented_chunk_ids),
        "omitted_units": context.omitted_units,
        "stop_reason": context.stop_reason,
        "first_unit_exceeds_limit": context.first_unit_exceeds_limit,
    }


def run(project: Path, source: Path, extractions: Path, output: Path) -> dict:
    output = private_output(output, project)
    if output.exists():
        raise FinragError("Page-context output exists; overwrite refused.")
    rows = read_jsonl(source / "predictions_top50.jsonl")
    baseline = read_json(source / "comparison.json")
    if baseline["split"] != "dev" or baseline["reference_version"] != "v1" or len(rows) != 28:
        raise FinragError("Expected 28 saved dev v1 questions.")
    candidates = [
        [
            ContextPassage.model_validate(
                {
                    key: hit[key]
                    for key in (
                        "document_id",
                        "source_filename",
                        "source_sha256",
                        "chunk_id",
                        "physical_page_number",
                        "text",
                    )
                }
            )
            for hit in row["cached_translation"]["hits"]
        ]
        for row in rows
    ]
    if any(len(group) != 50 for group in candidates):
        raise FinragError("Expected cached_translation top 50 for each question.")
    pages, page_files = load_extracted_pages(
        extractions, [p for group in candidates for p in group]
    )
    reference_path = (
        project / "artifacts/private/bm25_query_translation_dev/evaluation/references.jsonl"
    )
    if hash_file(reference_path) != baseline["input_hashes"].get(str(reference_path)):
        raise FinragError("Sealed dev v1 reference hash differs from routing run_1.")
    questions = [EvaluationQuestion.model_validate(q) for q in read_jsonl(reference_path)]
    by_id = {q.question_id: q for q in questions}
    if (
        len(by_id) != 28
        or len(questions) != 28
        or any(q.split != "dev" or not q.answerable for q in questions)
        or len({r["question_id"] for r in rows}) != 28
        or {r["question_id"] for r in rows} != set(by_id)
    ):
        raise FinragError("Only the same 28 answerable dev questions may be evaluated.")
    inputs = [*page_files, reference_path, *[p for p in source.iterdir() if p.is_file()]]
    hashes = {str(p): hash_file(p) for p in inputs}
    settings = {
        "character_limit": CONTEXT_CHARACTER_LIMIT,
        "separator": "\n\n",
        "variant": "cached_translation",
        "seed_passage_limits": [5, 50],
        "stop_at_first_exceeding_unit": True,
        "partial_units_allowed": False,
        "neighbor_pages": False,
        "count_separator_characters": True,
    }
    write_atomic(output / "settings.json", json_bytes(settings))
    observations, saved_contexts = [], []
    for row, passages in zip(rows, candidates, strict=True):
        question = by_id[row["question_id"]]
        historical = evaluate_candidates(question, row["cached_translation"]["hits"], False)
        for k in (5, 50):
            key = f"complete_excerpt_coverage@{k}"
            if historical[key] != row["cached_translation"]["metrics"][key]:
                raise FinragError("Historical candidate evidence metrics differ.")
        variants, contexts = {}, {}
        for k in (5, 50):
            compared = {}
            for mode in ("passages", "pages"):
                context = build_context(passages[:k], pages, mode)
                name = f"{mode}_from_top{k}"
                coverage = context_coverage(question, context)
                variants[name] = {**context_summary(context), **coverage}
                contexts[name] = asdict(context)
                compared[mode] = coverage["complete_excerpt_coverage"]
            variants[f"changes_top{k}"] = {
                "gain": compared["pages"] and not compared["passages"],
                "loss": compared["passages"] and not compared["pages"],
                "no_gain": not (compared["pages"] and not compared["passages"]),
            }
        observations.append(
            {
                "question_id": question.question_id,
                "variants": variants,
                "historical_complete_excerpts@5": historical["complete_excerpt_coverage@5"],
                "historical_complete_excerpts@50": historical["complete_excerpt_coverage@50"],
            }
        )
        saved_contexts.append({"question_id": question.question_id, "contexts": contexts})
    historical_counts = {
        k: sum(r[f"historical_complete_excerpts@{k}"] for r in observations) for k in (5, 50)
    }
    if historical_counts != {5: 11, 50: 20}:
        raise FinragError("Historical 11/28 and 20/28 evidence coverage was not reproduced.")
    aggregate = {
        name: {
            "numerator": sum(
                r["variants"][name]["complete_excerpt_coverage"] for r in observations
            ),
            "denominator": 28,
        }
        for name in (
            "passages_from_top5",
            "pages_from_top5",
            "passages_from_top50",
            "pages_from_top50",
        )
    }
    report = {
        "settings": settings,
        "input_hashes": hashes,
        "aggregate": aggregate,
        "historical_coverage": historical_counts,
        "observations": observations,
    }
    write_atomic(output / "contexts.jsonl", jsonl_bytes(saved_contexts))
    write_atomic(output / "comparison.json", json_bytes(report))
    write_atomic(output / "rapport.md", render_report(report).encode("utf-8"))
    if any(hash_file(Path(p)) != digest for p, digest in hashes.items()):
        raise FinragError("An input changed during page-context expansion.")
    return {"output": str(output), "aggregate": aggregate}


def render_report(report: dict) -> str:
    lines = [
        "# Expansion du contexte par page",
        "",
        "cached_translation seulement, 28 questions dev v1. Classements historiques "
        "inchangés : preuves complètes @5=11/28, @50=20/28. Aucun RRF.",
        "",
        "Budget fixé avant évaluation : 20 000 caractères Unicode, séparateurs inclus. "
        "Unités entières, arrêt au premier dépassement, sans page voisine. Les pages "
        "proviennent de l'extraction existante validée par empreintes ; les chunks "
        "chevauchants ne sont pas concaténés pour reconstituer une page.",
        "",
        "| Contexte construit | Couverture complète des extraits |",
        "|---|---|",
    ]
    for name, metric in report["aggregate"].items():
        lines.append(f"| {name} | {metric['numerator']}/{metric['denominator']} |")
    lines += [
        "",
        "| Question | Passages 5 | Pages 5 | Passages 50 | Pages 50 |",
        "|---|---|---|---|---|",
    ]
    for row in report["observations"]:
        cells = [
            f"{int(row['variants'][n]['complete_excerpt_coverage'])} "
            f"({row['variants'][n]['characters']} caractères, "
            f"{len(row['variants'][n]['pages'])} pages)"
            for n in report["aggregate"]
        ]
        lines.append(f"| {row['question_id']} | {' | '.join(cells)} |")
    for k in (5, 50):
        for change in ("gain", "loss", "no_gain"):
            ids = [
                r["question_id"]
                for r in report["observations"]
                if r["variants"][f"changes_top{k}"][change]
            ]
            lines.append(f"\nTop {k}, {change} : {', '.join(ids) or 'aucun'}.")
    lines += [
        "",
        "Les unités omises, raisons d'arrêt, premières pages trop longues, "
        "pages physiques, IDs de passages représentés et présence par extrait sont "
        "détaillés dans comparison.json ; le texte exact et la provenance dans "
        "contexts.jsonl. Les passages représentés désignent les candidats du top K "
        "présents dans les unités retenues, pas tous les chunks du corpus.",
        "",
        "Ces mesures évaluent un contexte construit, pas Hit@5 ou Recall@50. "
        "L'inclusion textuelle exacte d'un extrait v1 sur sa page ne valide ni "
        "la suffisance financière, ni la structure d'un tableau, ni une réponse. "
        "Les extraits ne sont pas recherchés à travers des séparateurs entre unités. "
        "Les passages seuls peuvent répéter du texte chevauchant, facturé au budget ; "
        "les pages gardent leur texte d'extraction intact. Les gains éventuels peuvent "
        "aussi venir de la suppression de ce chevauchement.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline expansion from existing extracted pages")
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/private/bm25_company_routing_dev/run_1")
    )
    parser.add_argument("--extractions", type=Path, default=Path("data/private/corpus/extracted"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(Path.cwd(), args.source, args.extractions, args.output)))
    except (FinragError, OSError, ValueError, KeyError) as error:
        print(f"Page context experiment failed: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
