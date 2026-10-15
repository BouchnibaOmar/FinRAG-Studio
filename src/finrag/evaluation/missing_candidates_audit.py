import argparse
import hashlib
import json
from pathlib import Path

from finrag.config import FileOperationError, FinragError, private_output
from finrag.domain.dense import ModelChunk
from finrag.domain.evaluation import EvaluationQuestion
from finrag.evaluation.candidate_reranking_experiment import sealed_inputs
from finrag.evaluation.document_oracle_metrics import evaluate_candidates
from finrag.evaluation.page_context_experiment import load_extracted_pages
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, read_json, read_jsonl, write_atomic
from finrag.retrieval.bm25 import tokenize
from finrag.retrieval.page_context import ContextPassage


def evidence_occurrences(question: EvaluationQuestion, chunks: list[dict]) -> list[dict]:
    results = []
    for evidence in question.evidence:
        matches = [
            c
            for c in chunks
            if c["document_id"] == question.document_id and evidence.text in c["text"]
        ]
        results.append(
            {
                "evidence": evidence.model_dump(mode="json"),
                "on_reference_page": [
                    c for c in matches if c["physical_page_number"] == evidence.physical_page_number
                ],
                "other_pages_unvalidated": [
                    c for c in matches if c["physical_page_number"] != evidence.physical_page_number
                ],
            }
        )
    return results


def vocabulary_difference(query: str, text: str) -> dict:
    query_terms, passage_terms = set(tokenize(query)), set(tokenize(text))
    return {
        "method": "existing BM25 tokenizer; unique terms, no ranking or query expansion",
        "shared": sorted(query_terms & passage_terms),
        "query_only": sorted(query_terms - passage_terms),
        "passage_only": sorted(passage_terms - query_terms),
    }


def page_chunks(question: EvaluationQuestion, chunks: list[dict]) -> list[dict]:
    return sorted(
        [
            c
            for c in chunks
            if c["document_id"] == question.document_id
            and c["physical_page_number"] in question.relevant_pages
        ],
        key=lambda c: (c["physical_page_number"], c["source_character_start"], c["chunk_id"]),
    )


def run(output: Path) -> dict:
    output = private_output(output, Path.cwd())
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FinragError("Missing-candidates audit output exists; overwrite refused.")
    bge = Path("artifacts/private/bge_reranker_dev")
    paths, hashes = set(), {}

    def track(path: Path) -> None:
        hashes[str(path)] = hash_file(path)
        paths.add(path)

    for folder in ("diagnostics_run_1", "depth_context_run_1"):
        validation_path = bge / folder / "validation.json"
        validation = read_json(validation_path)
        for name, digest in validation["input_hashes"].items():
            path = Path(name)
            track(path)
            if hashes[str(path)] != digest:
                raise FinragError(f"Earlier audit input hash differs: {path}")
        for name, digest in validation["output_hashes"].items():
            path = Path(name)
            if not path.is_file():
                path = bge / folder / Path(name).name
            track(path)
            if hashes[str(path)] != digest:
                raise FinragError(f"Earlier audit output hash differs: {path}")
        track(validation_path)
    previous = read_json(bge / "diagnostics_run_1/diagnostic.json")
    depth = read_json(bge / "depth_context_run_1/comparison.json")
    ids = depth["pool_incomplete_question_ids"]
    previous_ids = [
        f["question_id"]
        for f in previous["failures"]
        if not f["bm25_metrics"]["complete_excerpt_coverage@50"]
    ]
    if len(ids) != 8 or set(ids) != set(previous_ids):
        raise FinragError("The two previous diagnostics disagree on the eight incomplete pools.")
    sealed_inputs(Path("artifacts/private/bm25_candidate_reranking_dev"))
    ref_path = (
        Path.cwd() / "artifacts/private/bm25_query_translation_dev/evaluation/references.jsonl"
    )
    track(ref_path)
    refs = [EvaluationQuestion.model_validate(q) for q in read_jsonl(ref_path)]
    if len(refs) != 28 or any(q.split != "dev" or not q.answerable for q in refs):
        raise FinragError("Audit requires the sealed 28 answerable dev references only.")
    reference_seal = read_json(bge / "depth_context_run_1/validation.json")["reference_sha256"]
    if hashes[str(ref_path)] != reference_seal:
        raise FinragError("Sealed reference hash differs.")
    route_path = Path("artifacts/private/bm25_company_routing_dev/run_1/predictions_top50.jsonl")
    routes = {r["question_id"]: r["cached_translation"] for r in read_jsonl(route_path)}
    track(route_path)
    request_path = Path("artifacts/private/phase2b/faiss_exchange/request.json")
    track(request_path)
    request = read_json(request_path)
    chunk_path = Path(request["inputs"]["chunks"]["path"])
    track(chunk_path)
    if hashes[str(chunk_path)] != request["inputs"]["chunks"]["sha256"]:
        raise FinragError("Corpus chunk seal differs.")
    chunks = [ModelChunk.model_validate(c).model_dump(mode="json") for c in read_jsonl(chunk_path)]
    if len(chunks) != 8325 or len({c["chunk_id"] for c in chunks}) != 8325:
        raise FinragError("Expected the sealed 8325 unique corpus chunks.")
    by_chunk = {c["chunk_id"]: c for c in chunks}
    for c in chunks:
        if hashlib.sha256(c["text"].encode("utf-8")).hexdigest() != c["content_sha256"]:
            raise FinragError("Corpus content hash differs.")
    pages, access = {}, []
    selected = [q for q in refs if q.question_id in ids]
    for filename in sorted(
        {c["source_filename"] for q in selected for c in page_chunks(q, chunks)}
    ):
        identity = next(c for c in chunks if c["source_filename"] == filename)
        passage = ContextPassage.model_validate(
            {k: identity[k] for k in ContextPassage.model_fields}
        )
        try:
            loaded, files = load_extracted_pages(Path("data/private/corpus/extracted"), [passage])
        except FileOperationError as error:
            access.append(
                {
                    "source_filename": filename,
                    "status": "inaccessible",
                    "operation": error.operation,
                    "path": str(error.path),
                    "exception_type": type(error.__cause__).__name__,
                    "message": str(error.__cause__),
                    "permissions_changed": False,
                }
            )
        else:
            pages.update(loaded)
            for path in files:
                track(path)
            access.append(
                {
                    "source_filename": filename,
                    "status": "validated",
                    "files": [str(p.absolute()) for p in files],
                }
            )
    rows = []
    for q in selected:
        route = routes[q.question_id]
        old = next(f for f in previous["failures"] if f["question_id"] == q.question_id)
        if route["query_used"] != old["query_used"] or len(route["hits"]) != 50:
            raise FinragError("Saved query or candidate depth differs.")
        for h in route["hits"]:
            c = by_chunk[h["chunk_id"]]
            for field in (
                "text",
                "document_id",
                "source_filename",
                "source_sha256",
                "physical_page_number",
            ):
                if h[field] != c[field]:
                    raise FinragError("Saved candidate and corpus provenance differ.")
        metrics = evaluate_candidates(q, route["hits"], False)
        normalized = json.loads(json.dumps(metrics))
        if (
            any(
                normalized[k] != value
                for k, value in old["bm25_metrics"].items()
                if k != "exhaustive_candidates"
            )
            or metrics["complete_excerpt_coverage@50"]
        ):
            raise FinragError("Historical incomplete-pool metrics differ.")
        occurrences = evidence_occurrences(q, chunks)
        if not all(e["on_reference_page"] for e in occurrences):
            raise FinragError("At least one v1 excerpt is absent from sealed corpus chunks.")
        evidence = []
        for index, item in enumerate(occurrences):
            e = item["evidence"]
            for c in item["on_reference_page"]:
                page = pages.get((c["document_id"], c["physical_page_number"]))
                if (
                    page
                    and page.text[c["source_character_start"] : c["source_character_end"]]
                    != c["text"]
                ):
                    raise FinragError("Chunk offsets disagree with validated extraction.")
            evidence.append(
                item
                | {
                    "present_in_pool_on_reference_page": metrics["reference_excerpt_ranks"][index]
                    is not None,
                    "saved_bm25_rank": metrics["reference_excerpt_ranks"][index],
                    "pool_occurrences_other_pages_unvalidated": [
                        h
                        for h in route["hits"]
                        if e["text"] in h["text"]
                        and h["physical_page_number"] != e["physical_page_number"]
                    ],
                    "vocabulary_by_proof_chunk": [
                        {
                            "chunk_id": c["chunk_id"],
                            "difference": vocabulary_difference(route["query_used"], c["text"]),
                        }
                        for c in item["on_reference_page"]
                    ],
                }
            )
        rows.append(
            {
                "question_id": q.question_id,
                "question_original": q.question,
                "query_used": route["query_used"],
                "reference": q.model_dump(mode="json"),
                "metrics": metrics,
                "evidence": evidence,
                "all_chunks_on_reference_pages": page_chunks(q, chunks),
                "candidates_on_reference_pages": [
                    h for h in route["hits"] if h["physical_page_number"] in q.relevant_pages
                ],
                "validated_extracted_reference_pages": [
                    p.model_dump(mode="json")
                    for (doc, number), p in pages.items()
                    if doc == q.document_id and number in q.relevant_pages
                ],
                "observations": [],
                "hypotheses": [],
                "missing_information": [],
            }
        )
    track(Path(__file__))
    # The report's curated observations are imported from a private analysis note.
    notes_path = output.parent / "missing_candidates_audit_notes.json"
    notes = read_json(notes_path)
    track(notes_path)
    for row in rows:
        row.update(notes["questions"][row["question_id"]])
    diagnostic = {
        "split": "dev",
        "reference_version": "v1",
        "question_count": 8,
        "corpus_chunks": 8325,
        "query_variant": "cached_translation",
        "lookup_method": (
            "literal reference inclusion in existing chunks; "
            "diagnostic audit only, no retrieval or scoring"
        ),
        "financial_sufficiency_validated": False,
        "reference_statuses_unchanged": True,
        "extraction_access": access,
        "rows": rows,
        "recommendation": notes["recommendation"],
    }
    report = [
        "# Audit privé des huit pools incomplets — dev v1",
        "",
        "Les huit cas sont reproduits depuis les deux diagnostics et l’évaluateur existant. "
        "Chaque extrait v1 manque aux pages attendues dans le pool, mais existe intégralement "
        "dans au moins un chunk scellé à sa page v1. Aucune nouvelle recherche ni inférence. "
        "Les occurrences sur d’autres pages restent des indices non validés.",
        "",
        "| ID | Pages physiques v1 | Extraits présents / attendus dans le pool "
        "| Observation vérifiée |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        report.append(
            f"| {row['question_id']} | {row['reference']['relevant_pages']} | "
            f"{row['metrics']['reference_excerpts_found@50']}/{len(row['evidence'])} | "
            f"{row['observations'][0]} |"
        )
    report += ["", "## Synthèse", "", *notes["summary"], "", "## Accès aux extractions", ""]
    for item in access:
        report.append(
            f"- `{item['source_filename']}` : {item['status']}. "
            + (
                f"`{item['path']}` — {item['exception_type']}."
                if item["status"] == "inaccessible"
                else "Empreintes, identité, séquence et offsets des chunks de preuve validés."
            )
        )
    report += [
        "",
        "Les pages PDF n’ont pas été rendues : aucun défaut visuel de tableau n’est "
        "affirmé. Les refus d’accès empêchent de confirmer le rendu et les associations "
        "graphiques CASA/SG. Le texte scellé ne remplace pas cette vérification.",
        "",
    ]
    for row in rows:
        q = row["reference"]
        report += [
            f"## {row['question_id']}",
            "",
            f"**Question originale :** {row['question_original']}",
            "",
            f"**Requête effective :** {row['query_used']}",
            "",
            f"**Document :** `{row['all_chunks_on_reference_pages'][0]['source_filename']}` ; "
            f"ID `{q['document_id']}` ; pages physiques {q['relevant_pages']}.",
            "",
            "**Contexte financier annoté v1 (non revalidé) :**",
            "",
            "```json",
            json_bytes(
                {
                    k: q[k]
                    for k in (
                        "period",
                        "unit",
                        "currency",
                        "scope",
                        "annotation_status",
                        "context_annotation_status",
                    )
                }
            )
            .decode()
            .strip(),
            "```",
            "",
        ]
        for i, e in enumerate(row["evidence"], 1):
            report += [
                f"### Extrait {i} — page physique {e['evidence']['physical_page_number']}",
                "",
                "```text",
                e["evidence"]["text"],
                "```",
                "",
                "**Pool :** absent à la page v1. Présences littérales sur d’autres pages : "
                + str(
                    [
                        (h["physical_page_number"], h["rank"])
                        for h in e["pool_occurrences_other_pages_unvalidated"]
                    ]
                )
                + ".",
                "",
            ]
            for c, vocab in zip(
                e["on_reference_page"], e["vocabulary_by_proof_chunk"], strict=True
            ):
                report += [
                    f"**Chunk corpus :** `{c['chunk_id']}`",
                    "",
                    f"Page {c['physical_page_number']} ; offsets [{c['source_character_start']}, "
                    f"{c['source_character_end']}) ; SHA source `{c['source_sha256']}` ; "
                    f"SHA texte `{c['content_sha256']}` ; parent `{c['parent_chunk_id']}`.",
                    "",
                    "```text",
                    c["text"],
                    "```",
                    "",
                    "Termes de requête absents de ce chunk (tokenizer BM25 existant) : "
                    + ", ".join(vocab["difference"]["query_only"])
                    + ".",
                    "",
                ]
            report += [
                "Autres chunks contenant cet extrait, "
                "pages non validées comme références alternatives : "
                + str(
                    [
                        (c["physical_page_number"], c["chunk_id"])
                        for c in e["other_pages_unvalidated"]
                    ]
                )
                + ".",
                "",
            ]
        for label, field in (
            ("Observations vérifiées", "observations"),
            ("Hypothèses", "hypotheses"),
            ("Informations manquantes", "missing_information"),
        ):
            report += [f"**{label} :**", ""] + [f"- {v}" for v in row[field]] + [""]
    report += [
        "## Une expérience limitée recommandée (non implémentée)",
        "",
        notes["recommendation"],
        "",
        "## Commande manuelle de complément d’extraction",
        "",
        "À exécuter seulement dans un terminal ayant déjà l’accès normal aux fichiers. "
        "Aucune modification des permissions. Cette commande valide les extractions et écrit "
        "uniquement les pages de preuve dans un nouveau fichier ; un refus reste explicite.",
        "",
        "```powershell",
        notes["manual_command"],
        "```",
        "",
        "Les listes complètes des vocabulaires, provenance, chunks des pages de référence et "
        "occurrences sont dans diagnostic.json. Les questions et textes restent privés. "
        "Aucune annotation candidate n’est promue ; aucune suffisance financière n’est déduite "
        "de la présence littérale. Les fichiers sources sont inchangés par empreintes.",
    ]
    for path in paths:
        if hash_file(path) != hashes[str(path)]:
            raise FinragError(f"Audit input changed during analysis: {path}")
    write_atomic(output / "diagnostic.json", json_bytes(diagnostic))
    write_atomic(output / "rapport.md", ("\n".join(report) + "\n").encode("utf-8"))
    validation = {
        "input_hashes": hashes,
        "reference_sha256": reference_seal,
        "output_hashes": {
            name: hash_file(output / name) for name in ("diagnostic.json", "rapport.md")
        },
        "commands": notes["commands"],
        "tests": notes["tests"],
        "checks": {
            "eight_ids_agree": True,
            "sealed_8325_chunks": True,
            "all_content_hashes_valid": True,
            "saved_candidates_equal_corpus": True,
            "all_missing_excerpts_present_in_corpus_on_v1_pages": True,
            "previous_metrics_reproduced": True,
            "all_inputs_unchanged": True,
            "new_retrieval": False,
            "new_inference": False,
            "test_access": False,
            "permissions_changed": False,
        },
        "extraction_access": access,
    }
    write_atomic(output / "validation.json", json_bytes(validation))
    return {"questions": len(rows), "input_files_preserved": len(paths), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(run(args.output))
    except FinragError as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
