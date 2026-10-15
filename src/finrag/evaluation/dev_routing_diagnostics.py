import argparse
import ast
import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

from finrag.config import FinragError, filesystem_operation, private_output
from finrag.domain.dense import DenseHit, DensePrediction
from finrag.domain.evaluation import EvaluationQuestion
from finrag.domain.retrieval import LexicalPrediction
from finrag.evaluation.split_guard import load_dev_questions
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, read_json, read_jsonl, write_atomic

PageKey = tuple[str, int]
Hit = dict[str, Any]

ORACLE_COMMAND_SOURCE = """import json
from pathlib import Path

import numpy as np

from finrag.evaluation.split_guard import load_dev_questions
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import read_json, read_jsonl

root = Path("artifacts/private/phase2b")
request = read_json(root / "faiss_exchange/request.json")
questions = load_dev_questions(Path("data/private/evaluation/finrag_v1"))
manifest = read_json(root / "queries/cache_manifest.json")
assert manifest["fingerprint"]["dev_sha256"] == hash_file(
    Path("data/private/evaluation/finrag_v1/dev.jsonl")
)
paths = {key: Path(value["path"]) for key, value in request["inputs"].items()}
for key, path in paths.items():
    assert path.resolve().is_relative_to(root.resolve())
    assert hash_file(path) == request["inputs"][key]["sha256"]
chunks = read_jsonl(paths["chunks"])
passages = np.load(paths["passages"], allow_pickle=False)
queries = np.load(paths["queries"], allow_pickle=False)
assert passages.shape == (len(chunks), 384) and queries.shape == (len(questions), 384)
assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
for array in (passages, queries):
    assert array.dtype == np.float32 and np.isfinite(array).all()
    assert np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5)
for query_row, question in enumerate(questions):
    if not question.answerable:
        continue
    rows = [i for i, chunk in enumerate(chunks)
            if chunk["document_id"] == question.document_id]
    scores = passages[rows] @ queries[query_row]
    ordered = sorted(range(len(rows)),
                     key=lambda i: (-float(scores[i]), chunks[rows[i]]["chunk_id"]))
    pages = list(dict.fromkeys(chunks[rows[i]]["physical_page_number"] for i in ordered))
    print(json.dumps({"question_id": question.question_id,
                      "document_passages_searched": len(rows),
                      "document_distinct_pages": len(pages),
                      "first_five_pages": pages[:5]}))
"""


def page_key(hit: Hit) -> PageKey:
    return hit["document_id"], hit["physical_page_number"]


def distinct_pages(hits: list[Hit]) -> list[Hit]:
    seen: set[PageKey] = set()
    pages = []
    for hit in hits:
        key = page_key(hit)
        if key not in seen:
            seen.add(key)
            pages.append(hit)
    return pages


def prefix_hit(hits: list[Hit], expected: set[PageKey], k: int = 5) -> bool | None:
    if k <= 0:
        raise FinragError("Diagnostic prefix size must be positive.")
    if any(page_key(hit) in expected for hit in hits[:k]):
        return True
    # An absent reference in an incomplete saved prefix is unknown, not a negative.
    return False if len(hits) >= k else None


def first_reference_rank(hits: list[Hit], expected: set[PageKey]) -> int | None:
    return next((rank for rank, hit in enumerate(hits, 1) if page_key(hit) in expected), None)


def ranking_diagnostic(hits: list[Hit], expected: set[PageKey], document: str) -> dict:
    pages = distinct_pages(hits)
    oracle = [hit for hit in hits if hit["document_id"] == document]
    oracle_pages = distinct_pages(oracle)
    return {
        "saved_chunk_depth": len(hits),
        "saved_distinct_page_depth": len(pages),
        "historical_hit5": any(page_key(hit) in expected for hit in hits[:5]),
        "expected_document_present_top5": any(hit["document_id"] == document for hit in hits[:5]),
        "distinct_page_hit5": prefix_hit(pages, expected),
        "five_distinct_pages_available": len(pages) >= 5,
        "first_reference_chunk_rank": first_reference_rank(hits, expected),
        "first_reference_distinct_page_rank": first_reference_rank(pages, expected),
        "oracle_label": "oracle limité au classement sauvegardé",
        "oracle_saved_chunk_depth": len(oracle),
        "oracle_saved_distinct_page_depth": len(oracle_pages),
        "oracle_five_chunks_available": len(oracle) >= 5,
        "oracle_five_pages_available": len(oracle_pages) >= 5,
        "oracle_chunk_hit5": prefix_hit(oracle, expected),
        "oracle_page_hit5": prefix_hit(oracle_pages, expected),
        "oracle_first_reference_chunk_rank": first_reference_rank(oracle, expected),
        "oracle_first_reference_page_rank": first_reference_rank(oracle_pages, expected),
        "absence_meaning": "not observed within saved ranking; not exhaustive absence",
    }


def fraction(values: list[bool]) -> dict:
    numerator = sum(values)
    denominator = len(values)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def prefix_summary(rows: list[dict], value: str, complete: str) -> dict:
    eligible = [row for row in rows if row[complete]]
    incomplete = [row for row in rows if not row[complete]]
    return {
        "complete_prefix_only": fraction([bool(row[value]) for row in eligible]),
        "incomplete_questions": len(incomplete),
        "observed_positive_incomplete": sum(row[value] is True for row in incomplete),
        "unknown_incomplete": sum(row[value] is None for row in incomplete),
        "total_questions": len(rows),
    }


def aggregate(rows: list[dict]) -> dict:
    return {
        "historical_hit5": fraction([row["historical_hit5"] for row in rows]),
        "expected_document_present_top5": fraction(
            [row["expected_document_present_top5"] for row in rows]
        ),
        "distinct_page_hit5": prefix_summary(
            rows, "distinct_page_hit5", "five_distinct_pages_available"
        ),
        "oracle_chunk_hit5": prefix_summary(
            rows, "oracle_chunk_hit5", "oracle_five_chunks_available"
        ),
        "oracle_page_hit5": prefix_summary(rows, "oracle_page_hit5", "oracle_five_pages_available"),
    }


def breakdown(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[f"language:{row['language']}"].append(row)
        groups[f"document:{row['expected_document']}"].append(row)
        groups[f"language_document:{row['language']} / {row['expected_document']}"].append(row)
    return {name: aggregate(group) for name, group in sorted(groups.items())}


def validate_ranking(hits: list[Hit]) -> None:
    if [hit["rank"] for hit in hits] != list(range(1, len(hits) + 1)):
        raise FinragError("Saved ranking must have consecutive ordered ranks starting at one.")
    if len({hit["chunk_id"] for hit in hits}) != len(hits):
        raise FinragError("Saved ranking contains duplicate chunk IDs.")


def load_predictions(path: Path, questions: list[EvaluationQuestion], dense: bool) -> list[dict]:
    model = DensePrediction if dense else LexicalPrediction
    rows = [model.model_validate(row).model_dump(mode="json") for row in read_jsonl(path)]
    by_question = {row["question_id"]: row for row in rows}
    if len(by_question) != len(rows) or set(by_question) != {q.question_id for q in questions}:
        raise FinragError(f"{path}: predictions must match dev IDs exactly once.")
    ordered = []
    for question in questions:
        row = by_question[question.question_id]
        targets = {
            (target["document_id"], target["physical_page_number"])
            for target in row["expected_pages"]
        }
        if row["answerable"] != question.answerable or targets != {
            (question.document_id, page) for page in question.relevant_pages
        }:
            raise FinragError(f"{path}: answerability or references differ from frozen dev.")
        validate_ranking(row["retrieved"])
        ordered.append(row)
    return ordered


def inspect_npy(path: Path) -> dict:
    with filesystem_operation("read NPY header", path), path.open("rb") as source:
        if source.read(6) != b"\x93NUMPY":
            raise FinragError(f"{path}: invalid NPY magic.")
        version = tuple(source.read(2))
        if version not in ((1, 0), (2, 0), (3, 0)):
            raise FinragError(f"{path}: unsupported NPY version.")
        width = 2 if version == (1, 0) else 4
        length = struct.unpack("<H" if width == 2 else "<I", source.read(width))[0]
        if length > 65536:
            raise FinragError(f"{path}: unexpectedly large NPY header.")
        header = ast.literal_eval(source.read(length).decode("utf-8"))
    if header["descr"] != "<f4" or header["fortran_order"]:
        raise FinragError(f"{path}: expected C-order float32 cache.")
    return {"path": str(path), "format": "NPY C-order float32", "shape": header["shape"]}


def historical_check(rows: list[dict], stored: dict) -> dict:
    by_id = {row["question_id"]: row for row in rows if row["answerable"]}
    for qid, row in by_id.items():
        if row["historical_hit5"] != bool(stored[qid]):
            raise FinragError(f"Historical Hit@5 differs for {qid}.")
    result = fraction([row["historical_hit5"] for row in by_id.values()])
    return {"all_question_hit5_reproduced": True, **result}


def diagnostic_rows(
    questions: list[EvaluationQuestion], predictions: list[dict], names: dict[str, str]
) -> tuple[list[dict], list[dict]]:
    answerable, unanswerable = [], []
    for question, prediction in zip(questions, predictions, strict=True):
        hits = prediction["retrieved"]
        base = {
            "question_id": question.question_id,
            "language": question.language,
            "expected_document": names[question.document_id],
            "top5_documents": [names[hit["document_id"]] for hit in hits[:5]],
            "saved_page_order": [
                {
                    "document": names[hit["document_id"]],
                    "physical_page": hit["physical_page_number"],
                }
                for hit in distinct_pages(hits)
            ],
        }
        if not question.answerable:
            unanswerable.append(
                {
                    **base,
                    "saved_chunk_depth": len(hits),
                    "metrics": None,
                    "reason": "excluded: no positive reference",
                }
            )
            continue
        expected = {(question.document_id, page) for page in question.relevant_pages}
        answerable.append(
            {
                **base,
                "reference_pages": question.relevant_pages,
                "answerable": True,
                **ranking_diagnostic(hits, expected, question.document_id),
            }
        )
    return answerable, unanswerable


def format_fraction(value: dict) -> str:
    if not value["denominator"]:
        return "0/0 (non disponible)"
    return f"{value['numerator']}/{value['denominator']} ({value['rate']:.1%})"


def render_report(report: dict) -> str:
    lines = [
        "# Diagnostic dev A : documents, langues et classements sauvegardés",
        "",
        "Analyse privée, dev uniquement. Aucune recherche, inférence, traduction ou "
        "modification des références. Le split test n'a pas été consulté.",
        "",
        "## Formats, segmentations et profondeur réelle",
        "",
        "Phase 2A : JSONL LexicalPrediction/ RetrievalHit (lexical_score). "
        "A=1000/100, B=2000/200, C=3000/300 caractères. Phase 2B : JSONL "
        "DensePrediction/DenseHit (score, method, component_ranks), même pour BM25. "
        "Les trois méthodes Phase 2B utilisent les mêmes 8325 enfants E5, contre "
        "7141 fenêtres B. Les résultats Phase 2A ne sont pas un contrôle à segmentation "
        "identique ; le BM25 apparié Phase 2B l'est.",
        "",
        "| Source | Questions dev | Profondeur segments min–max | Pages distinctes min–max |",
        "| --- | --- | --- | --- |",
    ]
    for source, data in report["formats"].items():
        lines.append(
            f"| {source} | {data['questions']} | {data['chunk_min']}–"
            f"{data['chunk_max']} | {data['page_min']}–{data['page_max']} |"
        )
    lines.extend(
        [
            "",
            "Les prédictions finales sont des préfixes, pas des classements exhaustifs. "
            "Le JSON FAISS contient 50 candidats denses par ligne de requête, dans l'ordre "
            "du dev vérifié par le fingerprint et la correspondance du top 10. "
            "Le top 50 BM25 et le top 50 hybride ne sont pas sauvegardés. "
            "Leur reconstruction exigerait une nouvelle recherche : elle n'est pas faite.",
            "",
            "Caches : checkpoints NPY float32 avec métadonnées JSON et hashes ; "
            "échange FAISS avec JSONL des lignes/chunks et matrices NPY ; index natif FAISS "
            "avec rows.jsonl et manifeste. Aucun pickle ni chargement du modèle.",
            "",
            "## Vérification historique avant nouvelles métriques",
            "",
            "Hit@5 = au moins une paire document/page de référence dans les cinq premiers "
            "segments. Une comparaison peut donc réussir avec un seul opérande. "
            "Les non-répondables sont exclus, sans score de refus.",
            "",
        ]
    )
    for method, result in report["historical_verification"].items():
        lines.append(f"- {method} : {format_fraction(result)}, égalité par question et globale.")
    for source, data in report["analyses"].items():
        lines.extend(
            [
                "",
                f"## {source}",
                "",
                "### Ventilation historique",
                "",
                "| Groupe | Hit@5 historique | Document attendu présent top 5 |",
                "| --- | --- | --- |",
            ]
        )
        for name, values in data["groups"].items():
            lines.append(
                f"| {name} | {format_fraction(values['historical_hit5'])} | "
                f"{format_fraction(values['expected_document_present_top5'])} |"
            )
        overall = data["groups"]["overall"]
        lines.extend(
            [
                "",
                "### Nouvelles métriques et oracle limité au classement sauvegardé",
                "",
                "| Mesure | Succès / questions avec préfixe suffisant | "
                "Préfixes incomplets | Succès observés parmi incomplets | Inconnus |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for key, label in (
            ("distinct_page_hit5", "Hit@5 sur cinq pages distinctes"),
            ("oracle_chunk_hit5", "Oracle : cinq segments du document attendu"),
            ("oracle_page_hit5", "Oracle : cinq pages distinctes du document attendu"),
        ):
            value = overall[key]
            lines.append(
                f"| {label} | {format_fraction(value['complete_prefix_only'])} | "
                f"{value['incomplete_questions']} | "
                f"{value['observed_positive_incomplete']} | "
                f"{value['unknown_incomplete']} |"
            )
        lines.extend(
            [
                "",
                "Le dénominateur des nouvelles métriques est explicitement limité aux "
                "questions ayant cinq éléments disponibles. Un succès observé dans un "
                "préfixe plus court est conservé séparément ; une absence y reste inconnue. "
                "L'oracle filtre le document attendu avant de tronquer/dédupliquer. "
                "Il utilise la référence et ne constitue pas un routeur utilisable en démo.",
                "",
                "### Observations par question répondable",
                "",
                "`—` signifie non observé dans la profondeur sauvegardée, jamais une absence "
                "exhaustive. Les rangs pages suivent la première apparition de chaque paire.",
                "",
                "| ID | Langue / document attendu | Hit5 | Doc top5 | Profondeur "
                "segments/pages | Rang référence segments/pages | Hit5 pages | "
                "Oracle profondeur segments/pages | Oracle Hit5 segments/pages |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in data["questions"]:
            rank = {key: value if value is not None else "—" for key, value in row.items()}
            lines.append(
                f"| {row['question_id']} | {row['language']} / "
                f"{row['expected_document']} | {row['historical_hit5']} | "
                f"{row['expected_document_present_top5']} | "
                f"{row['saved_chunk_depth']}/{row['saved_distinct_page_depth']} | "
                f"{rank['first_reference_chunk_rank']}/"
                f"{rank['first_reference_distinct_page_rank']} | "
                f"{rank['distinct_page_hit5']} | {row['oracle_saved_chunk_depth']}/"
                f"{row['oracle_saved_distinct_page_depth']} | "
                f"{rank['oracle_chunk_hit5']}/{rank['oracle_page_hit5']} |"
            )
        lines.extend(["", "### Non-répondables : hors métriques", ""])
        for row in data["unanswerable"]:
            lines.append(
                f"- {row['question_id']} : {row['language']}, "
                f"{row['saved_chunk_depth']} segments disponibles ; aucune référence positive."
            )
    lines.extend(
        [
            "",
            "## Conclusions étayées et hypothèses ouvertes",
            "",
            "Les tableaux séparent deux observations : le mauvais document peut monopoliser "
            "le préfixe, ou le bon document peut être présent sans page de référence. "
            "La présence du document attendu ne garantit ni les preuves ni les deux operands. "
            "Les résultats par langue/document sont descriptifs sur un petit dev ; langue "
            "et document ne sont pas indépendants. Aucun effet causal de la langue, du modèle, "
            "des tableaux ou des répétitions n'est établi.",
            "",
            "L'écart entre positions de segments et pages distinctes mesure l'effet "
            "observable des doublons dans le préfixe disponible. L'oracle indique ce qui "
            "reste après filtrage, sans connaître les candidats écartés en amont. "
            "Ses dénominateurs sélectionnés ne permettent pas de le comparer directement "
            "au Hit@5 historique sur 28 questions.",
            "",
            "Les observations structurées, y compris les documents du top 5 et l'ordre "
            "complet des pages disponibles, sont dans diagnostic.json. Les références "
            "alternatives privées ne sont pas utilisées ; aucune référence v2 n'est créée.",
            "",
            "## Proposition de recherche restreinte réelle — non exécutée",
            "",
            "Les caches validés contiennent 8325 × 384 vecteurs passages et 30 × 384 "
            "vecteurs requêtes float32, associés aux lignes du dev. Une future recherche "
            "dense peut filtrer les lignes de passages par document avant le produit "
            "scalaire, puis trier avec chunk_id pour les égalités. Cela réutilise tous "
            "les vecteurs du document, sans ré-encodage et sans importer torch/FAISS.",
            "",
            "Commande proposée pour une future tâche :",
            "",
            "```powershell",
            f"& .\\.venv\\Scripts\\python.exe {report['oracle_command_path']}",
            "```",
            "",
            "Le script proposé est joint, mais n'a pas été exécuté. Il vérifie les hashes "
            "des matrices et des lignes dans le request, le fingerprint dev du cache requêtes, "
            "les dimensions, valeurs finies et normes. Il écrit uniquement des IDs et pages "
            "sur stdout, sans modifier de cache ou score. Il utilise le document de référence "
            "et reste donc un diagnostic oracle. Pour BM25, une "
            "nouvelle recherche sur les mêmes enfants peut être faite sans embeddings ; "
            "un hybride restreint exige les deux classements recalculés dans le document.",
            "",
            "Coût estimé, non mesuré : passages ~12,2 Mio, requêtes ~45 Kio, "
            "métadonnées JSONL en plus. Borne supérieure de 30 × 8325 × 384 ≈ 96 millions "
            "de produits scalaires élémentaires avant filtrage. Ordre de grandeur : "
            "quelques secondes à quelques dizaines de secondes sur ce CPU avec validation "
            "et lecture disque ; aucun engagement de latence ni benchmark réalisé.",
            "",
            "## Intégrité",
            "",
            "Les hashes avant/après de tous les fichiers lus sont conservés dans "
            "diagnostic.json. La lecture du test, l'encodage, les appels externes et la "
            "modification des fichiers d'entrée sont absents de cette commande.",
            "",
        ]
    )
    return "\n".join(lines)


def run_diagnostic(project_root: Path, output: Path) -> dict:
    output = private_output(output, project_root)
    if output.exists():
        raise FinragError("Choose a new diagnostic directory; existing output is preserved.")
    root = project_root
    phase2b = root / "artifacts/private/phase2b"
    dataset = root / "data/private/evaluation/finrag_v1"
    paths = [
        dataset / "dev.jsonl",
        dataset / "dataset_manifest.json",
        root / "experiments/phase2b/protocol.json",
        root / "experiments/phase2b/reports/dev_results.json",
        phase2b / "token_preflight.json",
        phase2b / "faiss_exchange/request.json",
        phase2b / "faiss_exchange/results.json",
        phase2b / "queries/cache_manifest.json",
        phase2b / "embeddings/cache_manifest.json",
        phase2b / "index/index_manifest.json",
        phase2b / "index/rows.jsonl",
    ]
    paths += [root / f"experiments/phase2a/configs/{name}.json" for name in "ABC"]
    paths += [root / f"artifacts/private/phase2a/{name}/predictions.jsonl" for name in "ABC"]
    paths += [phase2b / name / "predictions.jsonl" for name in ("bm25", "dense", "hybrid")]
    request = read_json(phase2b / "faiss_exchange/request.json")
    cache_inputs = {kind: Path(record["path"]) for kind, record in request["inputs"].items()}
    if any(not path.resolve().is_relative_to(phase2b.resolve()) for path in cache_inputs.values()):
        raise FinragError("FAISS exchange input must remain inside the Phase 2B private root.")
    paths += list(cache_inputs.values())
    before = {str(path): hash_file(path) for path in paths}
    questions = load_dev_questions(dataset)
    configs = {name: read_json(root / f"experiments/phase2a/configs/{name}.json") for name in "ABC"}
    protocol = read_json(root / "experiments/phase2b/protocol.json")
    token_preflight = read_json(phase2b / "token_preflight.json")
    for kind, record in request["inputs"].items():
        if before[str(cache_inputs[kind])] != record["sha256"]:
            raise FinragError(f"FAISS {kind} cache hash differs from request.")
    query_manifest = read_json(phase2b / "queries/cache_manifest.json")
    if query_manifest["fingerprint"]["dev_sha256"] != before[str(dataset / "dev.jsonl")]:
        raise FinragError("Cached query row order is not bound to the current frozen dev.")
    embedding_manifest = read_json(phase2b / "embeddings/cache_manifest.json")
    index_manifest = read_json(phase2b / "index/index_manifest.json")
    if (
        embedding_manifest["fingerprint"] != request["fingerprint"]
        or index_manifest["fingerprint"] != request["fingerprint"]
    ):
        raise FinragError("Embedding/index fingerprints differ from saved FAISS exchange.")
    chunks = read_jsonl(cache_inputs["chunks"])
    index_rows = read_jsonl(phase2b / "index/rows.jsonl")
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    if (
        len(chunks_by_id) != len(chunks)
        or len(index_rows) != len(chunks)
        or {row["chunk_id"]: row for row in index_rows} != chunks_by_id
    ):
        raise FinragError("Index row metadata differs from cached passage IDs.")
    names = {chunk["document_id"]: chunk["source_filename"] for chunk in chunks}
    array_formats = {kind: inspect_npy(cache_inputs[kind]) for kind in ("passages", "queries")}
    if array_formats["passages"]["shape"] != (len(chunks), 384) or array_formats["queries"][
        "shape"
    ] != (len(questions), 384):
        raise FinragError("Cached array dimensions do not match passages/dev query rows.")
    saved = {}
    for name in "ABC":
        saved[f"phase2a_{name}"] = load_predictions(
            root / f"artifacts/private/phase2a/{name}/predictions.jsonl", questions, False
        )
    for method in ("bm25", "dense", "hybrid"):
        saved[method] = load_predictions(phase2b / method / "predictions.jsonl", questions, True)
        if any(row["method"] != method for row in saved[method]):
            raise FinragError("Prediction method differs from its source directory.")
    stored = read_json(root / "experiments/phase2b/reports/dev_results.json")
    verification = {}
    # Reproduce the frozen definition before deriving page or oracle diagnostics.
    for method in ("bm25", "dense", "hybrid"):
        rows = [
            {
                "question_id": question.question_id,
                "answerable": True,
                "historical_hit5": any(
                    page_key(hit) in {(question.document_id, p) for p in question.relevant_pages}
                    for hit in prediction["retrieved"][:5]
                ),
            }
            for question, prediction in zip(questions, saved[method], strict=True)
            if question.answerable
        ]
        targets = {
            row["question_id"]: row["methods"][method]["metrics"]["hit@5"]
            for row in stored["comparison"]
            if row["answerable"]
        }
        if set(targets) != {row["question_id"] for row in rows}:
            raise FinragError("Historical comparison has different answerable dev IDs.")
        verification[method] = historical_check(rows, targets)
        overall = stored["groups"][method]["overall"]
        if overall["answerable"] != len(rows) or not math.isclose(
            verification[method]["rate"], overall["metrics"]["hit@5"], abs_tol=1e-12
        ):
            raise FinragError("Historical global Hit@5 differs from reproduced counts.")
    exchange = read_json(phase2b / "faiss_exchange/results.json")
    if exchange["request_sha256"] != before[str(phase2b / "faiss_exchange/request.json")]:
        raise FinragError("Saved FAISS results refer to another request.")
    if len(exchange["hits"]) != len(questions) or exchange["query_count"] != len(questions):
        raise FinragError("FAISS rows must match all dev questions.")
    dense50 = []
    for prediction, hits in zip(saved["dense"], exchange["hits"], strict=True):
        hits = [DenseHit.model_validate(hit).model_dump(mode="json") for hit in hits]
        validate_ranking(hits)
        for hit in hits:
            chunk = chunks_by_id.get(hit["chunk_id"])
            if chunk is None or any(
                hit[field] != chunk[field]
                for field in (
                    "document_id",
                    "physical_page_number",
                    "source_filename",
                    "source_sha256",
                    "text",
                )
            ):
                raise FinragError("Saved dense candidate provenance differs from cache metadata.")
        if hits[: len(prediction["retrieved"])] != prediction["retrieved"]:
            raise FinragError("Saved FAISS top 50 does not reproduce dense predictions.")
        dense50.append({**prediction, "retrieved": hits})
    saved["dense_saved50"] = dense50
    formats = {}
    analyses = {}
    for name, predictions in saved.items():
        depths = [len(row["retrieved"]) for row in predictions]
        pages = [len(distinct_pages(row["retrieved"])) for row in predictions]
        formats[name] = {
            "questions": len(predictions),
            "chunk_min": min(depths),
            "chunk_max": max(depths),
            "page_min": min(pages),
            "page_max": max(pages),
        }
        if name.startswith("phase2a"):
            continue
        rows, unanswerable = diagnostic_rows(questions, predictions, names)
        analyses[name] = {
            "groups": breakdown(rows),
            "questions": rows,
            "unanswerable": unanswerable,
        }
    after = {str(path): hash_file(path) for path in paths}
    if before != after:
        raise FinragError("An existing diagnostic input changed while reading it.")
    report = {
        "split": "dev",
        "test_read": False,
        "new_search": False,
        "new_encoding": False,
        "historical_verification": verification,
        "formats": formats,
        "configurations": {"phase2a": configs, "phase2b": protocol},
        "segmentation_counts": {"B": token_preflight["original_B_count"], "phase2b": len(chunks)},
        "cache_arrays": array_formats,
        "cache_metadata": {
            "passage_checkpoint_rows": embedding_manifest["rows"],
            "query_checkpoint_rows": query_manifest["rows"],
            "native_index_rows": len(index_rows),
            "fingerprints_match_exchange": True,
        },
        "analyses": analyses,
        "oracle_command_path": str(output / "oracle_dense_cache_propose.py"),
        "input_hashes_before": before,
        "input_hashes_after": after,
        "all_read_inputs_unchanged": True,
    }
    write_atomic(output / "diagnostic.json", json_bytes(report))
    write_atomic(output / "rapport.md", render_report(report).encode("utf-8"))
    write_atomic(output / "oracle_dense_cache_propose.py", ORACLE_COMMAND_SOURCE.encode("utf-8"))
    return {
        "report": str(output / "rapport.md"),
        "data": str(output / "diagnostic.json"),
        "historical_verification": verification,
        "all_read_inputs_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev diagnostic of saved rankings; no search")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run_diagnostic(Path.cwd(), args.output))


if __name__ == "__main__":
    main()
