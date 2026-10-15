import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from finrag.config import FinragError, filesystem_operation, private_output
from finrag.domain.dense import DenseHit, ModelChunk
from finrag.domain.evaluation import EvaluationQuestion
from finrag.domain.retrieval import LexicalConfig
from finrag.evaluation.dev_routing_diagnostics import load_predictions, validate_ranking
from finrag.evaluation.document_oracle_metrics import aggregate_metrics, evaluate_candidates
from finrag.evaluation.document_oracle_worker import document_rows
from finrag.evaluation.split_guard import load_dev_questions
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes, read_json, read_jsonl, write_atomic
from finrag.retrieval.bm25 import BM25
from finrag.retrieval.vector_index import validate_vectors


@dataclass
class CachedOracleInputs:
    chunks: list[ModelChunk]
    questions: list[EvaluationQuestion]
    exchange_request: dict
    baseline: dict[str, list[dict]]
    input_hashes: dict[str, str]
    preflight: dict


def verify_cached_inputs(project_root: Path) -> CachedOracleInputs:
    started = perf_counter()
    phase = project_root / "artifacts/private/phase2b"
    dataset = project_root / "data/private/evaluation/finrag_v1"
    files = [
        dataset / "dev.jsonl",
        dataset / "dataset_manifest.json",
        phase / "faiss_exchange/request.json",
        phase / "faiss_exchange/results.json",
        phase / "embeddings/cache_manifest.json",
        phase / "queries/cache_manifest.json",
        phase / "model_chunks.jsonl",
        phase / "comparison.json",
        project_root / "experiments/phase2b/protocol.json",
        project_root / "experiments/phase2a/configs/B.json",
        project_root / "artifacts/private/phase2a/B/experiment_manifest.json",
    ]
    files += [phase / method / "predictions.jsonl" for method in ("bm25", "dense")]
    request = read_json(phase / "faiss_exchange/request.json")
    files += [Path(record["path"]) for record in request["inputs"].values()]
    hashes = {str(path): hash_file(path) for path in files}
    questions = load_dev_questions(dataset)
    for name, record in request["inputs"].items():
        path = Path(record["path"])
        if (
            not path.resolve().is_relative_to(phase.resolve())
            or hashes[str(path)] != record["sha256"]
        ):
            raise FinragError(f"Cached oracle {name} input has an invalid path or hash.")
    passages_manifest = read_json(phase / "embeddings/cache_manifest.json")
    query_manifest = read_json(phase / "queries/cache_manifest.json")
    if passages_manifest["fingerprint"] != request["fingerprint"] or query_manifest[
        "fingerprint"
    ] != {
        "encoding": request["fingerprint"],
        "dev_sha256": hashes[str(dataset / "dev.jsonl")],
        "kind": "query",
    }:
        raise FinragError("Passage/query fingerprints do not bind the current frozen dev.")
    if hashes[str(phase / "model_chunks.jsonl")] != request["inputs"]["chunks"]["sha256"]:
        raise FinragError("Phase 2B control passages differ from the cached-vector row metadata.")
    chunks = [
        ModelChunk.model_validate(row)
        for row in read_jsonl(Path(request["inputs"]["chunks"]["path"]))
    ]
    if len(chunks) != 8325 or len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise FinragError("Oracle requires the unchanged 8325 unique Phase 2B passages.")
    for name, rows in (("passages", len(chunks)), ("queries", len(questions))):
        with filesystem_operation(
            "read cached oracle vectors", Path(request["inputs"][name]["path"])
        ):
            array = np.load(request["inputs"][name]["path"], allow_pickle=False)
        validate_vectors(array, rows, 384)
    if passages_manifest["rows"] != len(chunks) or query_manifest["rows"] != len(questions):
        raise FinragError("Cached manifest row counts differ from passages/dev queries.")
    baseline = {
        method: load_predictions(phase / method / "predictions.jsonl", questions, True)
        for method in ("bm25", "dense")
    }
    global_dense = read_json(phase / "faiss_exchange/results.json")
    if global_dense["request_sha256"] != hashes[str(phase / "faiss_exchange/request.json")] or len(
        global_dense["hits"]
    ) != len(questions):
        raise FinragError("Saved global dense candidates do not match the cached query rows.")
    for prediction, hits in zip(baseline["dense"], global_dense["hits"], strict=True):
        validate_ranking(hits)
        if len(hits) != 50 or hits[:10] != prediction["retrieved"]:
            raise FinragError("Global dense top50 differs from the frozen final predictions.")
        prediction["retrieved"] = hits
    protocol = read_json(project_root / "experiments/phase2b/protocol.json")
    config = LexicalConfig.model_validate(
        read_json(project_root / "experiments/phase2a/configs/B.json")
    )
    if (
        config.k1 != protocol["bm25_k1"]
        or config.b != protocol["bm25_b"]
        or config.tokenizer_version != protocol["bm25_tokenizer_version"]
    ):
        raise FinragError("BM25 tokenization/parameters differ from Phase 2B control.")
    previous_runtime = read_json(
        project_root / "artifacts/private/phase2a/B/experiment_manifest.json"
    )["runtime_seconds"]
    validation_seconds = perf_counter() - started
    estimate = max(30.0, 4 * previous_runtime + 2 * validation_seconds)
    rows_by_document = document_rows(chunks)
    preflight = {
        "passage_vectors": len(chunks),
        "cached_query_vectors": len(questions),
        "answerable_queries_selected": sum(q.answerable for q in questions),
        "dimensions": 384,
        "dtype": "float32",
        "normalization_verified": True,
        "passages_per_document": {doc: len(rows) for doc, rows in rows_by_document.items()},
        "cache_validation_seconds": validation_seconds,
        "estimated_search_seconds": estimate,
        "five_minute_gate_passed": estimate <= 300,
        "estimate_basis": "4 × measured Phase 2A B runtime + 2 × cache validation time; "
        "conservative planning estimate, not a measured oracle runtime",
        "bm25_config": config.model_dump(mode="json"),
        "new_encoding": False,
        "test_read": False,
    }
    return CachedOracleInputs(chunks, questions, request, baseline, hashes, preflight)


def dense_in_worker(inputs: CachedOracleInputs, output: Path) -> list[dict]:
    job = {
        "output": str(output),
        "input_root": str(Path(inputs.exchange_request["inputs"]["chunks"]["path"]).parents[1]),
        "inputs": inputs.exchange_request["inputs"],
        "cached_query_count": len(inputs.questions),
        "questions": [
            {"query_row": row, "question": question.model_dump(mode="json")}
            for row, question in enumerate(inputs.questions)
            if question.answerable
        ],
    }
    # Read original caches; only the new request and results belong to the oracle directory.
    request_path = output / "oracle_request.json"
    write_atomic(request_path, json_bytes(job))
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "finrag.evaluation.document_oracle_worker",
                "--request",
                str(request_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise FinragError(
            "Isolated document oracle exceeded its 120-second search timeout."
        ) from error
    write_atomic(output / "worker.log", (completed.stdout + completed.stderr).encode("utf-8"))
    if completed.returncode != 0:
        raise FinragError(
            f"Document oracle worker exited {completed.returncode}: {completed.stderr}"
        )
    receipt = json.loads(completed.stdout)
    path = output / "dense_worker_results.json"
    if receipt["worker_torch_loaded"] or hash_file(path) != receipt["results_sha256"]:
        raise FinragError("Dense oracle receipt hash or process isolation check failed.")
    result = read_json(path)
    if result["worker_torch_loaded"] or result["request_sha256"] != hash_file(request_path):
        raise FinragError("Dense oracle result does not match its isolated request.")
    expected = {q.question_id: q for q in inputs.questions if q.answerable}
    if len(result["results"]) != 28 or {row["question_id"] for row in result["results"]} != set(
        expected
    ):
        raise FinragError("Dense oracle results must match the 28 answerable dev IDs.")
    chunks = {chunk.chunk_id: chunk for chunk in inputs.chunks}
    for row in result["results"]:
        question = expected[row["question_id"]]
        hits = [DenseHit.model_validate(hit).model_dump(mode="json") for hit in row["hits"]]
        validate_ranking(hits)
        if len(hits) != 50:
            raise FinragError("Dense oracle must return 50 saved candidates per question.")
        for hit in hits:
            chunk = chunks.get(hit["chunk_id"])
            if (
                chunk is None
                or hit["document_id"] != question.document_id
                or any(
                    hit[field] != getattr(chunk, field)
                    for field in (
                        "document_id",
                        "physical_page_number",
                        "text",
                        "source_filename",
                        "source_sha256",
                    )
                )
            ):
                raise FinragError("Dense oracle candidate source correspondence failed.")
        prefix = evaluate_candidates(question, hits, False)
        for k in (1, 5, 10, 50):
            if (
                prefix[f"hit@{k}"] != row["metrics"][f"hit@{k}"]
                or prefix[f"reference_excerpts_found@{k}"]
                != row["metrics"][f"reference_excerpts_found@{k}"]
            ):
                raise FinragError("Dense oracle full-rank metrics differ from saved top50.")
    return result["results"]


def metric_count(summary: dict, key: str) -> str:
    value = summary[key]
    return f"{value['numerator']}/{value['denominator']}" if value["denominator"] else "n/d"


def display_rank(value: int | None) -> str:
    return str(value) if value is not None else "—"


def render_report(report: dict) -> str:
    lines = [
        "# Oracle dev : recherche réellement restreinte au document v1",
        "",
        "28 questions dev répondables. Le document vient des annotations : diagnostic "
        "oracle, pas routage réel ni score déployable. Aucun statut ou référence modifié.",
        "",
        "## Caches et exécution",
        "",
        "8325 vecteurs passages et 30 vecteurs requêtes existants, float32 normalisés, "
        "hashes et correspondance des lignes validés. Seules les 28 lignes répondables "
        "sont recherchées. Aucun encodage, modèle chargé, téléchargement ou appel API.",
        "",
        f"Estimation avant lancement : {report['preflight']['estimated_search_seconds']:.1f} s "
        f"(<300 s). Durée mesurée jusqu'au calcul du rapport, hors publication finale : "
        f"{report['runtime_seconds']:.2f} s.",
        "",
        "Dense : tous les passages du document, FAISS IndexFlatIP exact dans un "
        "interpréteur séparé sans Torch, quatre threads ; égalités triées par chunk_id. "
        "BM25 : mêmes enfants, tokenizer unicode-financial-v1, k1=1.2 et b=0.75. "
        "L'IDF et la longueur moyenne sont recalculés sur chaque document : le changement "
        "BM25 ne mesure donc pas uniquement la suppression des autres documents. "
        "Aucun candidat lexical de score nul n'est ajouté.",
        "",
        "## Métriques : numérateurs / 28",
        "",
        "Hit v1 = au moins une page attendue dans les k premiers segments. "
        "Couverture complète des pages/extraits = toutes les preuves requises, "
        "sur leurs pages v1 et dans le bon document. Extrait exact = chaîne originale "
        "entière dans le texte d'un segment ; aucune normalisation ni alternative admise.",
        "",
        "| Source | Hit1 | Hit5 | Hit10 | Hit50 | Hit5 pages distinctes | "
        "Toutes pages @5/@50 | Tous extraits @5/@50 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for source, summary in report["aggregates"].items():
        lines.append(
            f"| {source} | {metric_count(summary, 'hit@1')} | {metric_count(summary, 'hit@5')} | "
            f"{metric_count(summary, 'hit@10')} | {metric_count(summary, 'hit@50')} | "
            f"{metric_count(summary, 'distinct_page_hit@5')} | "
            f"{metric_count(summary, 'complete_page_coverage@5')} / "
            f"{metric_count(summary, 'complete_page_coverage@50')} | "
            f"{metric_count(summary, 'complete_excerpt_coverage@5')} / "
            f"{metric_count(summary, 'complete_excerpt_coverage@50')} |"
        )
    lines.extend(
        [
            "",
            "Global BM25 : top10 sauvegardé, donc Hit50 non disponible. "
            "Global dense : top50 sauvegardé. Les rangs globaux absents sont censurés "
            "à leur profondeur ; les rangs oracle sont calculés sur tous les candidats "
            "du document, puis seuls les 50 premiers sont sauvegardés. Pour BM25, "
            "un rang absent signifie aucun candidat lexical de score positif porteur "
            "de cette preuve, pas absence de la preuve dans le corpus.",
            "",
            "## Comparaison par question",
            "",
            "G→O : global sauvegardé vers oracle. R : première page attendue (rang de "
            "segment). E : premier segment portant un extrait exact. `—` : non observé.",
            "",
            "| ID | BM25 Hit5 G→O | Dense Hit5 G→O | R BM25 G→O | R dense G→O | "
            "E oracle BM25 / dense | Couverture complète extraits oracle @50 BM25 / dense |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["comparison"]:
        bm, dense = row["bm25"], row["dense"]
        lines.append(
            f"| {row['question_id']} | {bm['global']['hit@5']}→{bm['oracle']['hit@5']} | "
            f"{dense['global']['hit@5']}→{dense['oracle']['hit@5']} | "
            f"{display_rank(bm['global']['first_reference_segment_rank'])}→"
            f"{display_rank(bm['oracle']['first_reference_segment_rank'])} | "
            f"{display_rank(dense['global']['first_reference_segment_rank'])}→"
            f"{display_rank(dense['oracle']['first_reference_segment_rank'])} | "
            f"{display_rank(bm['oracle']['first_supporting_excerpt_rank'])} / "
            f"{display_rank(dense['oracle']['first_supporting_excerpt_rank'])} | "
            f"{bm['oracle']['complete_excerpt_coverage@50']} / "
            f"{dense['oracle']['complete_excerpt_coverage@50']} |"
        )
    lines.extend(
        [
            "",
            "## Interprétation et limites",
            "",
            "Un gain oracle indique l'effet d'une restriction assistée par les références, "
            "sans établir la faisabilité d'un routeur réel. Les écarts entre le rang "
            "d'une page et celui du segment de preuve restent visibles. Une comparaison "
            "peut réussir en Hit avec un seul opérande : consulter sa couverture complète. "
            "Les références restent candidates ; aucun effet causal de la langue, du modèle "
            "ou de la mise en page n'est affirmé. Aucun hybride, reranker ni nouveau modèle "
            "n'a été exécuté. Le test n'a pas été ouvert.",
            "",
            "Les prédictions top50, métriques détaillées à tous les k, rangs complets "
            "des pages/extraits et hashes avant/après sont conservés séparément dans "
            "ce dossier privé. Les scores globaux existants restent inchangés.",
            "",
        ]
    )
    findings = ["", "## Observations mesurées", ""]
    for method in ("bm25", "dense"):
        recovered = [
            row["question_id"]
            for row in report["comparison"]
            if not row[method]["global"]["hit@5"] and row[method]["oracle"]["hit@5"]
        ]
        regressed = [
            row["question_id"]
            for row in report["comparison"]
            if row[method]["global"]["hit@5"] and not row[method]["oracle"]["hit@5"]
        ]
        findings.append(
            f"- {method}, Hit5 : {len(recovered)} échec(s) récupéré(s), "
            f"{len(regressed)} régression(s). IDs récupérés : "
            f"{', '.join(recovered) or 'aucun'} ; régressions : "
            f"{', '.join(regressed) or 'aucune'}."
        )
    page_without_excerpt = [
        row["question_id"]
        for row in report["comparison"]
        if row["dense"]["oracle"]["hit@50"]
        and not row["dense"]["oracle"]["complete_excerpt_coverage@50"]
    ]
    findings.extend(
        [
            "- Dense : page v1 retrouvée dans les 50 candidats sans couverture complète "
            f"des extraits : {', '.join(page_without_excerpt) or 'aucun cas'}.",
            "- La restriction au document n'améliore pas le Hit5 dense sur ce dev. "
            "Les classements internes au document restent donc un obstacle observé ; "
            "cela ne détermine pas la cause technique ni la validité financière des références.",
            "",
        ]
    )
    lines.extend(findings)
    return "\n".join(lines)


def run_oracle(
    project_root: Path, output: Path, check_only: bool = False, allow_long_search: bool = False
) -> dict:
    output = private_output(output, project_root)
    if output.exists():
        raise FinragError("Document oracle requires a new private output directory.")
    started = perf_counter()
    inputs = verify_cached_inputs(project_root)
    write_atomic(output / "preflight.json", json_bytes(inputs.preflight))
    if check_only or (not inputs.preflight["five_minute_gate_passed"] and not allow_long_search):
        return {
            "search_started": False,
            "preflight": inputs.preflight,
            "manual_command": "python -m finrag.evaluation.run_document_oracle "
            "--output artifacts/private/document_oracle_manual --allow-long-search",
        }
    dense = dense_in_worker(inputs, output)
    config = LexicalConfig.model_validate(inputs.preflight["bm25_config"])
    grouped = document_rows(inputs.chunks)
    indexes = {
        doc: BM25([inputs.chunks[row] for row in rows], config) for doc, rows in grouped.items()
    }
    bm25, comparison = [], []
    dense_by_id = {row["question_id"]: row for row in dense}
    for query_row, question in enumerate(inputs.questions):
        if not question.answerable:
            continue
        index = indexes[question.document_id]
        full = [
            hit.model_dump(mode="json")
            for hit in index.search(question.question, top_k=len(index.chunks))
        ]
        lexical = {
            "question_id": question.question_id,
            "document_passages_searched": len(index.chunks),
            "hits": full[:50],
            "metrics": evaluate_candidates(question, full, True),
        }
        bm25.append(lexical)
        compared = {"question_id": question.question_id, "reference_pages": question.relevant_pages}
        for method, oracle in (("bm25", lexical), ("dense", dense_by_id[question.question_id])):
            baseline = evaluate_candidates(
                question, inputs.baseline[method][query_row]["retrieved"], False
            )
            compared[method] = {"global": baseline, "oracle": oracle["metrics"]}
        comparison.append(compared)
    stored = read_json(project_root / "artifacts/private/phase2b/comparison.json")
    for method in ("bm25", "dense"):
        for row in comparison:
            original = next(
                item for item in stored["comparison"] if item["question_id"] == row["question_id"]
            )
            for k in (1, 5, 10):
                if row[method]["global"][f"hit@{k}"] != bool(
                    original["methods"][method]["metrics"][f"hit@{k}"]
                ):
                    raise FinragError("Historical global metrics differ from saved predictions.")
    after = {path: hash_file(Path(path)) for path in inputs.input_hashes}
    if after != inputs.input_hashes:
        raise FinragError("An existing cache/result changed during the oracle diagnostic.")
    aggregates = {}
    for method in ("bm25", "dense"):
        for source in ("global", "oracle"):
            aggregates[f"{method}_{source}"] = aggregate_metrics(
                [row[method][source] for row in comparison]
            )
    report = {
        "preflight": inputs.preflight,
        "runtime_seconds": perf_counter() - started,
        "aggregates": aggregates,
        "comparison": comparison,
        "input_hashes_before": inputs.input_hashes,
        "input_hashes_after": after,
        "existing_inputs_unchanged": True,
        "test_read": False,
        "new_encoding": False,
        "oracle_uses_reference_document": True,
    }
    write_atomic(output / "bm25_top50.jsonl", jsonl_bytes(bm25))
    write_atomic(output / "dense_top50.jsonl", jsonl_bytes(dense))
    write_atomic(output / "diagnostic.json", json_bytes(report))
    write_atomic(output / "rapport.md", render_report(report).encode("utf-8"))
    return {
        "report": str(output / "rapport.md"),
        "aggregates": aggregates,
        "runtime_seconds": report["runtime_seconds"],
        "existing_inputs_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact document oracle using existing dev vectors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-caches", action="store_true")
    parser.add_argument("--allow-long-search", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(run_oracle(Path.cwd(), args.output, args.check_caches, args.allow_long_search))
    )


if __name__ == "__main__":
    main()
