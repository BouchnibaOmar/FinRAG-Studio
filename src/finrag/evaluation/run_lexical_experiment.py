import tempfile
from pathlib import Path
from time import perf_counter

from finrag.config import FinragError, private_output
from finrag.corpus.import_corpus import validate_corpus
from finrag.corpus.prepare_corpus import extraction_directory, load_chunk_cache, validate_extraction
from finrag.domain.chunks import ChunkRecord
from finrag.domain.evaluation import EvaluationQuestion
from finrag.domain.retrieval import LexicalConfig, LexicalPrediction, PageTarget
from finrag.evaluation.retrieval_metrics import grouped_metrics, question_metrics
from finrag.evaluation.split_guard import load_dev_questions
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import (
    json_bytes,
    jsonl_bytes,
    publish_directory,
    read_json,
    read_jsonl,
    write_atomic,
)
from finrag.retrieval.bm25 import BM25

SELECTION_PRIORITY = [
    "highest_answerable_hit@5",
    "highest_mrr@10",
    "highest_hit@1",
    "fewest_chunks",
    "smallest_overlap",
]


def load_config(path: Path) -> LexicalConfig:
    return LexicalConfig.model_validate_json(path.read_bytes())


def validate_selection_policy(path: Path) -> dict:
    policy = read_json(path)
    if (
        policy.get("phase") != "2A"
        or policy.get("evaluation_split") != "dev"
        or policy.get("selection_priority") != SELECTION_PRIORITY
        or policy.get("historical_test_metrics_used_for_selection") is not False
    ):
        raise FinragError("Selection policy differs from the predeclared dev-only rule.")
    return policy


def validate_predictions(
    questions: list[EvaluationQuestion],
    predictions: list[LexicalPrediction],
    chunks: list[ChunkRecord],
    config: LexicalConfig,
) -> None:
    known = {chunk.chunk_id: chunk for chunk in chunks}
    by_question = {question.question_id: question for question in questions}
    if len(predictions) != len(by_question) or {row.question_id for row in predictions} != set(
        by_question
    ):
        raise FinragError("Prediction IDs differ from the development input.")
    for prediction in predictions:
        question = by_question[prediction.question_id]
        expected = [
            PageTarget(document_id=question.document_id, physical_page_number=page)
            for page in question.relevant_pages
        ]
        if (
            prediction.split != "dev"
            or prediction.configuration_id != config.configuration_id
            or prediction.expected_pages != expected
            or prediction.answerable != question.answerable
            or len(prediction.retrieved) > config.top_k
        ):
            raise FinragError("Prediction split, configuration or expected pages are invalid.")
        previous = None
        identifiers = set()
        for rank, hit in enumerate(prediction.retrieved, start=1):
            chunk = known.get(hit.chunk_id)
            if (
                chunk is None
                or hit.rank != rank
                or hit.configuration_id != config.configuration_id
                or hit.chunk_id in identifiers
                or hit.lexical_score <= 0
                or hit.document_id != chunk.document_id
                or hit.source_filename != chunk.source_filename
                or hit.source_sha256 != chunk.source_sha256
                or hit.physical_page_number != chunk.physical_page_number
                or hit.text != chunk.text
            ):
                raise FinragError("Retrieved score, rank or chunk provenance is invalid.")
            order = (-hit.lexical_score, hit.chunk_id)
            if previous is not None and order < previous:
                raise FinragError(
                    "Predictions are not ordered by score and deterministic chunk ID."
                )
            previous = order
            identifiers.add(hit.chunk_id)


def failure_analysis(
    questions: list[EvaluationQuestion],
    predictions: list[LexicalPrediction],
    page_texts: dict[tuple[str, int], str],
) -> list[dict]:
    by_id = {prediction.question_id: prediction for prediction in predictions}
    failures = []
    for question in questions:
        prediction = by_id[question.question_id]
        metrics = question_metrics(question, prediction)
        if metrics is None or metrics["hit@5"]:
            continue
        hits = prediction.retrieved[:5]
        missing = [
            page
            for page in question.relevant_pages
            if not page_texts.get((question.document_id, page), "").strip()
        ]
        if missing:
            category = "page text missing"
            evidence = {"missing_physical_pages": missing}
        elif not any(hit.document_id == question.document_id for hit in hits):
            category = "expected document not retrieved"
            evidence = {"expected_document_present_in_top5": False}
        else:
            category = "expected document retrieved but wrong page"
            evidence = {
                "expected_document_present_in_top5": True,
                "expected_page_present_in_top5": False,
            }
        failures.append(
            {
                "question_id": question.question_id,
                "category": category,
                "support": evidence,
                "expected_pages": [target.model_dump() for target in prediction.expected_pages],
                "retrieved_top5": [
                    {
                        "rank": hit.rank,
                        "document_id": hit.document_id,
                        "physical_page_number": hit.physical_page_number,
                    }
                    for hit in hits
                ],
                "technical_cause": "unknown; category describes observed retrieval evidence",
                "annotation_status": question.annotation_status,
            }
        )
    return failures


def experiment_inputs(
    corpus_manifest: Path,
    extraction_root: Path,
    chunk_root: Path,
    config: LexicalConfig,
) -> tuple[list[ChunkRecord], dict, dict]:
    corpus = validate_corpus(corpus_manifest)
    chunks, page_texts, fingerprints = [], {}, {}
    for report in corpus.reports:
        extracted = extraction_directory(extraction_root, report)
        _, pages = validate_extraction(extracted, report)
        directory = extraction_directory(chunk_root / config.configuration_id, report)
        local, _ = load_chunk_cache(directory, pages, hash_file(extracted / "pages.jsonl"), config)
        chunks.extend(local)
        page_texts.update(
            {(page.document_id, page.physical_page_number): page.text for page in pages}
        )
        fingerprints[report.document_id] = {
            "pdf_sha256": report.source_sha256,
            "pages_sha256": hash_file(extracted / "pages.jsonl"),
            "chunks_sha256": hash_file(directory / "chunks.jsonl"),
            "chunk_manifest_sha256": hash_file(directory / "chunk_manifest.json"),
            "chunk_count": len(local),
        }
    return chunks, page_texts, fingerprints


def run_experiment(
    config_path: Path,
    policy_path: Path,
    dataset: Path,
    corpus_manifest: Path,
    extraction_root: Path,
    chunk_root: Path,
    output_root: Path,
    project_root: Path,
    split: str = "dev",
    allow_test: bool = False,
    evaluation_reason: str | None = None,
) -> dict:
    started = perf_counter()
    questions = load_dev_questions(dataset, split, allow_test, evaluation_reason)
    config = load_config(config_path)
    validate_selection_policy(policy_path)
    output = private_output(output_root, project_root) / config.configuration_id
    chunks, page_texts, fingerprints = experiment_inputs(
        corpus_manifest, extraction_root, chunk_root, config
    )
    inputs = {
        "bm25_version": "positive-idf-v1",
        "configuration_sha256": hash_file(config_path),
        "selection_policy_sha256": hash_file(policy_path),
        "dev_sha256": hash_file(dataset / "dev.jsonl"),
        "corpus_manifest_sha256": hash_file(corpus_manifest),
        "corpus": fingerprints,
    }
    if output.exists():
        return validate_experiment(output, questions, chunks, config, inputs, page_texts)
    retriever = BM25(chunks, config)
    predictions = [
        LexicalPrediction(
            question_id=question.question_id,
            split="dev",
            configuration_id=config.configuration_id,
            answerable=question.answerable,
            expected_pages=[
                PageTarget(document_id=question.document_id, physical_page_number=page)
                for page in question.relevant_pages
            ],
            retrieved=retriever.search(question.question, config.top_k),
        )
        for question in questions
    ]
    validate_predictions(questions, predictions, chunks, config)
    failures = failure_analysis(questions, predictions, page_texts)
    metrics = grouped_metrics(questions, predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finrag-", dir=output.parent) as temporary:
        staged = Path(temporary) / "experiment"
        staged.mkdir()
        (staged / "predictions.jsonl").write_bytes(
            jsonl_bytes([row.model_dump(mode="json") for row in predictions])
        )
        result = {
            "phase": "2A",
            "split": "dev",
            "configuration": config.model_dump(mode="json"),
            "configuration_selection": "dev-only",
            "test_evaluated": False,
            "inputs": inputs,
            "chunk_count": len(chunks),
            "groups": metrics,
            "failures": failures,
            "unanswerable_question_ids": [
                question.question_id for question in questions if not question.answerable
            ],
            "unanswerable_policy": (
                "Private retrieval outputs only; no correctness or refusal score."
            ),
            "predictions_sha256": hash_file(staged / "predictions.jsonl"),
            "runtime_seconds": perf_counter() - started,
        }
        (staged / "experiment_manifest.json").write_bytes(json_bytes(result))
        validate_experiment(staged, questions, chunks, config, inputs, page_texts)
        publish_directory(staged, output, False)
    return result


def validate_experiment(
    output: Path,
    questions: list[EvaluationQuestion],
    chunks: list[ChunkRecord],
    config: LexicalConfig,
    inputs: dict,
    page_texts: dict[tuple[str, int], str],
) -> dict:
    result = read_json(output / "experiment_manifest.json")
    if (
        result.get("phase") != "2A"
        or result.get("split") != "dev"
        or result.get("test_evaluated") is not False
        or result.get("configuration") != config.model_dump(mode="json")
        or result.get("inputs") != inputs
        or result.get("chunk_count") != len(chunks)
        or result.get("predictions_sha256") != hash_file(output / "predictions.jsonl")
    ):
        raise FinragError("Existing experiment inputs or private prediction hashes have changed.")
    predictions = [
        LexicalPrediction.model_validate(row) for row in read_jsonl(output / "predictions.jsonl")
    ]
    validate_predictions(questions, predictions, chunks, config)
    if result.get("groups") != grouped_metrics(questions, predictions):
        raise FinragError("Stored experiment metrics differ from prediction-derived metrics.")
    if result.get("failures") != failure_analysis(questions, predictions, page_texts):
        raise FinragError("Failure report differs from the available retrieval and page evidence.")
    if result.get("configuration_selection") != "dev-only" or result.get(
        "unanswerable_question_ids"
    ) != [question.question_id for question in questions if not question.answerable]:
        raise FinragError("Experiment selection or unanswerable diagnostics are inconsistent.")
    return result


def select_configuration(results: list[dict]) -> dict:
    if not results or any(
        result.get("split") != "dev" or result.get("test_evaluated") is not False
        for result in results
    ):
        raise FinragError("Configuration selection accepts development results only.")

    def priority(result: dict) -> tuple:
        metrics = result["groups"]["overall"]["metrics"]
        return (
            -metrics["hit@5"],
            -metrics["mrr@10"],
            -metrics["hit@1"],
            result["chunk_count"],
            result["configuration"]["chunking"]["overlap"],
            result["configuration"]["configuration_id"],
        )

    selected = min(results, key=priority)
    return {
        "selected_configuration": selected["configuration"]["configuration_id"],
        "evaluation_split": "dev",
        "selection_priority": SELECTION_PRIORITY,
        "selected_metrics": selected["groups"]["overall"]["metrics"],
    }


def public_report(results: list[dict], selection: dict) -> dict:
    # Whitelist aggregate fields; private prediction text never enters publication.
    return {
        "phase": "2A",
        "split": "dev",
        "test_metrics_generated": False,
        "test_policy": "Previously reported by prototype; frozen and excluded from Phase 2A.",
        "selection": selection,
        "configurations": [
            {
                "configuration": result["configuration"],
                "chunk_count": result["chunk_count"],
                "groups": result["groups"],
                "failures": result["failures"],
                "unanswerable_question_ids": result["unanswerable_question_ids"],
                "unanswerable_policy": result["unanswerable_policy"],
                "runtime_seconds": result["runtime_seconds"],
            }
            for result in results
        ],
        "limitations": [
            "Assisted annotations retain original statuses.",
            "No answer-generation or refusal-quality assessment.",
            "Prototype MPNet results are historical external references, not equivalent runs.",
        ],
    }


def publish_dev_report(results: list[dict], output: Path) -> dict:
    selection = select_configuration(results)
    write_atomic(output, json_bytes(public_report(results, selection)))
    return selection
