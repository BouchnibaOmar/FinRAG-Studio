import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finrag.config import APPROVED_EVALUATION_ASSETS, FinragError, private_output
from finrag.domain.evaluation import EvaluationQuestion, EvidencePageMapping
from finrag.domain.manifests import DatasetCounts, DatasetManifest, Transformation
from finrag.evaluation.validate_dataset import (
    parse_questions,
    read_corpus_documents,
    validate_bundle,
    validate_dataset,
    validate_record,
)
from finrag.ingestion.hashing import hash_bytes, hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes, publish_directory
from finrag.ingestion.pdf_loader import extract_page_texts

WORDING_ID = "finrag_v1_bnp_05"
ORIGINAL_WORDING = (
    "Quelle acquisition a renforcé BNP Paribas Asset Management en 2025 "
    "selon la présentation du pôle IPS ?"
)
IMPORTED_WORDING = ORIGINAL_WORDING.replace("Quelle acquisition", "Quelle intégration", 1)

LIMITATIONS = [
    "45 cases were compared against the 19 supplied evidence pages; "
    "55 excerpts matched mapped pages.",
    "This comparison does not constitute manual verification by Omar Bouchniba.",
    "Financial correctness beyond the supplied pages is not independently certified.",
    "The three unanswerable cases were accepted through corpus-period reasoning, "
    "not exhaustive manual review of every report page.",
    "Original row statuses remain 44 candidate and 1 verified; "
    "added financial context remains candidate.",
    "Legacy chunk IDs refer to prototype chunks, not Phase 1 character windows; "
    "only document/page references are checked here.",
    "Development and test splits share documents and some facts; this is not a document holdout.",
    "Source reports and evidence are private local research material "
    "with no redistribution permission established.",
]


def decode_json(content: bytes, name: str) -> Any:
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FinragError(f"{name}: invalid UTF-8 JSON.") from error


def normalize_snapshot(rows: Any) -> list[EvaluationQuestion]:
    if not isinstance(rows, list):
        raise FinragError("Annotations snapshot must be a JSON array.")
    legacy_fields = {
        "evidence_texts",
        "annotator_note",
        "document_title",
        "source_url",
        "source_sha256",
        "context_note",
    }
    comparison_fields = {
        "operand_1",
        "operand_2",
        "operation",
        "calculated_result",
        "rounding_rule",
    }
    unanswerable_fields = {
        "corpus_years",
        "requested_year",
        "possible_confusion",
        "reason_unanswerable",
    }
    optional_fields = comparison_fields | unanswerable_fields
    native_fields = set(EvaluationQuestion.model_fields) - {"evidence"} - optional_fields
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise FinragError("Snapshot question must be an object.")
        required = native_fields | legacy_fields
        if row.get("category") == "comparative":
            required |= comparison_fields
        if row.get("answerable") is False:
            required |= unanswerable_fields
        if set(row) != required:
            raise FinragError("Snapshot fields differ from the approved compatibility contract.")
        if row["document_id"] != f"sha256:{row['source_sha256']}":
            raise FinragError("Annotation source hash does not match document ID.")
        value = {name: row[name] for name in native_fields}
        value.update({name: row[name] for name in optional_fields if name in row})
        evidence = []
        if not isinstance(row["evidence_texts"], list):
            raise FinragError("Snapshot evidence must be an array.")
        for item in row["evidence_texts"]:
            if not isinstance(item, dict) or set(item) != {"page_number", "text"}:
                raise FinragError("Snapshot evidence fields are invalid.")
            evidence.append({"physical_page_number": item["page_number"], "text": item["text"]})
        value["evidence"] = evidence
        normalized.append(value)
    return parse_questions(normalized)


def transform_questions(
    questions: list[EvaluationQuestion],
) -> tuple[list[EvaluationQuestion], list[Transformation]]:
    matches = [question for question in questions if question.question_id == WORDING_ID]
    if len(matches) != 1 or matches[0].question != ORIGINAL_WORDING:
        raise FinragError(
            "BNP wording transformation requires exactly one unchanged source question."
        )
    transformed = [
        EvaluationQuestion.model_validate({**question.model_dump(), "question": IMPORTED_WORDING})
        if question.question_id == WORDING_ID
        else question
        for question in questions
    ]
    history = [
        Transformation(
            question_id=WORDING_ID,
            original_value=ORIGINAL_WORDING,
            new_value=IMPORTED_WORDING,
            reason=(
                "Use integration wording to match the supplied IPS evidence "
                "without asserting acquisition timing."
            ),
            version="bnp-wording-v1",
        )
    ]
    return transformed, history


def source_paths(
    annotations: Path,
    mapping: Path,
    evidence_pdf: Path,
    corpus_manifest: Path | None = None,
    frozen: Path | None = None,
    baseline_config: Path | None = None,
    baseline_metrics: Path | None = None,
) -> dict[str, Path]:
    evaluation_root = annotations.parent.parent
    return {
        "annotations_for_review.json": annotations,
        "evidence_page_mapping.json": mapping,
        "evidence_pages.pdf": evidence_pdf,
        "corpus_manifest.json": corpus_manifest or evaluation_root / "corpus/manifest.json",
        "finrag_v1.frozen.json": frozen or evaluation_root / "datasets/finrag_v1.frozen.json",
        "baseline.json": baseline_config or evaluation_root / "configs/baseline.json",
        "retrieval_baseline_metrics.json": baseline_metrics
        or evaluation_root / "results/retrieval_baseline_metrics.json",
    }


def validate_source_hashes(contents: dict[str, bytes]) -> None:
    if set(contents) != set(APPROVED_EVALUATION_ASSETS):
        raise FinragError("Import requires exactly the seven approved evaluation assets.")
    corpus = decode_json(contents["corpus_manifest.json"], "Corpus manifest")
    frozen = decode_json(contents["finrag_v1.frozen.json"], "Frozen manifest")
    config = decode_json(contents["baseline.json"], "Baseline configuration")
    metrics = decode_json(contents["retrieval_baseline_metrics.json"], "Baseline metrics")
    documents = read_corpus_documents(corpus)
    if not all(isinstance(value, dict) for value in (frozen, config, metrics)):
        raise FinragError("Historical configuration and results must be JSON objects.")
    split_ids = frozen.get("question_ids_by_split")
    if (
        not isinstance(split_ids, dict)
        or set(split_ids) != {"dev", "test"}
        or any(
            not isinstance(identifiers, list)
            or not all(isinstance(identifier, str) for identifier in identifiers)
            or len(identifiers) != len(set(identifiers))
            for identifiers in split_ids.values()
        )
    ):
        raise FinragError("Frozen split IDs must be unique string arrays for dev and test.")
    expected_hashes = {
        "corpus_manifest_sha256": hash_bytes(contents["corpus_manifest.json"]),
        "configuration_sha256": hash_bytes(contents["baseline.json"]),
    }
    if any(frozen.get(name) != digest for name, digest in expected_hashes.items()):
        raise FinragError("Historical frozen inputs do not match corpus/configuration hashes.")
    if (
        frozen.get("document_sha256")
        != {key: value.source_sha256 for key, value in documents.items()}
        or frozen.get("configuration") != config
        or frozen.get("question_count") != 45
        or frozen.get("chunk_count") != 3760
        or frozen.get("split_counts") != {"dev": 30, "test": 15}
        or frozen.get("source_annotation_status_counts") != {"candidate": 44, "verified": 1}
    ):
        raise FinragError("Historical frozen counts, documents or configuration are inconsistent.")
    if metrics.get("configuration_sha256") != expected_hashes[
        "configuration_sha256"
    ] or metrics.get("frozen_manifest_sha256") != hash_bytes(contents["finrag_v1.frozen.json"]):
        raise FinragError("Historical metrics do not match the frozen inputs.")


def historical_baseline_reference(contents: dict[str, bytes]) -> dict[str, Any]:
    validate_source_hashes(contents)
    config = decode_json(contents["baseline.json"], "Baseline configuration")
    metrics = decode_json(contents["retrieval_baseline_metrics.json"], "Baseline metrics")
    selected = {}
    for name, group in (
        ("overall", metrics["groups"]["overall"]["all"]),
        ("dev", metrics["groups"]["split"]["dev"]),
        ("test", metrics["groups"]["split"]["test"]),
    ):
        values = group["metrics_answerable_only"]
        selected[name] = {
            "questions": group["questions"],
            "answerable": group["answerable"],
            "unanswerable": group["unanswerable"],
            "metrics": {
                field: values[field]
                for field in (
                    "recall@1",
                    "recall@3",
                    "recall@5",
                    "mrr",
                    "document_accuracy",
                )
            },
        }
    return {
        "label": "historical prototype result",
        "prototype": "DocsGPT-Custom",
        "prototype_commit": "a6002cdfe0e5ecb9dd3cac5a32a349c5bf621eb3",
        "statements": [
            "Not reproduced by FinRAG-Studio.",
            "Not evidence of answer-generation quality.",
        ],
        "corpus_reports": 3,
        "chunks": 3760,
        "embedding_model": config["embedding_model"],
        "embedding_revision": config["embedding_revision"],
        "index": config["index"],
        "normalized_embeddings": True,
        "chunk_limit": config["chunking"]["max_tokens"],
        "chunk_unit": "cl100k_base tokens",
        "overlap": config["chunking"]["overlap_tokens"],
        "model_token_limit": config["model_max_seq_length"],
        "configuration_mismatch": (
            "1250 cl100k_base tokens per chunk versus MPNet's 384 model-token input limit."
        ),
        "groups": selected,
        "source_hashes": {
            name: hash_bytes(contents[name])
            for name in (
                "baseline.json",
                "finrag_v1.frozen.json",
                "retrieval_baseline_metrics.json",
            )
        },
        "prototype_test_count": {
            "static_methods": 87,
            "earlier_recorded_passes": 80,
            "stored_87_pass_record_found": False,
            "execution_count_status": "unresolved",
        },
    }


def validate_snapshot(contents: dict[str, bytes]) -> tuple[list[EvaluationQuestion], DatasetCounts]:
    validate_source_hashes(contents)
    questions = normalize_snapshot(
        decode_json(contents["annotations_for_review.json"], "Annotations")
    )
    rows = decode_json(contents["evidence_page_mapping.json"], "Evidence mapping")
    if not isinstance(rows, list):
        raise FinragError("Evidence mapping must be a JSON array.")
    mapping = [validate_record(EvidencePageMapping, row, "Evidence mapping") for row in rows]
    corpus = read_corpus_documents(decode_json(contents["corpus_manifest.json"], "Corpus manifest"))
    texts = extract_page_texts(contents["evidence_pages.pdf"], "evidence_pages.pdf")
    counts = validate_bundle(questions, mapping, corpus, texts)
    frozen = decode_json(contents["finrag_v1.frozen.json"], "Frozen manifest")
    for split in ("dev", "test"):
        identifiers = {question.question_id for question in questions if question.split == split}
        if identifiers != set(frozen.get("question_ids_by_split", {}).get(split, [])):
            raise FinragError("Annotation split IDs differ from the historical frozen snapshot.")
    return questions, counts


def import_evaluation(
    paths: dict[str, Path],
    output: Path,
    project_root: Path,
    overwrite: bool = False,
) -> DatasetManifest:
    if set(paths) != set(APPROVED_EVALUATION_ASSETS):
        raise FinragError("Import requires exactly the seven approved evaluation assets.")
    output = private_output(output, project_root)
    if output.exists() and not overwrite:
        raise FinragError("Dataset output exists; use --overwrite explicitly.")
    if output.exists() and not output.is_dir():
        raise FinragError("Dataset output must be a directory.")
    if any(path.resolve().is_relative_to(output) for path in paths.values()):
        raise FinragError("Import sources must be outside the destination dataset.")
    contents = {name: path.read_bytes() for name, path in paths.items()}
    questions, counts = validate_snapshot(contents)
    transformed, history = transform_questions(questions)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".finrag-{output.name}.lock.tmp"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise FinragError("Another import holds the destination lock.") from error
    os.close(descriptor)
    try:
        with tempfile.TemporaryDirectory(prefix=".finrag-", dir=output.parent) as temporary:
            staged = Path(temporary) / "dataset"
            staged.mkdir()
            manifest = _write_snapshot(staged, paths, contents, transformed, counts, history)
            validate_dataset(staged)
            if any(hash_file(path) != manifest.source_hashes[name] for name, path in paths.items()):
                raise FinragError(
                    "Source assets changed during import; destination was not published."
                )
            publish_directory(staged, output, overwrite)
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def _write_snapshot(
    staged: Path,
    paths: dict[str, Path],
    contents: dict[str, bytes],
    questions: list[EvaluationQuestion],
    counts: DatasetCounts,
    history: list[Transformation],
) -> DatasetManifest:
    sources = staged / "sources"
    sources.mkdir()
    for name, content in contents.items():
        (sources / name).write_bytes(content)
    for split in ("dev", "test"):
        rows = [
            question.model_dump(mode="json")
            for question in sorted(questions, key=lambda row: row.question_id)
            if question.split == split
        ]
        (staged / f"{split}.jsonl").write_bytes(jsonl_bytes(rows))
    hashes = {
        path.relative_to(staged).as_posix(): hash_file(path)
        for path in staged.rglob("*")
        if path.is_file()
    }
    manifest = DatasetManifest(
        dataset_name="finrag",
        dataset_version="finrag_v1",
        schema_version="1.0",
        creation_timestamp=datetime.now(UTC),
        review_status="externally_reviewed_ai_assisted",
        counts=counts,
        source_artifacts={name: path.name for name, path in paths.items()},
        source_hashes={name: hash_bytes(content) for name, content in contents.items()},
        imported_hashes=hashes,
        transformation_history=history,
        limitations=LIMITATIONS,
        prototype_baseline_reference="artifacts/prototype_baseline_reference.json",
    )
    (staged / "dataset_manifest.json").write_bytes(json_bytes(manifest.model_dump(mode="json")))
    return manifest
