from collections import Counter
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from finrag.config import APPROVED_EVALUATION_ASSETS, FinragError
from finrag.domain.documents import PositiveInt, SourceIdentity, StrictRecord
from finrag.domain.evaluation import EvaluationQuestion, EvidencePageMapping
from finrag.domain.manifests import DatasetCounts, DatasetManifest
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import read_json, read_jsonl
from finrag.ingestion.pdf_loader import extract_page_texts


class CorpusDocument(SourceIdentity):
    company: str
    physical_page_count: PositiveInt


EXPECTED_COUNTS = DatasetCounts(
    questions=45,
    dev=30,
    test=15,
    answerable=42,
    unanswerable=3,
    candidate=44,
    verified=1,
    evidence_pages=19,
    evidence_excerpts=55,
)


Record = TypeVar("Record", bound=StrictRecord)


def validate_record(model: type[Record], value: Any, description: str) -> Record:
    try:
        return model.model_validate(value)
    except ValidationError as error:
        location = ".".join(str(part) for part in error.errors()[0]["loc"])
        # Pydantic's full error includes the input; private evidence must not reach CLI logs.
        raise FinragError(f"{description}: invalid field {location or 'record'}.") from error


def read_corpus_documents(manifest: dict[str, Any]) -> dict[str, CorpusDocument]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("documents"), list):
        raise FinragError("Corpus manifest must declare its documents.")
    documents = {}
    for row in manifest["documents"]:
        if not isinstance(row, dict):
            raise FinragError("Corpus manifest document must be an object.")
        value = {
            "document_id": row.get("document_id"),
            "source_filename": row.get("local_filename"),
            "source_sha256": row.get("sha256"),
            "company": row.get("company"),
            "physical_page_count": row.get("page_count"),
        }
        document = validate_record(CorpusDocument, value, "Corpus document")
        if document.document_id in documents:
            raise FinragError("Corpus manifest contains duplicate document IDs.")
        documents[document.document_id] = document
    if len(documents) != 3:
        raise FinragError("finrag_v1 requires exactly three known source documents.")
    return documents


def parse_questions(rows: list[dict[str, Any]]) -> list[EvaluationQuestion]:
    return [validate_record(EvaluationQuestion, row, "Evaluation question") for row in rows]


def validate_counts(questions: list[EvaluationQuestion], page_count: int) -> DatasetCounts:
    ids = [question.question_id for question in questions]
    if len(ids) != len(set(ids)):
        raise FinragError("Duplicate question IDs across development and test splits.")
    splits = Counter(question.split for question in questions)
    statuses = Counter(question.annotation_status for question in questions)
    answerable = sum(question.answerable for question in questions)
    counts = DatasetCounts(
        questions=len(questions),
        dev=splits["dev"],
        test=splits["test"],
        answerable=answerable,
        unanswerable=len(questions) - answerable,
        candidate=statuses["candidate"],
        verified=statuses["verified"],
        evidence_pages=page_count,
        evidence_excerpts=sum(len(question.evidence) for question in questions),
    )
    if counts != EXPECTED_COUNTS:
        raise FinragError(
            "finrag_v1 counts must be 45 questions, 30 dev/15 test, 42 answerable/3 unanswerable, "
            "44 candidate/1 verified, 19 bundle pages and 55 evidence excerpts."
        )
    return counts


def validate_bundle(
    questions: list[EvaluationQuestion],
    mapping: list[EvidencePageMapping],
    documents: dict[str, CorpusDocument],
    page_texts: list[str],
) -> DatasetCounts:
    counts = validate_counts(questions, len(page_texts))
    if any(question.document_id not in documents for question in questions):
        raise FinragError("Evaluation question references an unknown document.")
    by_question = {question.question_id: question for question in questions}
    by_page = {}
    source_pages = set()
    for item in mapping:
        document = documents.get(item.document_id)
        if document is None:
            raise FinragError("Evidence mapping references an unknown document.")
        if (
            item.source_sha256 != document.source_sha256
            or item.source_filename != document.source_filename
            or item.company != document.company
            or item.original_physical_page > document.physical_page_count
        ):
            raise FinragError("Evidence mapping disagrees with corpus source provenance.")
        if not 1 <= item.bundle_page_number <= len(page_texts):
            raise FinragError("Evidence mapping bundle page is outside the PDF.")
        key = (item.document_id, item.original_physical_page)
        if item.bundle_page_number in by_page or key in source_pages:
            raise FinragError("Evidence mapping contains duplicate bundle or source pages.")
        if not item.question_ids or len(item.question_ids) != len(set(item.question_ids)):
            raise FinragError("Evidence mapping requires unique question IDs for each page.")
        for identifier in item.question_ids:
            question = by_question.get(identifier)
            if question is None:
                raise FinragError("Evidence mapping contains an unknown question ID.")
            if (
                not question.answerable
                or question.document_id != item.document_id
                or item.original_physical_page not in question.relevant_pages
                or item.bundle_page_number not in question.evidence_bundle_pages
            ):
                raise FinragError("Evidence mapping does not agree with the mapped question.")
        by_page[item.bundle_page_number] = item
        source_pages.add(key)
    if set(by_page) != set(range(1, len(page_texts) + 1)):
        raise FinragError("Evidence mapping must cover every bundle page exactly once.")
    for question in questions:
        _validate_question_provenance(question, documents, by_page, page_texts)
    return counts


def _validate_question_provenance(
    question: EvaluationQuestion,
    documents: dict[str, CorpusDocument],
    by_page: dict[int, EvidencePageMapping],
    page_texts: list[str],
) -> None:
    document = documents.get(question.document_id)
    if document is None:
        raise FinragError("Evaluation question references an unknown document.")
    if question.company != document.company:
        raise FinragError("Question company does not match its document.")
    mapped = {}
    for bundle_page in question.evidence_bundle_pages:
        item = by_page.get(bundle_page)
        if item is None:
            raise FinragError("Question bundle page is outside the mapped PDF.")
        if (
            item.document_id != question.document_id
            or question.question_id not in item.question_ids
        ):
            raise FinragError("Question and evidence mapping must reference each other.")
        mapped[item.original_physical_page] = bundle_page
    if set(mapped) != set(question.relevant_pages):
        raise FinragError("Question bundle pages do not cover its relevant physical pages.")
    for evidence in question.evidence:
        if evidence.text not in page_texts[mapped[evidence.physical_page_number] - 1]:
            raise FinragError("Evidence excerpt does not match its mapped PDF page exactly.")
    for identifier in question.relevant_chunk_ids:
        prefix = f"{question.document_id}:p"
        if not identifier.startswith(prefix):
            raise FinragError("Legacy chunk reference has a different source document.")
        try:
            page_number = int(identifier[len(prefix) :].split(":", 1)[0])
        except ValueError as error:
            raise FinragError("Legacy chunk reference has an invalid physical page.") from error
        if page_number not in question.relevant_pages:
            raise FinragError("Legacy chunk reference is outside the relevant pages.")
    for value in question.source_values:
        if (
            value.page_number not in mapped
            or value.evidence_text not in page_texts[mapped[value.page_number] - 1]
        ):
            raise FinragError("Financial source value is not supported by its mapped page.")


def dataset_file(root: Path, name: str) -> Path:
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise FinragError("Dataset manifest contains an unsafe artifact path.")
    if path.is_symlink() or not path.is_file():
        raise FinragError("Dataset artifact must be an existing regular file.")
    return path


def validate_dataset(root: Path) -> DatasetManifest:
    manifest_path = root / "dataset_manifest.json"
    try:
        manifest = DatasetManifest.model_validate_json(manifest_path.read_bytes())
    except ValidationError as error:
        raise FinragError("Dataset manifest does not satisfy the FinRAG contract.") from error
    required_sources = set(APPROVED_EVALUATION_ASSETS)
    if (
        set(manifest.source_artifacts) != required_sources
        or set(manifest.source_hashes) != required_sources
    ):
        raise FinragError("Dataset manifest must identify exactly the approved source artifacts.")
    required_outputs = {
        *(f"sources/{name}" for name in required_sources),
        "dev.jsonl",
        "test.jsonl",
    }
    if set(manifest.imported_hashes) != required_outputs:
        raise FinragError(
            "Dataset manifest must cover all source copies and transformed split files."
        )
    for name, digest in manifest.imported_hashes.items():
        if hash_file(dataset_file(root, name)) != digest:
            raise FinragError(f"{Path(name).name}: imported artifact hash changed.")
    for name, digest in manifest.source_hashes.items():
        if manifest.imported_hashes[f"sources/{name}"] != digest:
            raise FinragError("Source copy hash does not preserve original bytes.")
    from finrag.evaluation.import_assets import (
        LIMITATIONS,
        transform_questions,
        validate_snapshot,
    )

    sources = root / "sources"
    contents = {name: dataset_file(sources, name).read_bytes() for name in required_sources}
    source_questions, _ = validate_snapshot(contents)
    expected_questions, history = transform_questions(source_questions)
    questions = []
    for split in ("dev", "test"):
        split_questions = parse_questions(read_jsonl(root / f"{split}.jsonl"))
        if any(question.split != split for question in split_questions):
            raise FinragError("A split file contains a question from the other split.")
        questions.extend(split_questions)
    if sorted(questions, key=lambda row: row.question_id) != sorted(
        expected_questions, key=lambda row: row.question_id
    ):
        raise FinragError("Imported questions differ from the declared source transformation.")
    if manifest.transformation_history != history:
        raise FinragError("Transformation history does not match the imported wording change.")
    if manifest.limitations != LIMITATIONS:
        raise FinragError("Dataset manifest must retain the declared review limitations.")
    mapping = [
        validate_record(EvidencePageMapping, row, "Evidence mapping")
        for row in read_json(sources / "evidence_page_mapping.json")
    ]
    documents = read_corpus_documents(read_json(sources / "corpus_manifest.json"))
    texts = extract_page_texts(contents["evidence_pages.pdf"], "evidence_pages.pdf")
    counts = validate_bundle(questions, mapping, documents, texts)
    if manifest.counts != counts:
        raise FinragError("Dataset manifest counts differ from validated data.")
    return manifest
