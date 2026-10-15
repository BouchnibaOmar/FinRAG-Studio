import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from finrag.config import FinragError
from finrag.domain.documents import PageRecord
from finrag.domain.evaluation import EvaluationQuestion
from finrag.evaluation.import_assets import decode_json, normalize_snapshot, validate_snapshot
from finrag.ingestion.manifest import json_bytes


def snapshot_contents(paths: dict[str, Path]) -> dict[str, bytes]:
    return {name: path.read_bytes() for name, path in paths.items()}


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("duplicate", "Duplicate"),
        ("count", "counts"),
        ("split", "counts"),
        ("status", "counts"),
        ("unknown_document", "unknown document"),
        ("hash", "source hash"),
        ("no_evidence", "invalid field"),
        ("positive_unanswerable", "invalid field"),
        ("extra_field", "compatibility"),
        ("private_identity", "compatibility"),
        ("wording", "unchanged source"),
    ],
)
def test_annotation_rejections(snapshot: dict[str, Path], mutation: str, expected: str) -> None:
    from finrag.evaluation.import_assets import transform_questions

    contents = snapshot_contents(snapshot)
    rows = decode_json(contents["annotations_for_review.json"], "annotations")
    if mutation == "duplicate":
        rows[1]["question_id"] = rows[0]["question_id"]
    elif mutation == "count":
        rows.pop()
    elif mutation == "split":
        rows[0]["split"] = "test"
    elif mutation == "status":
        rows[0]["annotation_status"] = "verified"
    elif mutation == "unknown_document":
        rows[0]["document_id"] = "sha256:" + "a" * 64
        rows[0]["source_sha256"] = "a" * 64
    elif mutation == "hash":
        rows[0]["source_sha256"] = "a" * 64
    elif mutation == "no_evidence":
        rows[0]["evidence_texts"] = []
    elif mutation == "positive_unanswerable":
        absent = next(row for row in rows if not row["answerable"])
        absent["evidence_texts"] = [{"page_number": 1, "text": "invented evidence"}]
    elif mutation == "extra_field":
        rows[0]["surprise"] = True
    elif mutation == "private_identity":
        rows[0]["human_review"] = {"reviewer": "Private identity"}
    else:
        rows[4]["question"] = "Unexpected wording"
    contents["annotations_for_review.json"] = json_bytes(rows)
    with pytest.raises(FinragError, match=expected):
        questions, _ = validate_snapshot(contents)
        transform_questions(questions)


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("unknown_id", "unknown question"),
        ("wrong_document", "unknown document"),
        ("bundle_outside", "outside"),
        ("duplicate_page", "duplicate"),
        ("missing_page", "cover every"),
        ("wrong_hash", "invalid field"),
        ("wrong_company", "provenance"),
        ("wrong_physical", "provenance"),
        ("missing_question", "each other"),
    ],
)
def test_mapping_rejections(snapshot: dict[str, Path], mutation: str, expected: str) -> None:
    contents = snapshot_contents(snapshot)
    mapping = decode_json(contents["evidence_page_mapping.json"], "mapping")
    if mutation == "unknown_id":
        mapping[0]["question_ids"].append("missing-question")
    elif mutation == "wrong_document":
        mapping[0]["document_id"] = "sha256:" + "a" * 64
        mapping[0]["source_sha256"] = "a" * 64
    elif mutation == "bundle_outside":
        mapping[0]["bundle_page_number"] = 20
    elif mutation == "duplicate_page":
        mapping.append(copy.deepcopy(mapping[0]))
    elif mutation == "missing_page":
        mapping.pop()
    elif mutation == "wrong_hash":
        mapping[0]["source_sha256"] = "a" * 64
    elif mutation == "wrong_company":
        mapping[0]["company"] = "Different fictional company"
    elif mutation == "wrong_physical":
        mapping[0]["original_physical_page"] = 101
    else:
        mapping[0]["question_ids"].pop()
    contents["evidence_page_mapping.json"] = json_bytes(mapping)
    with pytest.raises(FinragError, match=expected):
        validate_snapshot(contents)


def test_fabricated_excerpt_is_rejected(snapshot: dict[str, Path]) -> None:
    contents = snapshot_contents(snapshot)
    rows = decode_json(contents["annotations_for_review.json"], "annotations")
    rows[0]["evidence_texts"][0]["text"] = "No such evidence in this synthetic PDF"
    contents["annotations_for_review.json"] = json_bytes(rows)
    with pytest.raises(FinragError, match="match.*exactly"):
        validate_snapshot(contents)


def test_whitespace_is_not_positive_evidence(snapshot: dict[str, Path]) -> None:
    contents = snapshot_contents(snapshot)
    rows = decode_json(contents["annotations_for_review.json"], "annotations")
    rows[0]["evidence_texts"][0]["text"] = " "
    contents["annotations_for_review.json"] = json_bytes(rows)
    with pytest.raises(FinragError, match="invalid field"):
        validate_snapshot(contents)


def test_snapshot_normalization_retains_financial_context(snapshot: dict[str, Path]) -> None:
    rows = json.loads(snapshot["annotations_for_review.json"].read_text(encoding="utf-8"))
    question = normalize_snapshot(rows)[0]
    assert question.period.values == [2025]
    assert question.scope.label == "Synthetic company bnp"
    assert question.context_annotation_status == "candidate"


def test_strict_domain_validation_rejects_coercion_and_paths() -> None:
    digest = "a" * 64
    values = dict(
        document_id=f"sha256:{digest}",
        source_sha256=digest,
        source_filename="synthetic.pdf",
        physical_page_number=1,
        text="value",
        is_blank=False,
        extraction_status="extracted",
    )
    for field, value in (
        ("physical_page_number", "1"),
        ("source_filename", "C:\\private.pdf"),
        ("is_blank", True),
        ("surprise", "unexpected"),
    ):
        with pytest.raises(ValidationError):
            PageRecord(**{**values, field: value})


def test_public_example_is_synthetic_and_valid() -> None:
    path = Path(__file__).resolve().parents[1] / "data/public_examples/evaluation_question.json"
    content = path.read_text(encoding="utf-8")
    question = EvaluationQuestion.model_validate_json(content)
    assert question.company == "Fictional Example Bank"
    for forbidden in ("BNP Paribas", "Crédit Agricole", "Société Générale", "human_review"):
        assert forbidden not in content


def test_bundle_requires_exactly_nineteen_pdf_pages(snapshot: dict[str, Path]) -> None:
    from conftest import pdf_bytes

    contents = snapshot_contents(snapshot)
    contents["evidence_pages.pdf"] = pdf_bytes(["Synthetic evidence"] * 18)
    with pytest.raises(FinragError, match="19 bundle pages"):
        validate_snapshot(contents)


def test_frozen_configuration_drift_is_rejected(snapshot: dict[str, Path]) -> None:
    contents = snapshot_contents(snapshot)
    config = decode_json(contents["baseline.json"], "configuration")
    config["chunking"]["max_tokens"] = 1000
    contents["baseline.json"] = json_bytes(config)
    with pytest.raises(FinragError, match="hashes"):
        validate_snapshot(contents)


def test_malformed_frozen_split_has_domain_error(snapshot: dict[str, Path]) -> None:
    contents = snapshot_contents(snapshot)
    frozen = decode_json(contents["finrag_v1.frozen.json"], "frozen")
    frozen["question_ids_by_split"] = []
    contents["finrag_v1.frozen.json"] = json_bytes(frozen)
    with pytest.raises(FinragError, match="Frozen split IDs"):
        validate_snapshot(contents)
