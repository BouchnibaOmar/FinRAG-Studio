import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from test_bm25 import lexical_config
from test_corpus_import import corpus  # noqa: F401

from finrag.config import FinragError
from finrag.corpus.prepare_corpus import prepare_chunks, prepare_extraction
from finrag.domain.evaluation import EvaluationQuestion
from finrag.domain.manifests import DatasetCounts, DatasetManifest
from finrag.evaluation.run_lexical_experiment import (
    SELECTION_PRIORITY,
    publish_dev_report,
    run_experiment,
    select_configuration,
)
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes


@pytest.fixture
def experiment_fixture(tmp_path: Path, request):
    _, manifest_path, report = request.getfixturevalue("corpus")
    config = lexical_config()
    config_path = tmp_path / "A.json"
    config_path.write_bytes(json_bytes(config.model_dump(mode="json")))
    policy = tmp_path / "policy.json"
    policy.write_bytes(
        json_bytes(
            {
                "phase": "2A",
                "evaluation_split": "dev",
                "selection_priority": SELECTION_PRIORITY,
                "historical_test_metrics_used_for_selection": False,
            }
        )
    )
    extracted = tmp_path / "data/private/corpus/extracted"
    chunk_root = tmp_path / "data/private/corpus/chunks"
    prepare_extraction(manifest_path, extracted, tmp_path)
    prepare_chunks(manifest_path, extracted, chunk_root, config, tmp_path)
    dataset = tmp_path / "data/private/evaluation/synthetic"
    dataset.mkdir(parents=True)
    example = json.loads(
        (
            Path(__file__).resolve().parents[1] / "data/public_examples/evaluation_question.json"
        ).read_text()
    )
    questions = []
    for index in range(1, 31):
        value = {
            **example,
            "question_id": f"finrag_v1_example_{index:02}",
            "document_id": report.document_id,
            "question": "profit revenue",
            "relevant_pages": [1],
            "evidence": [{"physical_page_number": 1, "text": "Fictional revenue"}],
        }
        if index > 28:
            value.update(
                answerable=False,
                category="unanswerable",
                relevant_pages=[],
                evidence=[],
                evidence_bundle_pages=[],
            )
        questions.append(EvaluationQuestion.model_validate(value))
    (dataset / "dev.jsonl").write_bytes(
        jsonl_bytes([question.model_dump(mode="json") for question in questions])
    )
    manifest = DatasetManifest(
        dataset_name="finrag",
        dataset_version="finrag_v1",
        schema_version="1.0",
        creation_timestamp=datetime.now(UTC),
        review_status="externally_reviewed_ai_assisted",
        counts=DatasetCounts(
            questions=45,
            dev=30,
            test=15,
            answerable=42,
            unanswerable=3,
            candidate=44,
            verified=1,
            evidence_pages=19,
            evidence_excerpts=55,
        ),
        source_artifacts={},
        source_hashes={},
        imported_hashes={"dev.jsonl": hash_file(dataset / "dev.jsonl")},
        transformation_history=[],
        limitations=[],
        prototype_baseline_reference="unused",
    )
    (dataset / "dataset_manifest.json").write_bytes(json_bytes(manifest.model_dump(mode="json")))
    # No test file exists: a development experiment must not need one.
    return (
        config_path,
        policy,
        dataset,
        manifest_path,
        extracted,
        chunk_root,
        tmp_path / "artifacts/private/runs",
        tmp_path,
    )


def test_experiment_and_cached_validation_never_load_test(experiment_fixture) -> None:
    result = run_experiment(*experiment_fixture)
    assert result["groups"]["overall"]["answerable"] == 28
    assert result["groups"]["overall"]["metrics"]["hit@5"] == 1
    assert result["test_evaluated"] is False
    assert len(result["unanswerable_question_ids"]) == 2
    with patch("finrag.retrieval.bm25.BM25.search", side_effect=AssertionError("do not rerun")):
        assert run_experiment(*experiment_fixture) == result


def test_predictions_tampering_is_detected(experiment_fixture) -> None:
    run_experiment(*experiment_fixture)
    path = experiment_fixture[6] / "A/predictions.jsonl"
    with path.open("ab") as output:
        output.write(b"\n")
    with pytest.raises(FinragError, match="prediction hashes"):
        run_experiment(*experiment_fixture)


def test_public_report_has_no_query_answer_or_chunk_text(
    experiment_fixture, tmp_path: Path
) -> None:
    result = run_experiment(*experiment_fixture)
    path = tmp_path / "public_report.json"
    publish_dev_report([result], path)
    text = path.read_text(encoding="utf-8")
    assert "profit revenue" not in text
    assert "Fictional revenue profit CET1" not in text
    assert "EUR 120 million in this synthetic example" not in text
    assert '"text"' not in text and '"question"' not in text and '"reference_answer"' not in text


@pytest.mark.parametrize("winning_field", ["hit@5", "mrr@10", "hit@1", "chunk_count", "overlap"])
def test_selection_priority_is_applied_without_test(winning_field: str) -> None:
    base = {
        "split": "dev",
        "test_evaluated": False,
        "groups": {"overall": {"metrics": {"hit@5": 0.5, "mrr@10": 0.5, "hit@1": 0.5}}},
        "configuration": {"configuration_id": "A", "chunking": {"overlap": 100}},
        "chunk_count": 100,
    }
    other = json.loads(json.dumps(base))
    other["configuration"]["configuration_id"] = "B"
    if winning_field in ("hit@5", "mrr@10", "hit@1"):
        other["groups"]["overall"]["metrics"][winning_field] = 0.6
    elif winning_field == "chunk_count":
        other["chunk_count"] = 99
    else:
        other["configuration"]["chunking"]["overlap"] = 99
    assert select_configuration([base, other])["selected_configuration"] == "B"


def test_selection_rejects_test_results() -> None:
    with pytest.raises(FinragError, match="development"):
        select_configuration([{"split": "test", "test_evaluated": True}])
