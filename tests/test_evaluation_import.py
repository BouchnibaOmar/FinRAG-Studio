import json
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import change_json

from finrag.config import FinragError
from finrag.evaluation.import_assets import IMPORTED_WORDING, ORIGINAL_WORDING, import_evaluation
from finrag.evaluation.validate_dataset import EXPECTED_COUNTS, validate_dataset
from finrag.ingestion.hashing import hash_file


def test_private_import_preserves_sources_and_statuses(
    snapshot: dict[str, Path], tmp_path: Path
) -> None:
    before = {name: hash_file(path) for name, path in snapshot.items()}
    output = tmp_path / "data/private/evaluation/finrag_v1"
    manifest = import_evaluation(snapshot, output, tmp_path)
    assert manifest.counts == EXPECTED_COUNTS
    assert validate_dataset(output) == manifest
    assert manifest.source_hashes == before
    assert {name: hash_file(path) for name, path in snapshot.items()} == before
    for name in snapshot:
        assert hash_file(output / "sources" / name) == before[name]
    rows = [
        json.loads(line)
        for split in ("dev", "test")
        for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    dev = {row["question_id"] for row in rows if row["split"] == "dev"}
    test = {row["question_id"] for row in rows if row["split"] == "test"}
    assert len(dev) == 30 and len(test) == 15 and dev.isdisjoint(test)
    assert sum(row["annotation_status"] == "candidate" for row in rows) == 44
    assert (
        next(row for row in rows if row["question_id"] == "finrag_v1_bnp_05")["question"]
        == IMPORTED_WORDING
    )
    source = json.loads(snapshot["annotations_for_review.json"].read_text(encoding="utf-8"))
    assert (
        next(row for row in source if row["question_id"] == "finrag_v1_bnp_05")["question"]
        == ORIGINAL_WORDING
    )
    assert manifest.transformation_history[0].original_value == ORIGINAL_WORDING
    assert manifest.transformation_history[0].version == "bnp-wording-v1"
    assert "human_review" not in (output / "dev.jsonl").read_text(encoding="utf-8")


def test_overwrite_requires_explicit_flag(snapshot: dict[str, Path], tmp_path: Path) -> None:
    output = tmp_path / "data/private/evaluation/finrag_v1"
    import_evaluation(snapshot, output, tmp_path)
    with pytest.raises(FinragError, match="overwrite"):
        import_evaluation(snapshot, output, tmp_path)
    import_evaluation(snapshot, output, tmp_path, overwrite=True)
    assert validate_dataset(output).counts == EXPECTED_COUNTS


def test_failed_validation_does_not_create_or_replace_destination(
    snapshot: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "data/private/evaluation/finrag_v1"
    import_evaluation(snapshot, output, tmp_path)
    before = (output / "dataset_manifest.json").read_bytes()
    change_json(snapshot["annotations_for_review.json"], lambda rows: rows.append(rows[0]))
    with pytest.raises(FinragError, match="Duplicate"):
        import_evaluation(snapshot, output, tmp_path, overwrite=True)
    assert (output / "dataset_manifest.json").read_bytes() == before
    absent = tmp_path / "data/private/evaluation/absent"
    with pytest.raises(FinragError):
        import_evaluation(snapshot, absent, tmp_path)
    assert not absent.exists()


def test_failed_publish_restores_existing_destination(
    snapshot: dict[str, Path], tmp_path: Path
) -> None:
    import os

    output = tmp_path / "data/private/evaluation/finrag_v1"
    import_evaluation(snapshot, output, tmp_path)
    before = (output / "dataset_manifest.json").read_bytes()
    rename = os.rename

    def fail_staged_rename(source, destination):
        if Path(source).name == "dataset":
            raise OSError("simulated publication failure")
        return rename(source, destination)

    with patch("finrag.ingestion.manifest.os.rename", side_effect=fail_staged_rename):
        with pytest.raises(OSError, match="publication failure"):
            import_evaluation(snapshot, output, tmp_path, overwrite=True)
    assert (output / "dataset_manifest.json").read_bytes() == before
    assert not list(output.parent.glob(".finrag-*"))


def test_sources_changed_during_import_are_rejected(
    snapshot: dict[str, Path], tmp_path: Path
) -> None:
    from finrag.evaluation import import_assets

    original = import_assets._write_snapshot

    def change_after_copy(*args, **kwargs):
        result = original(*args, **kwargs)
        snapshot["evidence_pages.pdf"].write_bytes(b"changed after validation")
        return result

    output = tmp_path / "data/private/evaluation/finrag_v1"
    with patch.object(import_assets, "_write_snapshot", side_effect=change_after_copy):
        with pytest.raises(FinragError, match="changed during import"):
            import_evaluation(snapshot, output, tmp_path)
    assert not output.exists()


def test_public_destination_is_rejected(snapshot: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(FinragError, match="private"):
        import_evaluation(snapshot, tmp_path / "data/public_examples/real", tmp_path)


def test_manifest_hashes_detect_tampering(snapshot: dict[str, Path], tmp_path: Path) -> None:
    output = tmp_path / "data/private/evaluation/finrag_v1"
    import_evaluation(snapshot, output, tmp_path)
    with (output / "dev.jsonl").open("ab") as file:
        file.write(b"\n")
    with pytest.raises(FinragError, match="hash changed"):
        validate_dataset(output)


def test_import_refuses_unapproved_file_types(snapshot: dict[str, Path], tmp_path: Path) -> None:
    paths = {**snapshot, "retrieval_baseline_predictions.jsonl": tmp_path / "private_predictions"}
    with pytest.raises(FinragError, match="seven approved"):
        import_evaluation(paths, tmp_path / "data/private/evaluation/finrag_v1", tmp_path)
