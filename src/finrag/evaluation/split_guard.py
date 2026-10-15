from pathlib import Path

from finrag.config import FinragError
from finrag.domain.evaluation import EvaluationQuestion
from finrag.domain.manifests import DatasetManifest
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import read_jsonl


def authorize_split(
    split: str, allow_test: bool = False, evaluation_reason: str | None = None
) -> None:
    if split not in ("dev", "test"):
        raise FinragError("Evaluation split must be dev or test.")
    if split == "test" and (
        not allow_test or not evaluation_reason or not evaluation_reason.strip()
    ):
        raise FinragError(
            "Future test evaluation requires --allow-test and a written --evaluation-reason."
        )


def load_dev_questions(
    dataset: Path,
    split: str = "dev",
    allow_test: bool = False,
    evaluation_reason: str | None = None,
) -> list[EvaluationQuestion]:
    authorize_split(split, allow_test, evaluation_reason)
    if split != "dev":
        raise FinragError("Phase 2A excludes test evaluation even with the future opt-in guard.")
    manifest = DatasetManifest.model_validate_json((dataset / "dataset_manifest.json").read_bytes())
    path = dataset / "dev.jsonl"
    if hash_file(path) != manifest.imported_hashes.get("dev.jsonl"):
        raise FinragError("Development dataset hash differs from the approved imported snapshot.")
    questions = [EvaluationQuestion.model_validate(row) for row in read_jsonl(path)]
    if (
        len(questions) != 30
        or sum(question.answerable for question in questions) != 28
        or any(question.split != "dev" for question in questions)
        or len({question.question_id for question in questions}) != 30
    ):
        raise FinragError("Dev-only run requires 30 distinct development questions, 28 answerable.")
    return questions
