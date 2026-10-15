import json
from pathlib import Path

import pytest

from finrag.config import FinragError
from finrag.evaluation import bm25_translation_experiment as experiment_module
from finrag.evaluation.question_translation import (
    TranslationInput,
    TranslationResponseError,
    TranslationSettings,
    input_identity,
    parse_response,
    request_identity,
)
from finrag.evaluation.translation_cache import (
    cached_translation,
    import_corrections,
    load_translation_set,
    save_translation_set,
)
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, read_json, write_atomic


def source(label: str = "A") -> TranslationInput:
    return TranslationInput(
        question=f"Quelle valeur fictive {label} en 2025 ?",
        source_language="fr",
        target_language="en",
    )


def settings(**changes) -> TranslationSettings:
    return TranslationSettings(model="fictional/model", **changes)


def response(content: object, finish: str = "stop", total: int = 12) -> dict:
    return {
        "model": "fictional/model",
        "choices": [
            {"finish_reason": finish, "message": {"role": "assistant", "content": content}}
        ],
        "usage": {"total_tokens": total, "cost": 0.001},
        "headers": {"Authorization": "Bearer offline-secret"},
        "unknown_provider_field": "offline-secret",
    }


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (None, "must be a string"),
        ([{"type": "text", "text": "What fictional value in 2025?"}], "must be a string"),
        ("   ", "empty or whitespace-only"),
        ("Question?\nExplanation", "multiple lines"),
        ("x" * 1000 + "?", "character length"),
        ("Translation: What fictional value in 2025?", "answer label"),
        ("What fictional value in 2025.", "final question mark"),
    ],
)
def test_each_validation_failure_keeps_precise_reason_and_private_content(
    content: object, reason: str
) -> None:
    with pytest.raises(TranslationResponseError, match=reason) as caught:
        parse_response(source(), response(content))
    error = caught.value
    assert isinstance(error.__cause__, FinragError)
    assert reason in str(error.__cause__)
    assert error.diagnostic["content_type"] == type(content).__name__
    assert error.diagnostic["finish_reason"] == "stop"
    assert error.diagnostic["content"] == content
    assert "offline-secret" not in json.dumps(error.diagnostic)


def test_truncated_question_is_never_accepted() -> None:
    with pytest.raises(TranslationResponseError, match="finish_reason must be stop"):
        parse_response(source(), response("What fictional value in 2025?", "length"))


def test_rejected_response_retains_usage_and_reservation_without_replay(tmp_path: Path) -> None:
    class InvalidResponseDouble:
        calls = 0

        def translate(self, item):
            self.calls += 1
            return parse_response(item, response("", total=3000))

    double = InvalidResponseDouble()
    with pytest.raises(TranslationResponseError, match="empty"):
        cached_translation(tmp_path, source(), settings(), double)
    attempt = next((tmp_path / "attempts").glob("*.json"))
    assert read_json(attempt)["reserved_tokens"] == 2304
    diagnostic = read_json(tmp_path / "responses" / attempt.name)
    outcome = read_json(tmp_path / "outcomes" / attempt.name)
    assert diagnostic["content"] == "" and diagnostic["finish_reason"] == "stop"
    assert outcome["validation_reason"].endswith("empty or whitespace-only.")
    assert outcome["usage"] == {"total_tokens": 3000, "cost": 0.001}
    assert outcome["excess_reported_tokens"] == 696
    before = {p: hash_file(p) for p in tmp_path.rglob("*.json")}
    with pytest.raises(FinragError, match="no replay"):
        cached_translation(tmp_path, source(), settings(), double)
    with pytest.raises(FinragError, match="allowance overrun"):
        cached_translation(tmp_path, source("B"), settings(), double)
    assert double.calls == 1
    assert before == {p: hash_file(p) for p in tmp_path.rglob("*.json")}


class IdentityTranslationDouble:
    def __init__(self):
        self.calls = []

    def translate(self, item):
        self.calls.append(item.model_dump())
        return {**parse_response(item, response(item.question)), "source": "offline_double"}


def legacy_failure(root: Path, item: TranslationInput) -> None:
    write_atomic(
        root / "attempts/legacy.json",
        json_bytes(
            {
                "request_identity": request_identity(item, settings()),
                "reserved_tokens": 2304,
                "call_count": 1,
            }
        ),
    )
    write_atomic(
        root / "outcomes/legacy.json",
        json_bytes({"status": "uncertain_or_invalid", "usage": None}),
    )


def test_resume_legacy_missing_response_preserves_cache_and_spending(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    experiment = tmp_path / "artifacts/private/translation"
    root = experiment / "translator/cache"
    inputs = [source(label) for label in "ABCD"]
    double = IdentityTranslationDouble()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment_module, "translator_inputs", lambda path: inputs)
    monkeypatch.setattr(experiment_module, "OpenRouterTranslator", lambda config: double)
    write_atomic(
        experiment / "evaluation/identities.json",
        json_bytes(
            [
                {
                    "input_id": input_identity(item),
                    "question_id": f"dev_{label}",
                    "needs_translation": True,
                    "target_language": "en",
                }
                for item, label in zip(inputs, "ABCD", strict=True)
            ]
        ),
    )
    cached_translation(root, inputs[0], settings(), double)
    legacy_failure(root, inputs[1])
    before = {p: hash_file(p) for p in root.rglob("*.json")}
    double.calls.clear()
    result = experiment_module.translate(experiment, settings(max_calls=4), resume=True)
    manifest = read_json(result)
    assert manifest["complete"] is False
    assert manifest["missing_input_ids"] == [input_identity(inputs[1])]
    assert len(load_translation_set(root, result)) == 3
    assert double.calls == [item.model_dump() for item in inputs[2:]]
    assert all(
        set(call) == {"question", "source_language", "target_language"} for call in double.calls
    )
    assert all(hash_file(p) == digest for p, digest in before.items())
    assert len(list((root / "attempts").glob("*.json"))) == 4
    review = read_json(experiment / "review" / f"pairs-{result.stem}.json")
    assert review[1]["translation"] is None
    assert "dev_B" in capsys.readouterr().err
    double.calls.clear()
    experiment_module.translate(experiment, settings(max_calls=4), resume=True)
    assert double.calls == []


def test_cli_reports_question_id_and_reason_without_traceback_or_replay(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    experiment = tmp_path / "artifacts/private/translation"
    root = experiment / "translator/cache"
    item = source()
    legacy_failure(root, item)
    write_atomic(experiment / "translator/settings.json", json_bytes(settings().model_dump()))
    write_atomic(
        experiment / "evaluation/identities.json",
        json_bytes([{"input_id": input_identity(item), "question_id": "dev_failed"}]),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment_module, "translator_inputs", lambda path: [item])
    double = IdentityTranslationDouble()
    monkeypatch.setattr(experiment_module, "OpenRouterTranslator", lambda config: double)
    monkeypatch.setattr("sys.argv", ["translation", "translate", "--experiment", str(experiment)])
    with pytest.raises(SystemExit) as caught:
        experiment_module.main()
    assert caught.value.code == 2 and not double.calls
    output = capsys.readouterr()
    assert "dev_failed" in output.err and "no replay" in output.err
    assert "Traceback" not in output.err and "offline-secret" not in output.err


def test_cli_preserves_content_validation_reason(monkeypatch, capsys) -> None:
    def failing_command():
        try:
            parse_response(source(), response(""))
        except FinragError as error:
            raise FinragError(f"dev_failed: {error}") from error

    monkeypatch.setattr(experiment_module, "run_command", failing_command)
    with pytest.raises(SystemExit) as caught:
        experiment_module.main()
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "dev_failed" in error and "empty or whitespace-only" in error
    assert "Traceback" not in error


def test_correction_of_cached_question_does_not_complete_a_partial_set(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    entry = cached_translation(root, source(), settings(), IdentityTranslationDouble())
    missing = input_identity(source("B"))
    base = save_translation_set(root, [entry], missing_input_ids=[missing])
    correction = tmp_path / "correction.json"
    write_atomic(
        correction,
        json_bytes(
            {
                "base_set_sha256": hash_file(base),
                "corrections": [
                    {
                        "input_id": entry["input_id"],
                        "translation": "Which fictional value A in 2025?",
                        "reason": "Local wording review",
                    }
                ],
            }
        ),
    )
    result = import_corrections(root, base, correction)
    assert read_json(result)["missing_input_ids"] == [missing]
    assert read_json(result)["complete"] is False


@pytest.mark.parametrize("case", ["valid", "no_attempt", "changed_year", "not_declared"])
def test_missing_local_translation_requires_sealed_input_and_failed_attempt(
    tmp_path: Path, case: str
) -> None:
    root = tmp_path / "cache"
    entry = cached_translation(root, source(), settings(), IdentityTranslationDouble())
    item = source("B")
    identity = input_identity(item)
    if case != "no_attempt":
        legacy_failure(root, item)
    base = save_translation_set(
        root, [entry], missing_input_ids=[] if case == "not_declared" else [identity]
    )
    correction = tmp_path / "correction.json"
    write_atomic(
        correction,
        json_bytes(
            {
                "base_set_sha256": hash_file(base),
                "corrections": [
                    {
                        "input_id": identity,
                        "translation": "What fictional value B in 2024?"
                        if case == "changed_year"
                        else "What fictional value B in 2025?",
                        "reason": "Local candidate for a missing translation",
                    }
                ],
            }
        ),
    )
    before = {p: hash_file(p) for p in root.rglob("*.json")}
    if case != "valid":
        with pytest.raises(FinragError):
            import_corrections(root, base, correction, missing_inputs={identity: item})
        assert before == {p: hash_file(p) for p in root.rglob("*.json")}
        return
    result = import_corrections(root, base, correction, missing_inputs={identity: item})
    assert all(hash_file(p) == digest for p, digest in before.items())
    manifest = read_json(result)
    assert manifest["complete"] is True and manifest["missing_input_ids"] == []
    records = load_translation_set(root, result)
    assert records[entry["input_id"]]["cache_entry"] == entry
    record = records[identity]
    assert record["source"] == "local_correction"
    assert record["review_status"] == "corrected_locally_not_validated"
    assert record["supersedes"] is None
    assert record["usage"] is None and record["reported_cost"] is None
    assert record["returned_model"] is None
    assert record["replaces_failed_attempt"]["sha256"] == before[root / "attempts/legacy.json"]
    assert len(list((root / "attempts").glob("*.json"))) == 2
