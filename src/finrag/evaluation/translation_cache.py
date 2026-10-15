from pathlib import Path
from typing import Protocol
from uuid import uuid4

from finrag.config import FileOperationError, FinragError, filesystem_operation, private_output
from finrag.evaluation.question_translation import (
    PROMPT_VERSION,
    TranslationInput,
    TranslationResponseError,
    TranslationSettings,
    input_identity,
    request_identity,
    request_payload,
    timestamp,
    translation_checks,
    validate_translation,
)
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, read_json, write_atomic


class Translator(Protocol):
    def translate(self, source: TranslationInput) -> dict: ...


def reserved_tokens(settings: TranslationSettings) -> int:
    return settings.prompt_token_allowance + settings.max_completion_tokens


def attempted_without_cache(
    root: Path, source: TranslationInput, settings: TranslationSettings
) -> bool:
    key = request_identity(source, settings)
    return not (root / "records" / f"{key}.json").exists() and any(
        read_json(path)["request_identity"] == key for path in (root / "attempts").glob("*.json")
    )


def cached_translation(
    root: Path, source: TranslationInput, settings: TranslationSettings, translator: Translator
) -> dict:
    with filesystem_operation("create translation cache", root):
        root.mkdir(parents=True, exist_ok=True)
    lock = root / "budget.lock"
    try:
        with filesystem_operation("acquire translation budget lock", lock):
            lock.mkdir()
    except FileOperationError as error:
        if not isinstance(error.__cause__, FileExistsError):
            raise
        raise FinragError(
            "Translation budget lock exists; inspect pending attempts before resuming."
        ) from error
    try:
        return _translate_with_budget(root, source, settings, translator)
    finally:
        with filesystem_operation("release translation budget lock", lock):
            lock.rmdir()


def _translate_with_budget(
    root: Path, source: TranslationInput, settings: TranslationSettings, translator: Translator
) -> dict:
    key = request_identity(source, settings)
    path = root / "records" / f"{key}.json"
    if path.exists():
        record = read_json(path)
        if (
            record["request_identity"] != key
            or record["input"] != source.model_dump(mode="json")
            or record["requested_model"] != settings.model
        ):
            raise FinragError("Existing translation cache fingerprint differs; overwrite refused.")
        validate_translation(source, record["translation"])
        return {
            "input_id": input_identity(source),
            "path": str(path.resolve()),
            "sha256": hash_file(path),
        }
    payload = request_payload(source, settings)
    admission_estimate = (
        sum(len(message["content"].encode("utf-8")) for message in payload["messages"]) + 256
    )
    if admission_estimate > settings.prompt_token_allowance:
        raise FinragError("Question exceeds the conservative prompt allowance; no call sent.")
    attempts = [read_json(p) for p in sorted((root / "attempts").glob("*.json"))]
    if any(attempt["request_identity"] == key for attempt in attempts):
        raise FinragError(
            "A previous attempt has no reusable cache; inspect or correct locally, no replay."
        )
    spent = sum(attempt["reserved_tokens"] for attempt in attempts)
    for result_path in (root / "outcomes").glob("*.json"):
        excess = read_json(result_path).get("excess_reported_tokens", 0)
        if excess:
            raise FinragError(
                "A supplier token allowance overrun requires budget review; no call sent."
            )
    if (
        len(attempts) >= settings.max_calls
        or spent + reserved_tokens(settings) > settings.max_total_tokens
    ):
        raise FinragError("Translation call/token budget exhausted; no call sent.")
    attempt_id = uuid4().hex
    write_atomic(
        root / "attempts" / f"{attempt_id}.json",
        json_bytes(
            {
                "request_identity": key,
                "timestamp": timestamp(),
                "requested_model": settings.model,
                "reserved_tokens": reserved_tokens(settings),
                "call_count": 1,
            }
        ),
    )
    try:
        response = translator.translate(source)
        if "diagnostic" in response:
            write_atomic(
                root / "responses" / f"{attempt_id}.json", json_bytes(response["diagnostic"])
            )
        text = validate_translation(source, response["translation"])
        usage = response.get("usage")
        total = usage.get("total_tokens") if usage else None
        excess = max(0, total - reserved_tokens(settings)) if total is not None else 0
        record = {
            "input_id": input_identity(source),
            "input": source.model_dump(mode="json"),
            "translation": text,
            "requested_model": settings.model,
            "returned_model": response["returned_model"],
            "prompt_version": PROMPT_VERSION,
            "parameters": {key: value for key, value in payload.items() if key != "messages"},
            "timestamp": timestamp(),
            "usage": usage,
            "reported_cost": response.get("reported_cost"),
            "response_id": response.get("response_id"),
            "source": response.get("source", "unspecified"),
            "review_status": "candidate_not_validated",
            "checks": translation_checks(source, text),
            "request_identity": key,
            "supplier_token_allowance_exceeded": excess > 0,
        }
        if response["returned_model"] != settings.model:
            record["checks"]["risks"].append("returned_model_differs_from_requested")
        write_atomic(path, json_bytes(record))
        write_atomic(
            root / "outcomes" / f"{attempt_id}.json",
            json_bytes(
                {
                    "status": "cached",
                    "excess_reported_tokens": excess,
                    "usage": usage,
                }
            ),
        )
        if excess:
            raise FinragError(
                "Supplier usage exceeded its allowance; cached response retained, stop calls."
            )
    except Exception as error:
        outcome = root / "outcomes" / f"{attempt_id}.json"
        if not outcome.exists():
            usage = None
            excess = 0
            if isinstance(error, TranslationResponseError):
                write_atomic(
                    root / "responses" / f"{attempt_id}.json", json_bytes(error.diagnostic)
                )
                usage = error.diagnostic.get("usage")
                total = usage.get("total_tokens") if usage else None
                excess = max(0, total - reserved_tokens(settings)) if total is not None else 0
            write_atomic(
                outcome,
                json_bytes(
                    {
                        "status": "uncertain_or_invalid",
                        "usage": usage,
                        "excess_reported_tokens": excess,
                        "validation_reason": str(error)
                        if isinstance(error, TranslationResponseError)
                        else None,
                    }
                ),
            )
        raise
    return {
        "input_id": input_identity(source),
        "path": str(path.resolve()),
        "sha256": hash_file(path),
    }


def save_translation_set(
    root: Path,
    records: list[dict],
    supersedes: str | None = None,
    missing_input_ids: list[str] | None = None,
) -> Path:
    if len({row["input_id"] for row in records}) != len(records):
        raise FinragError("Translation set contains duplicate input IDs.")
    path = root / "sets" / f"{uuid4().hex}.json"
    write_atomic(
        path,
        json_bytes(
            {
                "timestamp": timestamp(),
                "records": records,
                "supersedes": supersedes,
                "complete": not missing_input_ids,
                "missing_input_ids": missing_input_ids or [],
            }
        ),
    )
    return path


def load_translation_set(root: Path, path: Path) -> dict[str, dict]:
    if not path.resolve().is_relative_to((root / "sets").resolve()):
        raise FinragError("Translation set must remain inside this experiment's private cache.")
    records = read_json(path)["records"]
    selected = {}
    for entry in records:
        record_path = Path(entry["path"])
        if (
            not record_path.resolve().is_relative_to((root / "records").resolve())
            or hash_file(record_path) != entry["sha256"]
        ):
            raise FinragError("Translation record path/hash mismatch.")
        record = read_json(record_path)
        source = TranslationInput.model_validate(record["input"])
        if entry["input_id"] in selected or input_identity(source) != entry["input_id"]:
            raise FinragError("Translation set contains an inconsistent or duplicate input.")
        record["translation"] = validate_translation(source, record["translation"])
        selected[entry["input_id"]] = {**record, "cache_entry": entry}
    return selected


def import_corrections(
    root: Path,
    base_set: Path,
    correction_file: Path,
    missing_inputs: dict[str, TranslationInput] | None = None,
) -> Path:
    selected = load_translation_set(root, base_set)
    document = read_json(correction_file)
    if document["base_set_sha256"] != hash_file(base_set):
        raise FinragError("Corrections refer to another translation set.")
    missing = set(read_json(base_set).get("missing_input_ids", []))
    seen = set()
    for correction in document["corrections"]:
        identity = correction["input_id"]
        if identity in seen or not correction.get("reason", "").strip():
            raise FinragError("Correction must identify one existing input and explain the change.")
        seen.add(identity)
        if identity in selected:
            previous = selected[identity]
        else:
            source = (missing_inputs or {}).get(identity)
            if identity not in missing or source is None or input_identity(source) != identity:
                raise FinragError(
                    "Correction input is neither cached nor a declared missing input."
                )
            previous = missing_translation_record(root, source, selected)
        source = TranslationInput.model_validate(previous["input"])
        text = validate_translation(source, correction["translation"])
        checks = translation_checks(source, text)
        if not all(
            checks[key]
            for key in (
                "numbers_preserved",
                "numeric_years_preserved",
                "numeric_dates_preserved_literally",
            )
        ):
            raise FinragError("Local correction must preserve numbers, years and numeric dates.")
        path = root / "records" / f"manual-{uuid4().hex}.json"
        record = {
            **previous,
            "translation": text,
            "source": "local_correction",
            "timestamp": timestamp(),
            "checks": checks,
            "review_status": "corrected_locally_not_validated",
            "supersedes": previous.get("cache_entry"),
            "correction_reason": correction["reason"],
            "returned_model": None,
            "usage": None,
            "reported_cost": None,
            "response_id": None,
            "parameters": None,
            "prompt_version": None,
        }
        record.pop("cache_entry", None)
        write_atomic(path, json_bytes(record))
        selected[identity] = {
            **record,
            "cache_entry": {
                "input_id": identity,
                "path": str(path.resolve()),
                "sha256": hash_file(path),
            },
        }
        missing.discard(identity)
    return save_translation_set(
        root,
        [row["cache_entry"] for row in selected.values()],
        str(base_set),
        missing_input_ids=sorted(missing),
    )


def missing_translation_record(
    root: Path, source: TranslationInput, selected: dict[str, dict]
) -> dict:
    models = {record["requested_model"] for record in selected.values()}
    if len(models) != 1:
        raise FinragError("Missing correction requires one exact model in the base set.")
    model = next(iter(models))
    # A manual replacement belongs to the recorded failed request, not a new API call.
    keys = set()
    for record in selected.values():
        parameters = record.get("parameters")
        if not isinstance(parameters, dict):
            continue
        settings = TranslationSettings(
            model=model,
            temperature_zero="temperature" in parameters,
            max_completion_tokens=parameters["max_completion_tokens"],
        )
        keys.add(request_identity(source, settings))
    attempts = [
        path
        for path in (root / "attempts").glob("*.json")
        if read_json(path)["request_identity"] in keys
    ]
    if len(attempts) != 1:
        raise FinragError("Missing correction requires one matching historical attempt.")
    attempt = attempts[0]
    outcome = root / "outcomes" / attempt.name
    if not outcome.exists() or read_json(outcome)["status"] != "uncertain_or_invalid":
        raise FinragError("Missing correction requires a recorded unsuccessful outcome.")
    return {
        "input_id": input_identity(source),
        "input": source.model_dump(mode="json"),
        "requested_model": model,
        "request_identity": read_json(attempt)["request_identity"],
        "replaces_failed_attempt": {"path": str(attempt.resolve()), "sha256": hash_file(attempt)},
    }


def private_translation_cache(experiment: Path) -> Path:
    return private_output(experiment / "translator/cache", Path.cwd())
