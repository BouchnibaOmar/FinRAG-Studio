import base64
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from finrag.config import FileOperationError, FinragError, filesystem_operation
from finrag.evaluation.question_translation import timestamp
from finrag.ingestion.manifest import json_bytes, read_json, write_atomic
from finrag.retrieval.candidate_reranking import (
    PROMPT_VERSION,
    RerankInput,
    RerankSettings,
    TransportReply,
    parse_response,
    reported_usage,
    request_identity,
    request_payload,
    token_reservation,
    validate_order,
)


class RerankerTransport(Protocol):
    def request(self, payload: dict) -> TransportReply: ...


def reusable_record(root: Path, source: RerankInput, settings: RerankSettings) -> dict | None:
    key = request_identity(source, settings)
    path = root / "records" / f"{key}.json"
    if not path.exists():
        return None
    record = read_json(path)
    if record["request_identity"] != key or record["payload"] != request_payload(source, settings):
        raise FinragError("Cached reranking request fingerprint differs.")
    attempt = read_json(root / "attempts" / f"{record['attempt_id']}.json")
    previous = RerankSettings.model_validate(attempt["execution_settings"])
    # Global admission limits do not change a model request or its response contract.
    limits = {"max_calls", "max_total_tokens"}
    if previous.model_dump(exclude=limits) != settings.model_dump(exclude=limits):
        raise FinragError("Cached reranking request configuration differs.")
    validate_order(source, record["ordered_candidate_ids"])
    return record


def load_model_snapshot(path: Path, settings: RerankSettings) -> dict:
    snapshot = read_json(path)
    try:
        if snapshot["official_source"] != "https://openrouter.ai/api/v1/models":
            raise ValueError
        retrieved = datetime.fromisoformat(snapshot["retrieved_at"])
        age = datetime.now(UTC) - retrieved
        if retrieved.tzinfo is None or age < timedelta(0) or age > timedelta(hours=24):
            raise FinragError("Model snapshot must be dated within the previous 24 hours.")
        metadata = snapshot["model"]
        if metadata["id"] != settings.model:
            raise FinragError("Official snapshot must match the exact reranking model.")
        if type(metadata["context_length"]) is not int or metadata["context_length"] <= 0:
            raise ValueError
        supported = metadata["supported_parameters"]
        required = {"response_format", "structured_outputs"}
        if settings.temperature_zero:
            required.add("temperature")
        if settings.reasoning_effort is not None:
            required.add("reasoning")
            reasoning = metadata.get("reasoning", {})
            if settings.reasoning_effort not in reasoning.get("supported_efforts", []):
                raise FinragError("Configured reasoning effort is not advertised by this model.")
            if settings.reasoning_effort == "none" and reasoning.get("mandatory"):
                raise FinragError("This model requires reasoning; effort none is incompatible.")
        if not required.issubset(set(supported)):
            raise FinragError(
                "Configured model does not advertise the required request parameters."
            )
        prices = [
            metadata["pricing"],
            *[p["pricing"] for p in snapshot.get("provider_profiles", [])],
        ]
        metadata = {
            **metadata,
            "budget_pricing": {
                key: str(max(Decimal(p.get(key, "0")) for p in prices))
                for key in (
                    "prompt",
                    "completion",
                    *(["request"] if any("request" in p for p in prices) else []),
                )
            },
        }
        return metadata
    except (KeyError, TypeError, ValueError, ArithmeticError) as error:
        if isinstance(error, FinragError):
            raise
        raise FinragError("Official model snapshot has an invalid or incomplete schema.") from error


def context_check(
    source: RerankInput, settings: RerankSettings, metadata: dict | None
) -> bool | None:
    if metadata is None:
        return None
    if metadata["id"] != settings.model:
        raise FinragError("Context metadata differs from the configured model.")
    maximum = metadata.get("top_provider", {}).get("max_completion_tokens")
    if maximum is not None and settings.max_completion_tokens > maximum:
        return False
    return token_reservation(source, settings)[1] <= metadata["context_length"]


def estimated_cost(
    inputs: list[RerankInput], settings: RerankSettings, metadata: dict | None
) -> dict | None:
    if metadata is None:
        return None
    try:
        pricing = metadata.get("budget_pricing", metadata["pricing"])
        prompt, completion = Decimal(pricing["prompt"]), Decimal(pricing["completion"])
        request = Decimal(pricing.get("request", "0"))
        if any(not value.is_finite() or value < 0 for value in (prompt, completion, request)):
            raise ValueError
        cost = sum(
            token_reservation(source, settings)[0] * prompt
            + settings.max_completion_tokens * completion
            + request
            for source in inputs
        )
        return {
            "estimated_usd": str(cost),
            "request_price_reported": "request" in pricing,
            "basis": "UTF-8 admission reservations, catalogue rates; not a monetary guarantee",
        }
    except (KeyError, TypeError, ValueError, ArithmeticError) as error:
        raise FinragError("Official pricing is invalid; no cost can be estimated.") from error


def check_estimated_spend(
    inputs: list[RerankInput],
    settings: RerankSettings,
    metadata: dict | None,
    limit_usd: str | None,
) -> None:
    if limit_usd is None:
        return
    try:
        limit = Decimal(limit_usd)
        if not limit.is_finite() or limit <= 0:
            raise ValueError
    except (ValueError, ArithmeticError) as error:
        raise FinragError("Estimated spend limit must be a positive finite USD amount.") from error
    cost = estimated_cost(inputs, settings, metadata)
    if cost is None or Decimal(cost["estimated_usd"]) > limit:
        raise FinragError("Reserved estimated spend exceeds its limit or lacks verified prices.")


def cached_rerank(
    root: Path,
    source: RerankInput,
    settings: RerankSettings,
    transport: RerankerTransport,
    metadata: dict | None,
) -> dict:
    with filesystem_operation("create reranking cache", root):
        root.mkdir(parents=True, exist_ok=True)
    lock = root / "budget.lock"
    try:
        with filesystem_operation("acquire reranking budget lock", lock):
            lock.mkdir()
    except FileOperationError as error:
        if isinstance(error.__cause__, FileExistsError):
            raise FinragError(
                "Reranking budget is locked; inspect local pending attempts."
            ) from error
        raise
    try:
        return execute_locked(root, source, settings, transport, metadata)
    finally:
        with filesystem_operation("release reranking budget lock", lock):
            lock.rmdir()


def execute_locked(
    root: Path,
    source: RerankInput,
    settings: RerankSettings,
    transport: RerankerTransport,
    metadata: dict | None,
) -> dict:
    key = request_identity(source, settings)
    record_path = root / "records" / f"{key}.json"
    if record_path.exists():
        record = read_json(record_path)
        if record["request_identity"] != key or record["payload"] != request_payload(
            source, settings
        ):
            raise FinragError("Reranking cache fingerprint differs; overwrite refused.")
        validate_order(source, record["ordered_candidate_ids"])
        return record
    attempts = [read_json(path) for path in (root / "attempts").glob("*.json")]
    if any(row["request_identity"] == key for row in attempts):
        raise FinragError("Previous reranking attempt has no valid cache; no automatic replay.")
    if any(
        read_json(path).get("supplier_token_overrun") for path in (root / "outcomes").glob("*.json")
    ):
        raise FinragError("Supplier token overrun requires budget review; no call sent.")
    if context_check(source, settings, metadata) is not True:
        raise FinragError("Reranking context is unverified or insufficient; no call sent.")
    if not settings.model:
        raise FinragError("Exact reranking model is not configured; no call sent.")
    reserved = token_reservation(source, settings)[1]
    if (
        len(attempts) >= settings.max_calls
        or sum(row["reserved_tokens"] for row in attempts) + reserved > settings.max_total_tokens
    ):
        raise FinragError("Reranking call/token budget exhausted; no call sent.")
    attempt_id = uuid4().hex
    payload = request_payload(source, settings)
    write_atomic(
        root / "attempts" / f"{attempt_id}.json",
        json_bytes(
            {
                "request_identity": key,
                "requested_model": settings.model,
                "timestamp": timestamp(),
                "reserved_tokens": reserved,
                "call_count": 1,
                "prompt_version": PROMPT_VERSION,
                "output_schema_version": settings.output_schema_version,
                "payload": payload,
                "execution_settings": settings.model_dump(mode="json"),
                "model_context_length": metadata["context_length"],
                "candidate_input_order": [row.candidate_id for row in source.candidates],
            }
        ),
    )
    outcome = {
        "status": "failed_api",
        "usage": None,
        "latency_seconds": None,
        "supplier_token_overrun": False,
        "request_identity": key,
    }
    try:
        reply = transport.request(payload)
        outcome["latency_seconds"] = reply.latency_seconds
        # Retain body bytes, never authentication headers or transport error contents.
        write_atomic(
            root / "responses" / f"{attempt_id}.json",
            json_bytes(
                {
                    "http_status": reply.status,
                    "body_base64": base64.b64encode(reply.body).decode("ascii"),
                    "latency_seconds": reply.latency_seconds,
                }
            ),
        )
        try:
            raw = json.loads(reply.body.decode("utf-8"))
        except (ValueError, UnicodeError) as error:
            outcome["status"] = "failed_validation" if reply.status == 200 else "failed_api"
            raise FinragError("Reranker HTTP body is not valid UTF-8 JSON.") from error
        usage = reported_usage(raw)
        outcome["usage"] = usage
        total = usage.get("total_tokens") if usage else None
        outcome["supplier_token_overrun"] = total is not None and total > reserved
        if reply.status != 200:
            raise FinragError(f"OpenRouter reranker returned HTTP {reply.status}; no retry.")
        if isinstance(raw, dict) and "error" in raw:
            raise FinragError("OpenRouter returned an API error object; no retry.")
        outcome["status"] = "failed_validation"
        parsed = parse_response(source, raw)
        record = {
            **parsed,
            "request_identity": key,
            "requested_model": settings.model,
            "payload": payload,
            "prompt_version": PROMPT_VERSION,
            "output_schema_version": settings.output_schema_version,
            "timestamp": timestamp(),
            "candidate_input_order": [row.candidate_id for row in source.candidates],
            "attempt_id": attempt_id,
            "raw_response_path": str((root / "responses" / f"{attempt_id}.json").resolve()),
            "usage": usage,
            "reported_cost": usage.get("cost") if usage else None,
            "latency_seconds": reply.latency_seconds,
            "returned_model_differs": parsed["returned_model"] != settings.model,
        }
        write_atomic(record_path, json_bytes(record))
        outcome["status"] = "success"
        return record
    except FinragError as error:
        outcome["failure_reason"] = str(error)
        raise
    finally:
        write_atomic(root / "outcomes" / f"{attempt_id}.json", json_bytes(outcome))
