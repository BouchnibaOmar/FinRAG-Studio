import json
import math
import os
import re

from pydantic import ValidationError

from finrag.config import FinragError
from finrag.generation import answers, citation_units
from finrag.generation.answers import AnswerContext
from finrag.ingestion.hashing import hash_bytes

CREDENTIAL_FIELDS = {
    "authorization",
    "proxyauthorization",
    "headers",
    "requestheaders",
    "responseheaders",
    "apikey",
    "password",
    "accesstoken",
    "refreshtoken",
    "cookie",
    "setcookie",
}


def redact_private(value: object) -> object:
    if isinstance(value, dict):
        return {
            redact_private(key): "[REDACTED]"
            if re.sub(r"[^a-z]", "", key.lower()) in CREDENTIAL_FIELDS
            else redact_private(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_private(item) for item in value]
    if isinstance(value, str):
        secret = os.environ.get("OPENROUTER_API_KEY")
        if secret:
            value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)\bBearer\s+[^\s\"'<>;,]+", "Bearer [REDACTED]", value)
        return re.sub(r"\bsk-or-v1-[A-Za-z0-9_-]+", "[REDACTED]", value)
    return value


def response_snapshot(body: bytes, status: int, latency: float) -> dict:
    text = body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except ValueError:
        content = redact_private(text)
        representation = "body_text_redacted"
        changed = content != text
    else:
        content = redact_private(parsed)
        representation = "body_json_redacted"
        changed = content != parsed
    snapshot = {
        "diagnostic_version": "generation-response-v1",
        "http_status": status,
        "latency_seconds": latency,
        "received_body_bytes": len(body),
        "received_body_sha256": hash_bytes(body),
        representation: content,
        "redaction_applied": changed,
        "utf8_decode_replacement": text.encode("utf-8") != body,
        "authentication_headers_recorded": False,
    }
    if representation == "body_json_redacted" and not changed:
        snapshot["body_text_redacted"] = redact_private(text)
    return snapshot


def unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def usage_details(raw: object) -> dict | None:
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(usage, dict):
        return None

    def numbers(value: dict, depth: int = 0) -> dict:
        if depth > 8:
            return {}
        result = {}
        for key, item in value.items():
            if re.sub(r"[^a-z]", "", key.lower()) in CREDENTIAL_FIELDS:
                continue
            if item is None or (type(item) in (int, float) and math.isfinite(item) and item >= 0):
                result[key] = item
            elif isinstance(item, dict):
                result[key] = numbers(item, depth + 1)
        return result

    return numbers(usage)


def content_analysis(context: AnswerContext, raw: object) -> dict:
    contract = citation_units if isinstance(context, citation_units.UnitContext) else answers
    response_model = (
        citation_units.UnitAnswer if contract is citation_units else answers.GroundedAnswer
    )
    if not isinstance(raw, dict) or not isinstance(raw.get("choices"), list):
        return {"stage": "envelope", "code": "missing_choices"}
    if len(raw["choices"]) != 1 or not isinstance(raw["choices"][0], dict):
        return {"stage": "envelope", "code": "unexpected_choices"}
    message = raw["choices"][0].get("message")
    if not isinstance(message, dict):
        return {"stage": "content", "code": "missing_message"}
    text = message.get("content")
    if not isinstance(text, str):
        return {"stage": "content", "code": "non_text_content", "content_type": type(text).__name__}
    if not text.strip():
        return {"stage": "content", "code": "empty_content", "characters": len(text)}
    try:
        value = json.loads(text, object_pairs_hook=unique_object)
    except json.JSONDecodeError as error:
        return {
            "stage": "json",
            "code": "invalid_json",
            "parser_message": error.msg,
            "line": error.lineno,
            "column": error.colno,
            "position": error.pos,
            "characters": len(text),
            "error_at_end": error.pos >= len(text.rstrip()),
            "incompleteness_not_proven_by_parser_alone": True,
        }
    except ValueError:
        return {"stage": "json", "code": "duplicate_json_key"}
    try:
        response_model.model_validate(value)
    except ValidationError as error:
        return {
            "stage": "schema",
            "code": "invalid_schema",
            "errors": [
                {"type": e["type"], "location": list(e["loc"])}
                for e in error.errors(include_input=False, include_context=False, include_url=False)
            ],
        }
    try:
        contract.validate_answer(context, value)
    except FinragError as error:
        return {
            "stage": "grounding",
            "code": "invalid_citation_or_evidence_rule",
            "rule": str(error),
        }
    return {"stage": "complete", "code": "content_valid_independent_of_finish_reason"}
