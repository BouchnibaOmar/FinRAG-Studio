import http.client
import json
import math
import os
import re
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, Self

from pydantic import Field, model_validator

from finrag.config import FinragError
from finrag.domain.documents import StrictRecord
from finrag.ingestion.hashing import hash_bytes
from finrag.ingestion.manifest import json_bytes

PROMPT_VERSION = "financial-question-translation-v1"
TRANSLATION_INSTRUCTION = (
    "Traduire fidèlement la question dans la langue cible. "
    "Ne pas répondre, expliquer, enrichir ou ajouter des synonymes. "
    "Préserver entreprise, périmètre juridique, dates, chiffres, "
    "unités, devise, signe et distinction publié/sous-jacent. "
    "Retourner uniquement la question traduite, sans titre ni mise en forme."
)


class TranslationInput(StrictRecord):
    question: str = Field(min_length=1, max_length=2000)
    source_language: Literal["fr", "en"]
    target_language: Literal["fr", "en"]


class TranslationSettings(StrictRecord):
    model: str | None = None
    temperature_zero: bool = True
    timeout_seconds: int = Field(default=45, ge=1, le=120)
    max_completion_tokens: int = Field(default=256, ge=16, le=1024)
    prompt_token_allowance: int = Field(default=2048, ge=512, le=8192)
    max_calls: int = Field(default=17, ge=1, le=100)
    max_total_tokens: int = Field(default=40000, ge=1, le=1000000)
    attempts_per_input: Literal[1] = 1

    @model_validator(mode="after")
    def explicit_model(self) -> Self:
        if self.model is not None and (
            not re.fullmatch(r"[A-Za-z0-9_.:/-]+", self.model)
            or "/" not in self.model
            or self.model.startswith("openrouter/")
        ):
            raise ValueError(
                "An exact model ID is required; automatic OpenRouter routing is excluded"
            )
        return self


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def input_identity(source: TranslationInput) -> str:
    return hash_bytes(json_bytes(source.model_dump(mode="json")))


def request_payload(source: TranslationInput, settings: TranslationSettings) -> dict:
    payload = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": TRANSLATION_INSTRUCTION},
            {
                "role": "user",
                "content": json.dumps(source.model_dump(mode="json"), ensure_ascii=False),
            },
        ],
        "max_completion_tokens": settings.max_completion_tokens,
        "stream": False,
        "provider": {"allow_fallbacks": False, "require_parameters": True},
    }
    if settings.temperature_zero:
        payload["temperature"] = 0
    return payload


def request_identity(source: TranslationInput, settings: TranslationSettings) -> str:
    return hash_bytes(
        json_bytes({"prompt_version": PROMPT_VERSION, "payload": request_payload(source, settings)})
    )


def numeric_values(text: str) -> list[str]:
    text = text.replace("−", "-").replace("\u00a0", " ").replace("\u202f", " ")
    tokens = re.findall(r"[-+]?\d+(?:(?:[.,]\d+)|(?: \d{3}))*\s*%?", text)
    values = []
    for token in tokens:
        number = token.strip()
        percent = number.endswith("%")
        number = number.rstrip("% ").replace(" ", "")
        if "," in number and "." in number:
            decimal_separator = "," if number.rfind(",") > number.rfind(".") else "."
            number = number.replace("." if decimal_separator == "," else ",", "")
        number = number.replace(",", ".")
        # Three-digit separators are ambiguous across financial locales; flag them separately.
        if number.count(".") > 1:
            number = number.replace(".", "")
        values.append(
            ("+" if number.startswith("+") else "")
            + str(Decimal(number).normalize())
            + ("%" if percent else "")
        )
    return sorted(values)


def translation_checks(source: TranslationInput, translated: str) -> dict:
    number_match = numeric_values(source.question) == numeric_values(translated)
    years_match = Counter(re.findall(r"\b(?:19|20)\d{2}\b", source.question)) == Counter(
        re.findall(r"\b(?:19|20)\d{2}\b", translated)
    )
    risks = [
        "names_legal_scope_units_currency_and_stated_underlying_require_human_review",
        "month_names_and_non_numeric_dates_require_human_review",
        "target_language_and_faithfulness_not_verified_automatically",
    ]
    if "s.a." in source.question.casefold() and "s.a." not in translated.casefold():
        risks.append("legal_entity_suffix_changed")
    for unit_pattern in (r"\bmillions?\b", r"\b(?:billions?|milliards?)\b"):
        if len(re.findall(unit_pattern, source.question, re.IGNORECASE)) != len(
            re.findall(unit_pattern, translated, re.IGNORECASE)
        ):
            risks.append("million_billion_unit_changed")
    if re.search(r"\d[.,]\d{3}\b", source.question + " " + translated):
        risks.append("ambiguous_numeric_separator")
    dates_match = re.findall(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b", source.question) == re.findall(
        r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b", translated
    )
    if not dates_match:
        risks.append("numeric_date_literal_changed")
    for name in ("BNP Paribas", "Crédit Agricole", "Société Générale"):
        if (
            name.casefold() in source.question.casefold()
            and name.casefold() not in translated.casefold()
        ):
            risks.append("company_name_not_preserved_literally")
    for symbol in ("€", "$", "EUR", "USD", "%"):
        if source.question.count(symbol) != translated.count(symbol):
            risks.append(f"currency_or_unit_literal_changed:{symbol}")
    return {
        "numbers_preserved": number_match,
        "numeric_years_preserved": years_match,
        "numeric_dates_preserved_literally": dates_match,
        "risks": risks,
        "review_status": "candidate_not_validated",
    }


def validate_translation(source: TranslationInput, text: object) -> str:
    if not isinstance(text, str):
        raise FinragError("Translation message.content must be a string.")
    if not text.strip():
        raise FinragError("Translation message.content is empty or whitespace-only.")
    if "\n" in text.strip():
        raise FinragError("Translation contains multiple lines; explanations/formatting rejected.")
    if len(text) > min(2000, 3 * len(source.question) + 200):
        raise FinragError("Translation exceeds the permitted character length.")
    if text.strip().startswith(("```", "Translation:", "Traduction:", "Answer:", "Réponse:")):
        raise FinragError("Translation starts with a formatting, translation or answer label.")
    if source.question.rstrip().endswith("?") and not text.rstrip().endswith("?"):
        raise FinragError("Translation is missing the source question's final question mark.")
    return text.strip()


def response_diagnostic(response: object) -> dict:
    """Keep response fields needed for private review, never transport headers."""
    diagnostic = {"response_type": type(response).__name__}
    if not isinstance(response, dict):
        return diagnostic
    choices = response.get("choices")
    diagnostic["choices_type"] = type(choices).__name__
    if isinstance(choices, list):
        diagnostic["choice_count"] = len(choices)
        if len(choices) == 1 and isinstance(choices[0], dict):
            choice = choices[0]
            finish = choice.get("finish_reason")
            diagnostic["finish_reason"] = finish if isinstance(finish, str) else None
            message = choice.get("message")
            diagnostic["message_type"] = type(message).__name__
            if isinstance(message, dict):
                content = message.get("content")
                diagnostic["content_type"] = type(content).__name__
                diagnostic["content"] = content if isinstance(content, str) else None
                if isinstance(content, list):
                    diagnostic["content"] = [
                        {
                            key: part[key]
                            for key in ("type", "text")
                            if isinstance(part.get(key), str)
                        }
                        for part in content
                        if isinstance(part, dict)
                    ]
    for key in ("id", "model"):
        if isinstance(response.get(key), str):
            diagnostic[key] = response[key]
    usage = response.get("usage")
    diagnostic["usage"] = (
        {
            key: usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
            if type(usage.get(key)) in (int, float)
            and math.isfinite(usage[key])
            and usage[key] >= 0
        }
        if isinstance(usage, dict)
        else None
    )
    return diagnostic


class TranslationResponseError(FinragError):
    def __init__(self, reason: str, response: object):
        super().__init__(reason)
        self.diagnostic = response_diagnostic(response)


def parse_response(source: TranslationInput, response: object) -> dict:
    try:
        if not isinstance(response, dict) or "error" in response:
            raise ValueError
        choices = response["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError
        choice = choices[0]
        message = choice["message"]
        if choice["finish_reason"] != "stop":
            raise FinragError("Translation finish_reason must be stop; incomplete output rejected.")
        if (
            message.get("role") != "assistant"
            or message.get("tool_calls")
            or message.get("refusal")
        ):
            raise ValueError
        model = response["model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError
        text = validate_translation(source, message["content"])
        supplied_usage = response.get("usage")
        usage = None
        if supplied_usage is not None:
            if not isinstance(supplied_usage, dict):
                raise ValueError
            usage = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
                if key not in supplied_usage:
                    continue
                value = supplied_usage[key]
                if value is None:
                    usage[key] = None
                    continue
                if key == "cost":
                    if (
                        isinstance(value, bool)
                        or not math.isfinite(float(value))
                        or float(value) < 0
                    ):
                        raise ValueError
                elif type(value) is not int or value < 0:
                    raise ValueError
                usage[key] = value
            for key in ("prompt_tokens_details", "completion_tokens_details", "cost_details"):
                if key in supplied_usage:
                    details = supplied_usage[key]
                    if details is None:
                        usage[key] = None
                        continue
                    if not isinstance(details, dict):
                        raise ValueError
                    usage[key] = {}
                    for name, value in details.items():
                        if value is None:
                            usage[key][name] = None
                        elif (
                            type(value) not in (int, float) or not math.isfinite(value) or value < 0
                        ):
                            raise ValueError
                        else:
                            usage[key][name] = value
        return {
            "translation": text,
            "returned_model": model,
            "response_id": response.get("id"),
            "usage": usage,
            "reported_cost": usage.get("cost") if usage else None,
        }
    except FinragError as error:
        raise TranslationResponseError(str(error), response) from error
    except (KeyError, TypeError, IndexError, ValueError, AttributeError) as error:
        raise TranslationResponseError(
            "OpenRouter response has an invalid or incomplete translation schema.", response
        ) from error


class OpenRouterTranslator:
    def __init__(self, settings: TranslationSettings):
        if not settings.model:
            raise FinragError(
                "No exact OpenRouter model configured; supply --model or OPENROUTER_MODEL."
            )
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise FinragError("OPENROUTER_API_KEY is not configured in the environment.")
        self.settings = settings

    def translate(self, source: TranslationInput) -> dict:
        connection = http.client.HTTPSConnection(
            "openrouter.ai", timeout=self.settings.timeout_seconds
        )
        try:
            connection.request(
                "POST",
                "/api/v1/chat/completions",
                body=json.dumps(request_payload(source, self.settings)).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                raise FinragError(
                    f"OpenRouter returned HTTP {response.status}; no automatic retry."
                )
            content = response.read(1048577)
            if len(content) > 1048576:
                raise FinragError("OpenRouter response exceeds the accepted size.")
            parsed = json.loads(content.decode("utf-8"))
        except (OSError, http.client.HTTPException, UnicodeError, json.JSONDecodeError):
            raise FinragError(
                "OpenRouter transport/JSON failure; outcome may be billed, no retry."
            ) from None
        finally:
            connection.close()
        return {
            **parse_response(source, parsed),
            "source": "openrouter",
            "diagnostic": response_diagnostic(parsed),
        }


def official_price_estimate(metadata: dict, settings: TranslationSettings, calls: int) -> dict:
    if metadata.get("id") != settings.model:
        raise FinragError("Pricing snapshot must match the exact configured model ID.")
    pricing = metadata["pricing"]
    prompt = Decimal(pricing["prompt"])
    completion = Decimal(pricing["completion"])
    request = Decimal(pricing.get("request", "0"))
    if any(not value.is_finite() or value < 0 for value in (prompt, completion, request)):
        raise FinragError("Official model pricing must be finite and nonnegative.")
    estimate = calls * (
        settings.prompt_token_allowance * prompt
        + settings.max_completion_tokens * completion
        + request
    )
    return {
        "model": settings.model,
        "estimated_usd": str(estimate),
        "basis": "reserved prompt allowance and maximum completion tokens; not an invoice",
        "request_price_reported": "request" in pricing,
        "official_source": "https://openrouter.ai/api/v1/models",
    }


def fetch_official_model_metadata(model: str) -> dict:
    TranslationSettings(model=model)
    connection = http.client.HTTPSConnection("openrouter.ai", timeout=45)
    try:
        connection.request("GET", "/api/v1/models")
        response = connection.getresponse()
        if response.status != 200:
            raise FinragError(f"Official model catalog returned HTTP {response.status}.")
        body = response.read(16777217)
        if len(body) > 16777216:
            raise FinragError("Official model catalog exceeds the accepted size.")
        catalog = json.loads(body.decode("utf-8"))["data"]
        selected = [row for row in catalog if row["id"] == model]
        if len(selected) != 1:
            raise FinragError("Exact configured model not found uniquely in the official catalog.")
    except (OSError, http.client.HTTPException, UnicodeError, KeyError, json.JSONDecodeError):
        raise FinragError("Official model catalog transport/schema failure.") from None
    finally:
        connection.close()
    return {
        "official_source": "https://openrouter.ai/api/v1/models",
        "retrieved_at": timestamp(),
        "model": selected[0],
    }
