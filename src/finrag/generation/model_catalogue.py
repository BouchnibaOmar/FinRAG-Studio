import http.client
import json
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import Field, model_validator

from finrag.config import FinragError
from finrag.domain.documents import NonEmptyText, StrictRecord
from finrag.generation.answers import AnswerSettings

CATALOGUE_URL = "https://openrouter.ai/api/v1/models"
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class ModelInfo(StrictRecord):
    model_id: NonEmptyText
    name: NonEmptyText
    context_length: int | None
    max_output_tokens: int | None
    input_price: Decimal | None
    output_price: Decimal | None
    request_price: Decimal | None
    parameters: list[str]
    input_modalities: list[str]
    output_modalities: list[str]
    reasoning: dict

    @property
    def url(self) -> str:
        return f"https://openrouter.ai/{self.model_id}"

    @property
    def efforts(self) -> list[str]:
        if "reasoning" not in self.parameters:
            return []
        if "supported_efforts" not in self.reasoning:
            return []
        declared = self.reasoning["supported_efforts"]
        values = list(EFFORTS) if declared is None else declared
        if not isinstance(values, list):
            return []
        return [
            e
            for e in EFFORTS
            if e in values and not (e == "none" and self.reasoning.get("mandatory") is True)
        ]


class ModelCatalogue(StrictRecord):
    retrieved_at: NonEmptyText
    source: Literal["https://openrouter.ai/api/v1/models"] = CATALOGUE_URL
    models: list[ModelInfo] = Field(min_length=1)


def price(value: object) -> Decimal | None:
    if not isinstance(value, str | int | float) or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    return result if result.is_finite() and result >= 0 else None


def positive_integer(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def strings(value: object) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def parse_catalogue(raw: object, retrieved_at: str) -> ModelCatalogue:
    if not isinstance(raw, dict) or not isinstance(raw.get("data"), list):
        raise FinragError("Le catalogue OpenRouter ne contient pas de liste de modèles.")
    models = []
    for row in raw["data"]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise FinragError("Le catalogue contient un modèle sans identifiant.")
        identifier = row["id"]
        pricing = row.get("pricing") or {}
        top = row.get("top_provider") or {}
        architecture = row.get("architecture") or {}
        if not all(isinstance(v, dict) for v in (pricing, top, architecture)):
            raise FinragError("Métadonnées du catalogue invalides.")
        models.append(
            ModelInfo(
                model_id=identifier,
                name=row.get("name") or identifier,
                context_length=positive_integer(row.get("context_length")),
                max_output_tokens=positive_integer(top.get("max_completion_tokens")),
                input_price=price(pricing.get("prompt")),
                output_price=price(pricing.get("completion")),
                request_price=price(pricing.get("request")),
                parameters=strings(row.get("supported_parameters")),
                input_modalities=strings(architecture.get("input_modalities")),
                output_modalities=strings(architecture.get("output_modalities")),
                reasoning=row.get("reasoning") if isinstance(row.get("reasoning"), dict) else {},
            )
        )
    if len({m.model_id for m in models}) != len(models):
        raise FinragError("Le catalogue contient des identifiants de modèles dupliqués.")
    return ModelCatalogue(retrieved_at=retrieved_at, models=models)


def fetch_catalogue() -> ModelCatalogue:
    connection = http.client.HTTPSConnection("openrouter.ai", timeout=15)
    try:
        # Public metadata only: no key, question or documentary text in this GET.
        connection.request("GET", "/api/v1/models", headers={"Accept": "application/json"})
        response = connection.getresponse()
        body = response.read(8 * 1024 * 1024 + 1)
        if response.status != 200:
            raise FinragError(f"Catalogue OpenRouter indisponible (HTTP {response.status}).")
        if len(body) > 8 * 1024 * 1024:
            raise FinragError("Catalogue OpenRouter trop volumineux.")
        return parse_catalogue(json.loads(body), datetime.now(UTC).isoformat())
    except (OSError, http.client.HTTPException, ValueError) as error:
        raise FinragError(
            "Lecture du catalogue impossible ; réessayez son actualisation."
        ) from error
    finally:
        connection.close()


def compatibility_errors(model: ModelInfo, max_output: int) -> list[str]:
    errors = []
    if model.model_id in ("openrouter/auto", "openrouter/free") or "/" not in model.model_id:
        errors.append("Sélection dynamique : choisissez un identifiant de modèle exact.")
    if "text" not in model.input_modalities or "text" not in model.output_modalities:
        errors.append("Entrée et sortie textuelles non déclarées.")
    for parameter in ("response_format", "structured_outputs", "max_tokens"):
        if parameter not in model.parameters:
            errors.append(f"Paramètre requis non déclaré : {parameter}.")
    if model.context_length is None:
        errors.append("Limite de contexte non disponible.")
    elif max_output >= model.context_length:
        errors.append("La sortie réservée ne laisse pas de place au contexte.")
    if model.max_output_tokens is None:
        errors.append("Limite de sortie non disponible : compatibilité non vérifiable.")
    elif max_output > model.max_output_tokens:
        errors.append("La sortie demandée dépasse la limite déclarée du modèle.")
    return errors


class GenerationSettings(AnswerSettings):
    max_completion_tokens: int = Field(default=1024, ge=1, le=32768)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = (
        None
    )
    supported_reasoning_efforts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def supported_reasoning(self) -> "GenerationSettings":
        if self.reasoning_effort is not None and (
            self.reasoning_effort not in self.supported_reasoning_efforts
        ):
            raise ValueError("Effort de raisonnement non déclaré pour ce modèle.")
        return self


class CostEstimate(StrictRecord):
    total_usd: Decimal | None
    output_reservation_usd: Decimal | None
    input_tokens: int | None
    method: str


def estimate_cost(
    model: ModelInfo, output_tokens: int, input_tokens: int | None = None
) -> CostEstimate:
    output = None if model.output_price is None else model.output_price * output_tokens
    total = None
    if input_tokens is not None and model.input_price is not None and output is not None:
        # A missing per-request surcharge is unknown, not zero.
        if model.request_price is not None:
            total = model.input_price * input_tokens + output + model.request_price
    return CostEstimate(
        total_usd=total,
        output_reservation_usd=output,
        input_tokens=input_tokens,
        method="Comptage compatible fourni, sortie maximale réservée, tarifs du catalogue."
        if input_tokens is not None
        else "Aucun compteur compatible : coût total inconnu, sans conversion caractères/tokens.",
    )


def budget_error(estimate: CostEstimate, budget: Decimal, accept_unknown: bool) -> str | None:
    if not budget.is_finite() or budget <= 0:
        return "Définissez un budget estimé strictement positif."
    if estimate.total_usd is not None:
        if estimate.total_usd > budget:
            return "L’estimation dépasse le budget : réduisez la sortie ou changez de modèle."
    elif estimate.output_reservation_usd is not None and estimate.output_reservation_usd > budget:
        return "La sortie réservée dépasse déjà le budget estimé : réduisez sa limite."
    elif not accept_unknown:
        return "Acceptez explicitement le coût incertain avant l’envoi."
    return None
