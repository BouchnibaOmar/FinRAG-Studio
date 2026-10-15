import http.client
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from finrag.config import FileOperationError, FinragError, private_output
from finrag.evaluation.question_translation import timestamp
from finrag.generation import answers, citation_units
from finrag.generation.answers import AnswerContext, AnswerSettings, GroundedAnswer
from finrag.generation.diagnostics import (
    content_analysis,
    redact_private,
    response_snapshot,
    unique_object,
    usage_details,
)
from finrag.generation.openrouter import GenerationFailure, OpenRouterGenerator
from finrag.ingestion.hashing import hash_bytes
from finrag.ingestion.manifest import json_bytes, read_json, write_atomic
from finrag.retrieval.candidate_reranking import TransportReply


class GenerationTransport(Protocol):
    def request(self, payload: dict) -> TransportReply: ...


@dataclass(frozen=True)
class GenerationResult:
    answer: GroundedAnswer | citation_units.UnitAnswer
    reused: bool
    request_identity: str
    metadata: dict


def parse_reply(
    context: AnswerContext, settings: AnswerSettings, raw: object
) -> GroundedAnswer | citation_units.UnitAnswer:
    contract = citation_units if isinstance(context, citation_units.UnitContext) else answers
    try:
        if not isinstance(raw, dict) or len(raw["choices"]) != 1:
            raise ValueError
        choice = raw["choices"][0]
        message = choice["message"]
        analysis = content_analysis(context, raw)
        reason = choice["finish_reason"]
        if reason != "stop":
            raise GenerationFailure(
                "invalid_output",
                "finish_reason=length : fin complète non confirmée ; aucun retry."
                if reason == "length"
                else "Génération refusée : fin de réponse non acceptée ; aucun retry.",
                {
                    "validation_stage": "completion",
                    "validation_code": "finish_reason_not_stop",
                    "content_analysis": analysis,
                },
            )
        if message.get("refusal") or message.get("tool_calls"):
            raise GenerationFailure(
                "invalid_output",
                "Refus fournisseur ou appel d’outil inattendu ; aucun retry.",
                {"validation_stage": "completion", "validation_code": "refusal_or_tool_calls"},
            )
        if raw["model"] != settings.model:
            raise GenerationFailure(
                "invalid_output", "Le modèle retourné diffère du modèle exact demandé."
            )
        if analysis["stage"] != "complete":
            labels = {
                "content": "Contenu retourné vide, absent ou non textuel",
                "json": "JSON retourné invalide ou incomplet",
                "schema": "Schéma de réponse invalide",
                "grounding": "Citation ou règle de preuve invalide",
            }
            raise GenerationFailure(
                "invalid_output",
                labels.get(analysis["stage"], "Enveloppe de réponse invalide") + " ; aucun retry.",
                {"validation_stage": analysis["stage"], "content_analysis": analysis},
            )
        value = json.loads(message["content"], object_pairs_hook=unique_object)
        return contract.validate_answer(context, value)
    except GenerationFailure:
        raise
    except FinragError as error:
        raise GenerationFailure("invalid_output", str(error)) from error
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        raise GenerationFailure(
            "invalid_output",
            "Enveloppe OpenRouter invalide ; aucun retry.",
            {"validation_stage": "envelope", "exception_type": type(error).__name__},
        ) from error


def generate(
    project: Path,
    context: AnswerContext,
    settings: AnswerSettings,
    transport: GenerationTransport | None = None,
) -> GenerationResult:
    contract = citation_units if isinstance(context, citation_units.UnitContext) else answers
    version = "v2" if isinstance(context, citation_units.UnitContext) else "v1"
    root = private_output(project / "artifacts/private/local_generation" / version, project)
    key = contract.request_identity(context, settings)
    payload = contract.request_payload(context, settings)
    secret = os.environ.get("OPENROUTER_API_KEY")
    if secret and secret in json.dumps(payload, ensure_ascii=False):
        raise GenerationFailure(
            "configuration", "Le texte à envoyer contient la clé API configurée ; envoi refusé."
        )
    record_path = root / "records" / f"{key}.json"
    if record_path.exists():
        record = read_json(record_path)
        if (
            record["request_identity"] != key
            or record["context"] != context.model_dump(mode="json")
            or record["settings"] != settings.model_dump(mode="json", exclude_none=True)
            or record["prompt_version"] != contract.PROMPT_VERSION
            or record["metadata"]["returned_model"] != settings.model
        ):
            raise GenerationFailure("invalid_cache", "L’empreinte de la réponse en cache diffère.")
        try:
            answer = contract.validate_answer(context, record["answer"])
        except FinragError as error:
            raise GenerationFailure(
                "invalid_cache", "Les citations de la réponse en cache ne sont plus valides."
            ) from error
        return GenerationResult(answer, True, key, record["metadata"])
    attempt_path = root / "attempts" / f"{key}.json"
    if attempt_path.exists():
        raise GenerationFailure(
            "previous_attempt",
            "Cette requête a déjà été tentée sans réponse valide en cache. "
            "Aucun nouvel appel : consultez les diagnostics privés avant toute reprise.",
        )
    client = transport if transport is not None else OpenRouterGenerator(settings)
    try:
        write_atomic(
            attempt_path,
            json_bytes(
                {
                    "request_identity": key,
                    "requested_model": settings.model,
                    "prompt_version": contract.PROMPT_VERSION,
                    "timestamp": timestamp(),
                    "settings": settings.model_dump(mode="json", exclude_none=True),
                    "call_count": 1,
                }
            ),
        )
    except FileOperationError as error:
        if isinstance(error.__cause__, FileExistsError):
            raise GenerationFailure(
                "previous_attempt",
                "Cette requête est déjà en cours ou a été tentée. Aucun nouvel appel.",
            ) from error
        raise
    diagnostic = {
        "request_identity": key,
        "requested_model": settings.model,
        "timestamp": timestamp(),
        "status": "failed_api",
        "usage": None,
        "billing_unknown": True,
    }
    try:
        write_atomic(
            root / "requests" / f"{key}.json",
            json_bytes(
                redact_private(
                    {
                        "diagnostic_version": "generation-request-v1",
                        "request_identity": key,
                        "payload": payload,
                        "serialized_body_sha256": hash_bytes(
                            json.dumps(payload, ensure_ascii=False).encode("utf-8")
                        ),
                        "redaction_applied": redact_private(payload) != payload
                        or redact_private(context.model_dump(mode="json"))
                        != context.model_dump(mode="json"),
                        "context_for_offline_validation": context.model_dump(mode="json"),
                        "reasoning_parameters_sent": {
                            name: payload[name]
                            for name in ("reasoning", "include_reasoning", "reasoning_effort")
                            if name in payload
                        },
                        "reasoning_parameters_omitted": [
                            name
                            for name in ("reasoning", "include_reasoning", "reasoning_effort")
                            if name not in payload
                        ],
                        "authentication_headers_recorded": False,
                    }
                )
            ),
        )
        try:
            reply = client.request(payload)
        except (OSError, http.client.HTTPException) as error:
            raise GenerationFailure(
                "failed_api",
                "Erreur réseau ; facturation inconnue, aucun retry.",
                {
                    "exception_type": type(error).__name__,
                    "errno": getattr(error, "errno", None),
                    "timeout": isinstance(error, TimeoutError),
                },
            ) from error
        diagnostic.update(http_status=reply.status, latency_seconds=reply.latency_seconds)
        write_atomic(
            root / "responses" / f"{key}.json",
            json_bytes(response_snapshot(reply.body, reply.status, reply.latency_seconds)),
        )
        try:
            raw = json.loads(reply.body.decode("utf-8"), object_pairs_hook=unique_object)
        except (UnicodeError, ValueError) as error:
            category = "invalid_output" if reply.status == 200 else "failed_api"
            raise GenerationFailure(
                category, "Réponse OpenRouter non JSON ; aucun retry."
            ) from error
        usage = usage_details(raw)
        diagnostic.update(usage=usage, billing_unknown=not usage or "cost" not in usage)
        if isinstance(raw, dict):
            diagnostic.update({name: redact_private(raw.get(name)) for name in ("id", "model")})
            choices = raw.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                reason = choices[0].get("finish_reason")
                diagnostic["finish_reason"] = (
                    reason
                    if reason in ("stop", "length", "error", "content_filter", "tool_calls")
                    else "unknown"
                )
                diagnostic["native_finish_reason"] = redact_private(
                    choices[0].get("native_finish_reason")
                )
        if reply.status != 200 or (isinstance(raw, dict) and "error" in raw):
            # Provider messages may echo private text or credentials; retain codes, never headers.
            diagnostic["api_error_object"] = isinstance(raw, dict) and "error" in raw
            if isinstance(raw, dict) and isinstance(raw.get("error"), dict):
                code = raw["error"].get("code")
                if type(code) is int:
                    diagnostic["provider_error_code"] = code
            raise GenerationFailure(
                "failed_api", f"Erreur API OpenRouter HTTP {reply.status} ; aucun retry."
            )
        if secret and secret in json.dumps(raw, ensure_ascii=False):
            raise GenerationFailure(
                "invalid_output", "La réponse contient une donnée sensible ; sortie non conservée."
            )
        answer = parse_reply(context, settings, raw)
        metadata = {
            "requested_model": settings.model,
            "returned_model": raw["model"],
            "provider": redact_private(raw.get("provider"))
            if isinstance(raw.get("provider"), str)
            else None,
            "usage": usage,
            "latency_seconds": reply.latency_seconds,
            "timestamp": timestamp(),
        }
        record = {
            "request_identity": key,
            "context": context.model_dump(mode="json"),
            "settings": settings.model_dump(mode="json", exclude_none=True),
            "prompt_version": contract.PROMPT_VERSION,
            "answer": answer.model_dump(mode="json"),
            "metadata": metadata,
        }
        write_atomic(record_path, json_bytes(record))
        diagnostic["status"] = "success"
        return GenerationResult(answer, False, key, metadata)
    except GenerationFailure as error:
        diagnostic.update(error.diagnostic)
        diagnostic.update(status=error.category, reason=str(error))
        error.diagnostic = redact_private(diagnostic)
        raise
    except (OSError, FinragError) as error:
        diagnostic.update(status="local_error", exception_type=type(error).__name__)
        raise GenerationFailure(
            "local_error", "Erreur du cache local ; aucune nouvelle tentative automatique."
        ) from error
    finally:
        write_atomic(root / "outcomes" / f"{key}.json", json_bytes(redact_private(diagnostic)))
