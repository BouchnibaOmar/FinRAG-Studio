import http.client
import json
import os
import time

from finrag.config import FinragError
from finrag.generation.answers import AnswerSettings
from finrag.retrieval.candidate_reranking import TransportReply


class GenerationFailure(FinragError):
    def __init__(self, category: str, message: str, diagnostic: dict | None = None):
        super().__init__(message)
        self.category = category
        self.diagnostic = diagnostic or {}


class OpenRouterGenerator:
    def __init__(self, settings: AnswerSettings):
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise GenerationFailure("configuration", "OPENROUTER_API_KEY n’est pas configurée.")
        self.settings = settings

    def request(self, payload: dict) -> TransportReply:
        if payload.get("model") != self.settings.model:
            raise GenerationFailure("configuration", "Le modèle de génération demandé diffère.")
        connection = http.client.HTTPSConnection(
            "openrouter.ai", timeout=self.settings.timeout_seconds
        )
        start = time.monotonic()
        try:
            connection.request(
                "POST",
                "/api/v1/chat/completions",
                body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                },
            )
            response = connection.getresponse()
            body = response.read(1048577)
            if len(body) > 1048576:
                raise GenerationFailure(
                    "failed_api",
                    "Réponse OpenRouter trop volumineuse ; aucun retry.",
                    {"http_status": response.status, "response_exceeds_1_mib": True},
                )
            return TransportReply(response.status, body, time.monotonic() - start)
        except (OSError, http.client.HTTPException) as error:
            raise GenerationFailure(
                "failed_api",
                "Erreur réseau OpenRouter ; facturation inconnue, aucun retry.",
                {
                    "exception_type": type(error).__name__,
                    "errno": getattr(error, "errno", None),
                    "timeout": isinstance(error, TimeoutError),
                    "latency_seconds": time.monotonic() - start,
                },
            ) from error
        finally:
            connection.close()
