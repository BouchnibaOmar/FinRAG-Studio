import json
import os
from collections.abc import Callable
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from finrag.config import FinragError
from finrag.generation.answers import REFUSAL
from finrag.generation.cache import GenerationResult, generate
from finrag.generation.citation_units import (
    UnitContext,
    build_context,
    request_identity,
    request_payload,
    resolve_citations,
)
from finrag.generation.model_catalogue import budget_error, estimate_cost
from finrag.generation.openrouter import GenerationFailure
from finrag.retrieval.company_router import CatalogueDocument
from finrag.ui.model_controls import price_label


def clear_answer() -> None:
    st.session_state.pop("generated_answer", None)


def generation_panel(
    project: Path,
    question: str,
    hits: list[dict],
    catalogue: list[CatalogueDocument],
    ranking: str,
    precalculated: bool,
    source_viewer: Callable[[list[dict], list[CatalogueDocument], str], None],
) -> None:
    if not hits:
        clear_answer()
        st.info("Aucun passage disponible pour la génération.")
        return
    choice = st.session_state.get("generation_choice")
    if choice is None:
        clear_answer()
        st.info("Choisissez un modèle compatible dans le panneau latéral pour rédiger une réponse.")
        st.button("Générer une réponse", disabled=True)
        return
    settings = choice.settings
    try:
        context = build_context(question, hits, catalogue, ranking)
        identity = request_identity(context, settings)
        payload = request_payload(context, settings)
    except (FinragError, ValidationError, KeyError) as error:
        clear_answer()
        st.error(f"Configuration de génération invalide : {type(error).__name__}.")
        return
    current = st.session_state.get("generated_answer")
    if current is not None and current.request_identity != identity:
        clear_answer()
        current = None
    estimate = estimate_cost(choice.model, settings.max_completion_tokens)
    st.caption(f"Modèle : {choice.model.name} · {len(context.passages)} passages envoyés")
    if estimate.total_usd is None:
        st.write("Coût total inconnu")
        st.caption("Le coût total ne peut pas être estimé précisément.")
    else:
        st.write(f"Coût estimé : {estimate.total_usd:f} USD")
    st.caption("La question et les passages seront transmis à OpenRouter.")
    accepted = (
        st.checkbox(
            "J’accepte le coût variable de cet appel OpenRouter.",
            value=False,
            key=f"cost_accept_{identity}_{choice.retrieved_at}_{choice.budget_usd}",
        )
        if estimate.total_usd is None
        else False
    )
    blocked = budget_error(estimate, choice.budget_usd, accepted)
    if st.button("Générer une réponse", key="generate_answer", disabled=bool(blocked)):
        clear_answer()
        current = None
        try:
            with st.spinner("Génération à partir des passages sélectionnés…"):
                current = generate(project, context, settings)
            st.session_state["generated_answer"] = current
        except GenerationFailure as error:
            labels = {
                "failed_api": "Erreur API ou réseau",
                "invalid_output": "Sortie rejetée par la validation",
                "configuration": "Configuration incomplète",
                "previous_attempt": "Tentative antérieure conservée",
                "local_error": "Erreur du cache local",
                "invalid_cache": "Cache invalide",
            }
            st.error(f"{labels.get(error.category, 'Erreur de génération')} : {error}")
            st.caption(
                "Consultez les diagnostics privés avant toute nouvelle tentative. Aucun retry."
            )
            display_usage(error.diagnostic, "Tentative rejetée ou échouée")
        except (FinragError, OSError, KeyError) as error:
            st.error(f"Erreur de lecture ou écriture du cache privé : {type(error).__name__}.")
    if blocked:
        st.caption("Bouton désactivé : " + blocked)
    if current is not None:
        st.subheader("Réponse")
        render_answer(current, context, hits, catalogue, source_viewer, identity)
    with st.expander("Détails de l’envoi et du coût", expanded=False):
        st.caption(f"Tarifs du {choice.retrieved_at}")
        st.caption("Entrée : " + price_label(choice.model.input_price))
        st.caption("Sortie : " + price_label(choice.model.output_price))
        st.caption("Tarifs du catalogue : variations possibles selon le fournisseur.")
        st.caption(estimate.method)
        if estimate.output_reservation_usd is not None:
            st.caption(
                f"Montant réservé pour la sortie : {estimate.output_reservation_usd:f} USD "
                "(partiel, hors entrée et frais éventuels)."
            )
        st.caption("Le budget estimé n’est pas un plafond de facturation garanti.")
        st.caption("Sans compteur compatible, la taille en tokens du contexte reste inconnue.")
        st.caption(f"Sortie maximale : {settings.max_completion_tokens} tokens")
        if settings.reasoning_effort is not None:
            st.caption(f"Configuration envoyée : reasoning.effort={settings.reasoning_effort}.")
        else:
            st.caption("Raisonnement : paramètre omis, valeurs par défaut du fournisseur.")
        st.json(
            {
                "schéma": payload["response_format"],
                "paramètres": settings.model_dump(mode="json", exclude_none=True),
                "fournisseur": payload["provider"],
            }
        )
        st.caption("Contrat de citation v2 : texte original et positions exactes conservés.")
        st.text(question)
        for passage in context.passages:
            st.text(f"{passage.passage_id} · page physique {passage.physical_page_number}")
            st.text(passage.text)
        content = [message["content"] for message in payload["messages"]]
        st.text(
            f"{len(context.passages)} passages · "
            f"{sum(len(p.text) for p in context.passages)} caractères de passages · "
            f"{sum(map(len, content))} caractères Unicode de messages."
        )
        body = json.dumps(payload, ensure_ascii=False)
        st.caption(
            f"Corps JSON complet : {len(body)} caractères, "
            f"{len(body.encode('utf-8'))} octets UTF-8, schéma de sortie inclus."
        )
        if not os.environ.get("OPENROUTER_API_KEY"):
            st.caption("Clé OPENROUTER_API_KEY absente : seul le cache peut être réutilisé.")


def render_answer(
    current: GenerationResult,
    context: UnitContext,
    hits: list[dict],
    catalogue: list[CatalogueDocument],
    source_viewer: Callable[[list[dict], list[CatalogueDocument], str], None],
    identity: str,
) -> None:
    st.caption(
        "Réponse réutilisée depuis le cache privé."
        if current.reused
        else "Réponse conservée dans cette session et dans le cache privé."
    )
    answer = current.answer
    if answer.statut == "preuves_insuffisantes":
        st.info(REFUSAL)
    else:
        with st.container(border=True):
            for claim in answer.affirmations:
                st.text(claim.texte + " [" + ", ".join(claim.unit_ids) + "]")
    by_chunk = {hit["chunk_id"]: hit for hit in hits[:5]}
    cited_hits = []
    for citation in resolve_citations(context, answer):
        hit = by_chunk[citation.chunk_id]
        st.caption(
            f"{citation.unit_id} · {hit['source_filename']} · "
            f"page physique {citation.physical_page_number}"
        )
        st.caption(
            "Aperçu de citation : "
            + citation.text[:160]
            + ("…" if len(citation.text) > 160 else "")
        )
        with st.expander(f"Citation originale complète · {citation.unit_id}"):
            st.text(citation.text)
            st.text(hit["chunk_id"])
            st.text(f"Tranche Unicode [{citation.start}, {citation.end}) du passage original.")
        if hit not in cited_hits:
            cited_hits.append(hit)
    st.caption("Une citation traçable ne garantit pas une réponse correcte. Vérifiez les sources.")
    if cited_hits:
        with st.expander("Consulter les PDF cités", expanded=False):
            source_viewer(cited_hits, catalogue, f"answer_{identity}")
    display_usage(current.metadata, "Réponse en cache" if current.reused else "Nouvelle réponse")


def display_usage(metadata: dict, provenance: str) -> None:
    usage = metadata.get("usage") or {}
    with st.expander("Détails de la génération et coût déclaré"):
        st.text(provenance)
        for label, value in (
            ("Modèle demandé", metadata.get("requested_model")),
            ("Modèle retourné", metadata.get("returned_model") or metadata.get("model")),
            ("Fournisseur", metadata.get("provider")),
            ("Tokens d’entrée", usage.get("prompt_tokens")),
            ("Tokens de sortie", usage.get("completion_tokens")),
            ("Coût déclaré (USD)", usage.get("cost")),
            ("Durée de l’appel initial (secondes)", metadata.get("latency_seconds")),
        ):
            st.text(f"{label} : {value if value is not None else 'non disponible'}")
