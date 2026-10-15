import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import streamlit as st
from pydantic import ValidationError

from finrag.config import FinragError, private_output
from finrag.generation.answers import MISTRAL_LARGE_4
from finrag.generation.model_catalogue import (
    GenerationSettings,
    ModelCatalogue,
    ModelInfo,
    compatibility_errors,
    fetch_catalogue,
)
from finrag.ingestion.manifest import json_bytes, write_atomic


@dataclass(frozen=True)
class GenerationChoice:
    settings: GenerationSettings
    model: ModelInfo
    retrieved_at: str
    budget_usd: Decimal


def catalogue_directory(project: Path) -> Path:
    return project / "artifacts/private/local_ui/model_catalogues"


def read_latest_catalogue(project: Path) -> ModelCatalogue | None:
    paths = sorted(catalogue_directory(project).glob("*.json"), reverse=True)
    return (
        ModelCatalogue.model_validate_json(paths[0].read_text(encoding="utf-8")) if paths else None
    )


def store_catalogue(project: Path, catalogue: ModelCatalogue) -> Path:
    root = private_output(catalogue_directory(project), project)
    stamp = catalogue.retrieved_at.replace(":", "").replace("+", "_")
    path = root / f"{stamp}_{uuid4().hex}.json"
    write_atomic(path, json_bytes(catalogue.model_dump(mode="json")))
    return path


def price_label(value: Decimal | None) -> str:
    return (
        "non disponible"
        if value is None
        else f"{format((value * 1_000_000).normalize(), 'f')} USD / million de tokens"
    )


def model_controls(project: Path) -> GenerationChoice | None:
    if "model_catalogue" not in st.session_state:
        try:
            st.session_state["model_catalogue"] = read_latest_catalogue(project)
        except (OSError, FinragError, ValidationError):
            st.warning("Catalogue local illisible. Actualisez-le dans les paramètres avancés.")
            st.session_state["model_catalogue"] = None
    catalogue = st.session_state["model_catalogue"]
    search = st.text_input(
        "Rechercher un modèle", placeholder="Nom ou identifiant", key="model_filter"
    )
    by_id = {model.model_id: model for model in catalogue.models} if catalogue else {}
    initial = st.session_state.get("generation_model") or os.environ.get(
        "OPENROUTER_GENERATION_MODEL"
    )
    options = [
        mid
        for mid, model in by_id.items()
        if search.casefold() in (model.name + " " + mid).casefold()
    ]
    if initial and initial not in options:
        options.insert(0, initial)
    identifier = st.selectbox(
        "Modèle",
        options,
        index=options.index(initial) if initial in options else None,
        format_func=lambda mid: by_id[mid].name if mid in by_id else mid,
        key="generation_model",
        placeholder="Choisir un modèle",
    )
    budget = Decimal(
        str(
            st.number_input(
                "Budget estimé maximal par appel (USD)",
                min_value=0.000001,
                value=0.05,
                step=0.01,
                format="%.6f",
                key="generation_budget",
                help="Budget estimé, sans garantie de plafond de facturation.",
            )
        )
    )
    model = by_id.get(identifier)
    efforts = model.efforts if model else []
    default = "none" if identifier == MISTRAL_LARGE_4 and "none" in efforts else "omis"
    with st.expander("Paramètres avancés", expanded=False):
        max_output = int(
            st.number_input(
                "Limite de sortie (tokens)",
                min_value=1,
                max_value=32768,
                value=1024,
                step=128,
                key="generation_output_tokens",
            )
        )
        reasoning = st.selectbox(
            "Effort de raisonnement",
            ["omis", *efforts],
            index=(["omis", *efforts]).index(default),
            key=f"reasoning_{identifier}",
            help="Omis : paramètres par défaut du fournisseur. Aucun effort n’est inventé.",
        )
        if catalogue:
            st.caption(f"Catalogue récupéré le {catalogue.retrieved_at} · prix en USD")
        if model:
            st.text(f"Identifiant technique : {model.model_id}")
            st.caption("Entrée : " + price_label(model.input_price))
            st.caption("Sortie : " + price_label(model.output_price))
            st.caption(f"Contexte : {model.context_length or 'non disponible'} tokens")
            st.caption(
                f"Sortie maximale déclarée : {model.max_output_tokens or 'non disponible'} tokens"
            )
            st.caption(
                "JSON structuré déclaré : "
                + ("oui" if "structured_outputs" in model.parameters else "non disponible")
            )
            st.text("Paramètres déclarés : " + (", ".join(model.parameters) or "non disponible"))
            st.json(model.reasoning or {"raisonnement": "non disponible"})
            st.link_button("Fiche officielle du modèle", model.url)
            st.caption(
                "Déclarations du catalogue : aucune garantie d’exécution par le fournisseur."
            )
        if st.button("Actualiser le catalogue", key="refresh_models"):
            try:
                updated = fetch_catalogue()
                store_catalogue(project, updated)
                st.session_state["model_catalogue"] = updated
                st.rerun()
            except (FinragError, OSError, ValidationError) as error:
                st.error(
                    f"Catalogue indisponible ({type(error).__name__}). Réessayez l’actualisation."
                )
        if st.button("Recharger les ressources locales"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.session_state.pop("generated_answer", None)
            for name in list(st.session_state):
                if name == "free_result" or name.endswith("_pdf"):
                    del st.session_state[name]
            st.rerun()
    if model is None or catalogue is None:
        st.info("Choisissez un modèle du catalogue. Son actualisation est dans Paramètres avancés.")
        return None
    errors = compatibility_errors(model, max_output)
    for reason in errors:
        st.warning("Génération indisponible : " + reason)
    if errors:
        return None
    try:
        settings = GenerationSettings(
            model=identifier,
            max_completion_tokens=max_output,
            reasoning_effort=None if reasoning == "omis" else reasoning,
            supported_reasoning_efforts=efforts,
        )
    except ValidationError:
        st.warning(
            "Identifiant ou paramètres incompatibles. Choisissez un modèle exact compatible."
        )
        return None
    return GenerationChoice(settings, model, catalogue.retrieved_at, budget)
