import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from pydantic import ValidationError

from finrag.config import FinragError
from finrag.retrieval.company_router import CatalogueDocument
from finrag.ui.generation_panel import clear_answer, generation_panel
from finrag.ui.model_controls import model_controls
from finrag.ui.resources import (
    FreeSearch,
    SavedExample,
    authorized_pdf_path,
    load_catalogue,
    load_examples,
    load_free_search,
    read_source_pdf,
)

PROJECT = Path(__file__).resolve().parents[3]
PRECOMPUTED_NOTICE = "Exemple précalculé : le classement n’est pas exécuté maintenant."
TRANSLATION_LABELS = {
    "openrouter": "Traduction exacte réutilisée depuis le cache OpenRouter, sans appel API.",
    "local_correction": "Traduction exacte du cache, avec correction linguistique locale assistée.",
    "no_compatible_cached_translation_original": (
        "Aucune traduction en cache ne correspond exactement à cette question et à la langue "
        "des documents. Recherche avec la question originale."
    ),
    "mixed_document_languages_original": (
        "Les documents sélectionnés ont plusieurs langues. Recherche avec la question originale."
    ),
}


@st.cache_resource(show_spinner=False)
def search_resource(project: str) -> FreeSearch:
    return load_free_search(Path(project))


@st.cache_data(show_spinner=False)
def saved_examples(project: str) -> list[SavedExample]:
    return load_examples(Path(project))


@st.cache_data(show_spinner=False)
def document_catalogue(project: str) -> list[CatalogueDocument]:
    return load_catalogue(Path(project))


def technical_error(error: Exception) -> None:
    st.error(
        "Erreur technique : les données locales n’ont pas pu être utilisées. "
        "Vérifiez leur présence et leurs droits d’accès."
    )
    with st.expander("Détail du problème"):
        st.text(f"{type(error).__name__} : {error}")


def source_viewer(hits: list[dict], catalogue: list[CatalogueDocument], key: str) -> None:
    st.subheader("Consulter une source")
    index = st.selectbox(
        "Passage à consulter",
        range(len(hits)),
        format_func=lambda i: (
            f"{i + 1} · {hits[i]['source_filename']} · "
            f"page physique {hits[i]['physical_page_number']}"
        ),
        key=f"{key}_source",
    )
    hit = hits[index]
    st.caption(f"Page physique du PDF : {hit['physical_page_number']} (comptée depuis 1).")
    if st.button("Consulter le PDF local", key=f"{key}_open"):
        st.session_state[f"{key}_pdf"] = (hit["document_id"], hit["physical_page_number"])
    selected = (hit["document_id"], hit["physical_page_number"])
    if st.session_state.get(f"{key}_pdf") != selected:
        return
    try:
        data, filename = read_source_pdf(PROJECT, catalogue, *selected)
        path = authorized_pdf_path(PROJECT, catalogue, selected[0])
    except (FinragError, OSError, ValueError) as error:
        technical_error(error)
        return
    st.download_button(
        "Télécharger le PDF pour consultation locale",
        data=data,
        file_name=filename,
        mime="application/pdf",
        key=f"{key}_download",
        on_click="ignore",
    )
    st.caption(
        "L’ouverture à la page demandée dépend du navigateur. Si le lecteur reste vide ou "
        "ouvre une autre page, utilisez le téléchargement ou ouvrez le fichier ci-dessous "
        "dans votre lecteur PDF, puis saisissez le numéro de page physique."
    )
    st.code(str(path), language=None)
    encoded = base64.b64encode(data).decode("ascii")
    page = hit["physical_page_number"]
    # Only verified PDF bytes and an integer enter this HTML; document text is never HTML.
    components.html(
        f'<iframe title="Source PDF, page physique {page}" '
        f'src="data:application/pdf;base64,{encoded}#page={page}" '
        'width="100%" height="650" style="border:0"></iframe>',
        height=660,
    )


def render_hits(
    hits: list[dict],
    catalogue: list[CatalogueDocument],
    score_field: str,
    key: str,
) -> None:
    if not hits:
        st.info("Aucun passage trouvé pour cette requête.")
        return
    with st.expander("Explorer davantage de passages", expanded=False):
        count = st.select_slider(
            "Nombre de passages affichés",
            options=[5, 10, 20, 50],
            value=5,
            key=f"{key}_depth",
        )
    st.caption(
        f"{len(hits)} passages disponibles ; affichage des {min(count, len(hits))} premiers."
    )
    companies = {doc.document_id: doc.company_family for doc in catalogue}
    for hit in hits[:count]:
        st.text(
            f"{hit['rank']}. {hit['source_filename']} · page physique {hit['physical_page_number']}"
        )
        st.caption(f"{companies[hit['document_id']]} · Aperçu du passage")
        st.text(hit["text"][:220] + ("…" if len(hit["text"]) > 220 else ""))
        with st.expander(f"Texte original complet et provenance · passage {hit['rank']}"):
            st.text(hit["text"])
            st.text(hit["chunk_id"])
            st.text(f"Document : {hit['document_id']}")
            score_type = "BM25 — score lexical" if score_field == "lexical_score" else "BGE — logit"
            st.caption(
                f"{score_type} : {hit[score_field]:.6f} · Ce score n’est pas une probabilité."
            )
        with st.expander(f"Consulter la source · passage {hit['rank']}"):
            source_viewer([hit], catalogue, f"{key}_{hit['rank']}")


def free_search(catalogue: list[CatalogueDocument]) -> None:
    st.subheader("Poser une question")
    question = st.text_area(
        "Votre question", placeholder="Saisissez une question financière…", key="free_question"
    )
    submitted = st.button("Rechercher", key="free_search_submit")
    if submitted:
        st.session_state.pop("free_result", None)
        st.session_state.pop("free_pdf", None)
        st.session_state.pop("free_depth", None)
        st.session_state.pop("free_source", None)
        clear_answer()
        if not question.strip():
            st.warning("Saisissez une question avant de rechercher.")
        else:
            try:
                with st.spinner("Recherche dans les passages locaux…"):
                    resource = search_resource(str(PROJECT))
                    result = resource.search(question)
                st.session_state["free_result"] = (question, result, resource.notices)
            except (FinragError, OSError, ValidationError, KeyError) as error:
                technical_error(error)
    if "free_result" not in st.session_state:
        return
    original_question, result, notices = st.session_state["free_result"]
    if question != original_question:
        clear_answer()
        st.info("La question a changé. Cliquez sur Rechercher pour actualiser les passages.")
        return
    for notice in notices:
        st.warning(notice)
    st.caption("Méthode : recherche par mots-clés locale.")
    with st.expander("Détails de la recherche", expanded=False):
        st.caption("BM25 recherche les termes de la question dans les passages. Aucun BGE exécuté.")
        st.caption("Provenance : recherche locale exécutée après clic sur Rechercher.")
        st.caption("Requête réellement utilisée")
        st.text(result.query_used)
        st.info(TRANSLATION_LABELS[result.translation_status])
        st.text("Entreprises détectées : " + (", ".join(result.route.companies) or "aucune"))
        st.text(
            "Documents sélectionnés : "
            + ", ".join(
                doc.source_filename
                for doc in catalogue
                if doc.document_id in result.route.document_ids
            )
        )
        st.caption(
            "Sans entreprise reconnue, recherche dans tout le corpus. "
            "Sélectionner un rapport ne garantit pas le périmètre financier d’un chiffre."
        )
    hits = [hit.model_dump(mode="json") for hit in result.hits[:50]]
    generation_panel(
        PROJECT,
        original_question,
        hits[:5],
        catalogue,
        "free_bm25",
        precalculated=False,
        source_viewer=source_viewer,
    )
    st.subheader("Sources retrouvées")
    render_hits(hits, catalogue, "lexical_score", "free")


def precomputed(catalogue: list[CatalogueDocument]) -> None:
    st.subheader("Explorer les exemples")
    st.info(PRECOMPUTED_NOTICE)
    try:
        examples = saved_examples(str(PROJECT))
    except (FinragError, OSError, ValidationError, KeyError) as error:
        technical_error(error)
        return
    index = st.selectbox(
        "Question dev sauvegardée",
        range(len(examples)),
        format_func=lambda i: f"{examples[i].question_id} · {examples[i].original_question}",
    )
    example = examples[index]
    method = st.radio(
        "Méthode de classement",
        ["BM25", "BGE"],
        horizontal=True,
        key="saved_method",
        format_func=lambda value: "Recherche par mots-clés"
        if value == "BM25"
        else "Reclassement par pertinence",
        help="BM25 : mots-clés. BGE : comparaison question/passage, déjà exécutée ici.",
    )
    hits = example.bm25 if method == "BM25" else example.bge
    with st.expander("Détails de la recherche", expanded=False):
        st.caption("Provenance : classement sauvegardé, variante cached_translation.")
        st.caption(
            "BM25 recherche les mots-clés ; BGE compare question et passage pour les réordonner."
        )
        st.caption("Requête réellement recherchée — variante cached_translation")
        st.text(example.query_used)
        st.text("Entreprises sélectionnées : " + ", ".join(example.companies))
        st.caption(f"Méthode technique : {method}. Aucun BGE exécuté maintenant.")
    generation_panel(
        PROJECT,
        example.original_question,
        hits[:5],
        catalogue,
        f"precomputed_{method}",
        precalculated=True,
        source_viewer=source_viewer,
    )
    st.subheader("Sources de l’exemple")
    render_hits(
        hits,
        catalogue,
        "lexical_score" if method == "BM25" else "bge_score",
        f"saved_{example.question_id}_{method}",
    )


def explain_workflow() -> None:
    st.subheader("Comprendre le fonctionnement")
    st.write(
        "1. L’entreprise explicitement citée oriente la recherche vers ses rapports disponibles. "
        "Sans entreprise reconnue, les passages de tout le corpus sont recherchés."
    )
    st.write(
        "2. La recherche par mots-clés (BM25) retrouve les termes de la question dans les "
        "passages. Une traduction n’est réutilisée que si elle correspond exactement au cache."
    )
    st.write(
        "3. Le reclassement par pertinence (BGE) compare la question à chaque passage pour "
        "réordonner les résultats. Il est seulement affiché depuis les exemples sauvegardés."
    )
    st.write(
        "4. Après votre clic, le modèle génératif rédige depuis la question originale et les "
        "cinq premiers passages. Ces textes sont alors envoyés à OpenRouter."
    )
    st.info(
        "Une citation traçable ne garantit pas une réponse correcte. Vérifiez la société, "
        "le périmètre, la période, la devise, l’unité et le signe dans le PDF original."
    )
    st.write(
        "Un exemple déjà analysé ne mesure pas la vitesse d’une nouvelle recherche. "
        "La génération éventuelle est une opération distincte, déclenchée explicitement."
    )
    with st.expander("Contrat de citation et limites"):
        st.write(
            "Le modèle référence des unités identifiées. L’application vérifie les IDs et "
            "leurs positions, puis affiche le texte original, le document et la page physique. "
            "Cette validation ne prouve pas que le texte justifie financièrement l’affirmation."
        )
        st.write(
            "Sans compteur de tokens compatible, aucun coût total précis n’est annoncé. "
            "L’acceptation du coût incertain ne constitue pas un plafond de facturation."
        )


def main() -> None:
    st.set_page_config(page_title="FinRAG Studio", layout="wide")
    # Keep choices when their widgets are temporarily absent in the help space.
    for key in list(st.session_state):
        if key in {
            "generation_model",
            "generation_budget",
            "generation_output_tokens",
            "saved_method",
        } or key.startswith("reasoning_"):
            st.session_state[key] = st.session_state[key]
    st.title("FinRAG Studio")
    st.write("Rechercher des passages financiers et consulter leurs sources locales.")
    st.caption("Prototype expérimental : vérifiez les sources, périodes et périmètres.")
    mode = st.sidebar.radio(
        "Espace",
        [
            "Poser une question",
            "Explorer les exemples",
            "Comprendre le fonctionnement",
        ],
        key="navigation_space",
    )
    if st.session_state.get("answer_mode") != mode:
        clear_answer()
        st.session_state["answer_mode"] = mode
    if mode == "Comprendre le fonctionnement":
        explain_workflow()
        return
    with st.sidebar:
        st.session_state["generation_choice"] = model_controls(PROJECT)
    try:
        catalogue = document_catalogue(str(PROJECT))
    except (FinragError, OSError, ValidationError) as error:
        technical_error(error)
        return
    if mode == "Poser une question":
        free_search(catalogue)
    else:
        precomputed(catalogue)


if __name__ == "__main__":
    main()
