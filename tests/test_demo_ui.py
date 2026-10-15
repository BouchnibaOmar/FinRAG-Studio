import http.client

import pytest
from streamlit.testing.v1 import AppTest
from ui_fixtures import synthetic_catalogue

from finrag.generation.model_catalogue import ModelCatalogue
from finrag.ui import app, model_controls


@pytest.fixture(autouse=True)
def offline_catalogue(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected network or inference")

    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden)
    monkeypatch.setattr(model_controls, "fetch_catalogue", forbidden)
    monkeypatch.setattr(model_controls, "read_latest_catalogue", lambda _: synthetic_catalogue())
    monkeypatch.setenv("OPENROUTER_GENERATION_MODEL", "fictional/generator")
    monkeypatch.setenv("OPENROUTER_API_KEY", "never-display-this-key")


def controls_ui(tmp_path):
    script = tmp_path / "models.py"
    script.write_text(
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from finrag.ui.model_controls import model_controls\n"
        "with st.sidebar:\n"
        "    choice = model_controls(Path.cwd())\n"
        "st.session_state['choice'] = choice\n",
        encoding="utf-8",
    )
    return AppTest.from_file(str(script), default_timeout=15).run()


def test_model_selection_search_parameters_and_unknown_initial_model(tmp_path, monkeypatch):
    ui = controls_ui(tmp_path)
    assert not ui.exception
    assert ui.session_state["choice"].settings.model == "fictional/generator"
    assert ui.session_state["choice"].settings.reasoning_effort is None
    ui.text_input(key="model_filter").input("other").run()
    assert "fictional/other" in ui.selectbox(key="generation_model").options[1]
    ui.selectbox(key="generation_model").set_value("fictional/other").run()
    ui.number_input(key="generation_output_tokens").set_value(2048).run()
    assert ui.session_state["choice"].settings.model == "fictional/other"
    assert ui.session_state["choice"].settings.max_completion_tokens == 2048
    ui.run()
    assert not ui.exception
    assert not any("never-display-this-key" in text.value for text in ui.text)
    monkeypatch.setenv("OPENROUTER_GENERATION_MODEL", "fictional/absent")
    fresh = controls_ui(tmp_path)
    assert fresh.session_state["choice"] is None
    assert fresh.selectbox(key="generation_model").value == "fictional/absent"


def test_incompatible_model_and_missing_catalogue_never_fall_back(tmp_path, monkeypatch):
    models = synthetic_catalogue().models
    incompatible = models[0].model_copy(update={"parameters": []})
    catalogue = ModelCatalogue(retrieved_at="2026-10-15", models=[incompatible, models[1]])
    monkeypatch.setattr(model_controls, "read_latest_catalogue", lambda _: catalogue)
    ui = controls_ui(tmp_path)
    assert ui.session_state["choice"] is None
    assert any("structured_outputs" in warning.value for warning in ui.warning)
    ui.selectbox(key="generation_model").set_value("fictional/other").run()
    assert ui.session_state["choice"].settings.model == "fictional/other"
    monkeypatch.setattr(model_controls, "read_latest_catalogue", lambda _: None)
    fresh = controls_ui(tmp_path)
    assert not fresh.exception and fresh.session_state["choice"] is None
    assert fresh.button(key="refresh_models")


def test_explanations_do_not_read_corpus_or_run_network(tmp_path, monkeypatch):
    def unavailable(*args):
        raise FileNotFoundError("Synthetic catalogue unavailable")

    monkeypatch.setattr(app, "document_catalogue", unavailable)
    script = tmp_path / "explain.py"
    script.write_text("from finrag.ui.app import main\nmain()", encoding="utf-8")
    ui = AppTest.from_file(str(script), default_timeout=15).run()
    ui.sidebar.radio[0].set_value("Comprendre le fonctionnement").run()
    assert not ui.exception and not ui.error
    assert any("citation traçable" in message.value for message in ui.info)
    assert not ui.get("generate_answer")


def test_advanced_controls_are_closed_and_keep_model_and_budget(tmp_path):
    ui = controls_ui(tmp_path)
    advanced = next(e for e in ui.expander if e.label == "Paramètres avancés")
    assert advanced.proto.expanded is False
    assert advanced.number_input(key="generation_output_tokens")
    assert advanced.selectbox(key="reasoning_fictional/generator")
    assert advanced.button(key="refresh_models")
    ui.number_input(key="generation_budget").set_value(0.12).run()
    ui.number_input(key="generation_output_tokens").set_value(2048).run()
    ui.run()
    assert ui.session_state["choice"].settings.model == "fictional/generator"
    assert str(ui.session_state["choice"].budget_usd) == "0.12"
    assert ui.session_state["choice"].settings.max_completion_tokens == 2048


def test_consent_and_saved_bge_response_survive_visual_reruns(tmp_path, monkeypatch):
    from test_generation import fixture_context
    from test_generation_units import unit_answer
    from ui_fixtures import accept_uncertain_cost

    from finrag.generation.cache import GenerationResult
    from finrag.generation.citation_units import UnitAnswer, request_identity
    from finrag.ui import generation_panel as panel
    from finrag.ui.resources import SavedExample

    original, hits, doc = fixture_context()
    for hit in hits:
        hit.update(lexical_score=1.0, bge_score=2.0)
    example = SavedExample("synthetic", original.question, "query?", ["BNP Paribas"], hits, hits)
    monkeypatch.setattr(app, "document_catalogue", lambda _: [doc])
    monkeypatch.setattr(app, "saved_examples", lambda _: [example])
    calls = []

    def cached(project, context, settings):
        calls.append(context)
        return GenerationResult(
            UnitAnswer.model_validate(unit_answer()),
            True,
            request_identity(context, settings),
            {"returned_model": settings.model},
        )

    monkeypatch.setattr(panel, "generate", cached)
    script = tmp_path / "compact.py"
    script.write_text("from finrag.ui.app import main\nmain()", encoding="utf-8")
    ui = AppTest.from_file(str(script), default_timeout=15).run()
    ui.sidebar.radio[0].set_value("Explorer les exemples").run()
    ui.radio(key="saved_method").set_value("BGE").run()
    checkbox = next(c for c in ui.checkbox if c.key.startswith("cost_accept_"))
    assert checkbox.label == "J’accepte le coût variable de cet appel OpenRouter."
    assert checkbox.value is False and ui.button(key="generate_answer").disabled
    assert ui.button(key="generate_answer").label == "Générer une réponse"
    for label in (
        "Détails de la recherche",
        "Détails de l’envoi et du coût",
        "Explorer davantage de passages",
    ):
        assert next(e for e in ui.expander if e.label == label).proto.expanded is False
    assert ui.select_slider[0].value == 5
    assert not calls
    accept_uncertain_cost(ui)
    assert not ui.button(key="generate_answer").disabled
    ui.button(key="generate_answer").click().run()
    assert not ui.exception and len(calls) == 1
    assert any("cache privé" in c.value for c in ui.caption)
    assert any("S1.U001" in c.value for c in ui.caption)
    assert any(b.label == "Consulter le PDF local" for b in ui.button)
    ui.select_slider[0].set_value(20).run()
    ui.number_input(key="generation_budget").set_value(0.12).run()
    ui.run()
    assert len(calls) == 1 and ui.radio(key="saved_method").value == "BGE"
    assert ui.selectbox(key="generation_model").value == "fictional/generator"
    assert any(t.value.startswith("Le revenu 2025 est") for t in ui.text)
    assert [p.chunk_id for p in calls[0].passages] == [h["chunk_id"] for h in hits[:5]]
    ui.sidebar.radio[0].set_value("Comprendre le fonctionnement").run()
    ui.sidebar.radio[0].set_value("Explorer les exemples").run()
    assert ui.radio(key="saved_method").value == "BGE"
    assert ui.selectbox(key="generation_model").value == "fictional/generator"
    assert ui.number_input(key="generation_budget").value == 0.12
    assert len(calls) == 1


@pytest.mark.parametrize("scenario", ["error", "insufficient"])
def test_generation_failure_and_refusal_remain_distinct(tmp_path, monkeypatch, scenario):
    from ui_fixtures import accept_uncertain_cost

    from finrag.generation.answers import REFUSAL
    from finrag.generation.cache import GenerationResult
    from finrag.generation.citation_units import UnitAnswer, request_identity
    from finrag.generation.openrouter import GenerationFailure
    from finrag.ui import generation_panel as panel

    calls = []

    def simulated(project, context, settings):
        calls.append(context)
        if scenario == "error":
            raise GenerationFailure("failed_api", "Erreur réseau simulée ; aucun retry.")
        answer = UnitAnswer(
            statut="preuves_insuffisantes",
            affirmations=[],
            comparaison=False,
            operandes=[],
        )
        return GenerationResult(
            answer, True, request_identity(context, settings), {"returned_model": settings.model}
        )

    monkeypatch.setattr(panel, "generate", simulated)
    script = tmp_path / "outcome.py"
    script.write_text(
        "from pathlib import Path\n"
        "import streamlit as st\n"
        "from finrag.ui.model_controls import model_controls\n"
        "from finrag.ui.generation_panel import generation_panel\n"
        "from test_generation import fixture_context\n"
        "ctx, hits, doc = fixture_context()\n"
        "with st.sidebar:\n"
        "    st.session_state['generation_choice'] = model_controls(Path.cwd())\n"
        "generation_panel(Path.cwd(), ctx.question, hits, [doc], 'precomputed_BGE', True, "
        "lambda *args: None)\n",
        encoding="utf-8",
    )
    ui = AppTest.from_file(str(script), default_timeout=15).run()
    assert not calls
    accept_uncertain_cost(ui)
    ui.button(key="generate_answer").click().run()
    assert len(calls) == 1 and not ui.exception
    if scenario == "error":
        assert any("Erreur API ou réseau" in error.value for error in ui.error)
        assert not any(message.value == REFUSAL for message in ui.info)
    else:
        assert not ui.error and any(message.value == REFUSAL for message in ui.info)
    ui.run()
    assert len(calls) == 1
