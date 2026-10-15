import pytest
from test_generation import fixture_context, valid_answer

from finrag.config import FinragError
from finrag.generation.answers import build_context, validate_answer


@pytest.mark.parametrize(
    ("source", "quote"),
    [
        ("Holding\u00a0Alpha is the parent company.", "Holding Alpha is the parent company."),
        ("Holding Alpha is the parent \ncompany.", "Holding Alpha is the parent company."),
    ],
)
def test_typographically_normalized_quote_is_not_an_exact_citation(source, quote) -> None:
    _, hits, document = fixture_context()
    hits[0]["text"] = source
    context = build_context("Quelle société mère ?", hits, [document], "precomputed_BGE")
    answer = valid_answer()
    answer["affirmations"][0]["texte"] = "Holding Alpha est la société mère."
    answer["citations"][0]["extrait"] = quote
    with pytest.raises(FinragError, match="n’existe pas exactement"):
        validate_answer(context, answer)
    assert answer["citations"][0]["extrait"] == quote


def test_exact_nonbreaking_spaces_and_linebreaks_are_accepted() -> None:
    _, hits, document = fixture_context()
    source = "Holding\u00a0Alpha is the parent \ncompany."
    hits[0]["text"] = source
    context = build_context("Quelle société mère ?", hits, [document], "precomputed_BGE")
    answer = valid_answer()
    answer["affirmations"][0]["texte"] = "Holding Alpha est la société mère."
    answer["citations"][0]["extrait"] = source
    assert validate_answer(context, answer).citations[0].extrait == source


def test_one_exact_citation_does_not_silently_remove_other_invalid_citations() -> None:
    _, hits, document = fixture_context()
    hits[0]["text"] = "Holding\u00a0Alpha is the parent company."
    hits[1]["text"] = "Holding Alpha, the parent company."
    context = build_context("Quelle société mère ?", hits, [document], "precomputed_BGE")
    answer = valid_answer()
    answer["affirmations"][0]["source_ids"] = ["S1", "S2"]
    answer["citations"] = [
        {"passage_id": "S1", "extrait": "Holding Alpha is the parent company."},
        {"passage_id": "S2", "extrait": hits[1]["text"]},
    ]
    with pytest.raises(FinragError, match="n’existe pas exactement"):
        validate_answer(context, answer)
    assert len(answer["citations"]) == 2
