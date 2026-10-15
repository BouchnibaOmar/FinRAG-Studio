from types import SimpleNamespace

from finrag.evaluation.missing_candidates_audit import evidence_occurrences, vocabulary_difference


class Evidence:
    physical_page_number = 7
    text = "NET INCOME 7,074"

    def model_dump(self, mode: str) -> dict:
        return {"physical_page_number": self.physical_page_number, "text": self.text}


def test_literal_evidence_requires_document_and_page() -> None:
    q = SimpleNamespace(document_id="A", evidence=[Evidence()])
    chunks = [
        {"document_id": "A", "physical_page_number": 7, "text": "NET INCOME 7,074"},
        {"document_id": "A", "physical_page_number": 8, "text": "NET INCOME 7,074"},
        {"document_id": "B", "physical_page_number": 7, "text": "NET INCOME 7,074"},
        {"document_id": "A", "physical_page_number": 7, "text": "NET INCOME"},
    ]
    found = evidence_occurrences(q, chunks)[0]
    assert found["on_reference_page"] == chunks[:1]
    assert found["other_pages_unvalidated"] == chunks[1:2]


def test_split_excerpt_is_not_reassembled_or_normalized() -> None:
    q = SimpleNamespace(document_id="A", evidence=[Evidence()])
    chunks = [
        {"document_id": "A", "physical_page_number": 7, "text": "NET INCOME"},
        {"document_id": "A", "physical_page_number": 7, "text": "7,074"},
        {"document_id": "A", "physical_page_number": 7, "text": "NET  INCOME 7,074"},
    ]
    assert evidence_occurrences(q, chunks)[0]["on_reference_page"] == []


def test_vocabulary_uses_existing_tokenizer_without_semantic_equivalence() -> None:
    diff = vocabulary_difference("Créances douteuses 2025", "Encours douteux 2025")
    assert diff["shared"] == ["2025"]
    assert diff["query_only"] == ["creances", "douteuses"]
    assert diff["passage_only"] == ["douteux", "encours"]
