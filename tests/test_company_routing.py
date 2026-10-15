import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_bm25 import lexical_config, synthetic_chunks
from test_retrieval_metrics import evaluation_question

from finrag.config import FinragError
from finrag.corpus.import_corpus import REPORTS
from finrag.evaluation.company_routing_experiment import run
from finrag.evaluation.document_oracle_metrics import evaluate_candidates
from finrag.evaluation.question_translation import TranslationInput, input_identity
from finrag.evaluation.translation_cache import save_translation_set
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, jsonl_bytes, read_json, read_jsonl, write_atomic
from finrag.retrieval.bm25 import BM25
from finrag.retrieval.company_router import CatalogueDocument, route_company
from finrag.retrieval.routed_bm25 import CachedQueryTranslation, RoutedBM25, select_cached_query


def corpus():
    documents, chunks = [], []
    for index, (family, entity, language) in enumerate(
        [
            ("BNP Paribas", "BNP Paribas", "en"),
            ("Crédit Agricole", "Crédit Agricole S.A.", "en"),
            ("Société Générale", "Société Générale", "fr"),
        ],
        1,
    ):
        digest = str(index) * 64
        document = CatalogueDocument(
            document_id="sha256:" + digest,
            source_sha256=digest,
            source_filename=f"report_{index}.pdf",
            company_family=family,
            reporting_entity=entity,
            language=language,
        )
        documents.append(document)
        for chunk in synthetic_chunks(["profit capital revenue", "profit profit"]):
            chunks.append(
                chunk.model_copy(
                    update={
                        "document_id": document.document_id,
                        "source_sha256": digest,
                        "source_filename": document.source_filename,
                        "chunk_id": f"{index}-{chunk.chunk_id}",
                    }
                )
            )
    return documents, chunks


@pytest.mark.parametrize(
    ("question", "company"),
    [
        ("Quel capital pour CRÉDIT AGRICOLE S.A. en 2023 et 2025 ?", "Crédit Agricole"),
        ("Que publie societe-generale ?", "Société Générale"),
        ("BNP/PARIBAS : profit ?", "BNP Paribas"),
        ("Compare 2020 et 2025 chez B.N.P. Paribas.", "BNP Paribas"),
        ("Cre\u0301dit Agricole : capital fictif ?", "Crédit Agricole"),
    ],
)
def test_synthetic_explicit_names_and_normalization(question: str, company: str) -> None:
    documents, _ = corpus()
    route = route_company(question, documents)
    assert route.status == "single_company" and route.companies == [company]
    assert route.triggering_terms[company]
    assert route.document_ids == [
        doc.document_id for doc in documents if doc.company_family == company
    ]


@pytest.mark.parametrize(
    "question",
    [
        "Quel CA et quels frais SG ?",
        "Acacia et MSG",
        "BNP Paribass",
        "xCrédit Agricole",
        "Sociétés Générales",
        "Banque inconnue : comparer 2024 et 2025",
        "profit",
        "",
    ],
)
def test_word_boundaries_unknown_names_and_ambiguous_abbreviations(question: str) -> None:
    documents, _ = corpus()
    route = route_company(question, documents)
    assert route.status == "no_company" and not route.companies
    assert set(route.document_ids) == {doc.document_id for doc in documents}


def test_union_scope_notice_and_years_do_not_filter_reports() -> None:
    documents, _ = corpus()
    second = documents[1].model_copy(
        update={
            "document_id": "sha256:" + "4" * 64,
            "source_sha256": "4" * 64,
            "source_filename": "older_report.pdf",
        }
    )
    route = route_company("BNP Paribas et Groupe Crédit Agricole, 1999 ?", documents + [second])
    assert route.status == "multiple_companies"
    assert set(route.document_ids) == {
        documents[0].document_id,
        documents[1].document_id,
        second.document_id,
    }
    sa = route_company("Crédit Agricole S.A.", documents)
    group = route_company("Groupe Crédit Agricole", documents)
    assert sa.document_ids == group.document_ids
    assert documents[1].reporting_entity == "Crédit Agricole S.A."
    assert "does not verify" in group.scope_notice


def test_router_contract_has_no_annotation_parameters() -> None:
    assert list(inspect.signature(route_company).parameters) == ["question", "catalogue"]
    assert list(inspect.signature(RoutedBM25.search).parameters) == [
        "self",
        "question",
        "use_cached_translation",
    ]
    documents, _ = corpus()
    with pytest.raises(TypeError):
        route_company("profit", documents, expected_document_id=documents[0].document_id)
    with pytest.raises(ValidationError):
        CatalogueDocument.model_validate({**documents[0].model_dump(), "relevant_pages": [1]})


def test_public_catalogue_matches_import_metadata_not_annotations() -> None:
    project = Path(__file__).resolve().parents[1]
    documents = [
        CatalogueDocument.model_validate(row)
        for row in read_json(project / "experiments/company_routing/corpus_catalog.json")
    ]
    assert len(documents) == len(REPORTS) == 3
    for document, report in zip(documents, REPORTS, strict=True):
        assert document.document_id == report.document_id
        assert document.source_sha256 == report.source_sha256
        assert document.source_filename == report.source_filename
        assert document.reporting_entity == report.company
        assert document.language == ("en" if document.source_filename.endswith("_en.pdf") else "fr")


def test_singleton_and_union_statistics_are_exact_bm25_controls() -> None:
    documents, chunks = corpus()
    engine = RoutedBM25(chunks, documents, lexical_config("B"))
    for question, selected in [
        ("BNP Paribas profit", chunks[:2]),
        ("BNP Paribas et Crédit Agricole profit", chunks[:4]),
        ("profit", chunks),
    ]:
        ranking = engine.search(question)
        control = BM25(selected, lexical_config("B"))
        assert ranking.hits == control.search(question, len(selected))
        assert ranking.corpus_statistics["average_length"] == control.average_length
        assert ranking.corpus_statistics["passages"] == len(selected)


def test_matching_translation_absent_incompatible_and_mixed_languages() -> None:
    documents, chunks = corpus()
    question = "BNP Paribas bénéfice ?"
    translated = CachedQueryTranslation(
        original_question=question,
        translated_question="BNP Paribas profit?",
        target_language="en",
        provenance="local_correction",
    )
    engine = RoutedBM25(chunks, documents, lexical_config("B"), [translated])
    result = engine.search(question, True)
    assert result.query_used == translated.translated_question
    assert result.translation_status == "local_correction"
    assert select_cached_query(question + " ", [documents[0]], [translated])[0] == question + " "
    assert select_cached_query(question, [documents[2]], [translated])[0] == question
    missing = engine.search("BNP Paribas capital ?", True)
    assert missing.query_used == "BNP Paribas capital ?"
    assert missing.translation_status == "no_compatible_cached_translation_original"
    mixed = engine.search("BNP Paribas et Société Générale bénéfice ?", True)
    assert mixed.translation_status == "mixed_document_languages_original"
    with pytest.raises(FinragError, match="Multiple cached"):
        select_cached_query(question, [documents[0]], [translated, translated])


def test_missing_report_and_invalid_catalogue_are_explicit() -> None:
    documents, chunks = corpus()
    with pytest.raises(FinragError, match="unique"):
        route_company("", documents + [documents[0]])
    with pytest.raises(FinragError, match="match exactly"):
        RoutedBM25(chunks, documents[:1], lexical_config("B"))
    engine = RoutedBM25(chunks[:2], documents[:1], lexical_config("B"))
    with pytest.raises(FinragError, match="no available report"):
        engine.search("Société Générale profit ?")


def test_output_overwrite_is_refused_before_private_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "artifacts/private/existing"
    output.mkdir(parents=True)
    with pytest.raises(FinragError, match="overwrite refused"):
        run(tmp_path, tmp_path / "absent", tmp_path / "absent_set", output)


def test_synthetic_evaluation_checks_oracle_identity_without_reference_routing(
    tmp_path: Path, monkeypatch
) -> None:
    documents, chunks = corpus()
    config = lexical_config("B")
    baseline = evaluation_question(chunks[:2])
    questions = [
        baseline.model_copy(
            update={
                "question_id": f"synthetic_dev_{index}",
                "question": f"BNP Paribas revenue {chr(65 + index)} en 2025 ?",
                "language": "fr",
            }
        )
        for index in range(28)
    ]
    # The last reference deliberately disagrees with the explicit company in the question.
    questions[-1] = questions[-1].model_copy(update={"document_id": documents[1].document_id})
    experiment = tmp_path / "artifacts/private/bm25_query_translation_dev"
    cache = experiment / "translator/cache"
    entries = []
    translated_queries = {}
    for index, question in enumerate(questions[:17]):
        source = TranslationInput(
            question=question.question, source_language="fr", target_language="en"
        )
        translated = f"BNP Paribas profit {chr(65 + index)} in 2025?"
        translated_queries[question.question_id] = translated
        path = cache / "records" / f"synthetic-{index}.json"
        write_atomic(
            path,
            json_bytes(
                {
                    "input": source.model_dump(),
                    "input_id": input_identity(source),
                    "translation": translated,
                    "source": "local_correction",
                    "requested_model": "fictional/model",
                }
            ),
        )
        entries.append(
            {"input_id": input_identity(source), "path": str(path), "sha256": hash_file(path)}
        )
    base_set = save_translation_set(cache, entries)
    original_rows, translated_rows = [], []
    for question in questions:
        index = BM25([c for c in chunks if c.document_id == question.document_id], config)
        for translated, rows in [(False, original_rows), (True, translated_rows)]:
            query = (
                translated_queries.get(question.question_id, question.question)
                if translated
                else question.question
            )
            hits = [hit.model_dump(mode="json") for hit in index.search(query, len(index.chunks))]
            rows.append(
                {
                    "question_id": question.question_id,
                    "query_used": query,
                    "hits": hits,
                    "metrics": evaluate_candidates(question, hits, True),
                }
            )
    write_atomic(
        experiment / "experiment.json", json_bytes({"source_hashes": {}, "prepared_hashes": {}})
    )
    write_atomic(experiment / "evaluation/original_top50.jsonl", jsonl_bytes(original_rows))
    write_atomic(
        experiment / "evaluation_run_2/translated_top50.jsonl", jsonl_bytes(translated_rows)
    )
    write_atomic(
        experiment / "evaluation_run_2/comparison.json",
        json_bytes({"translation_set_sha256": hash_file(base_set)}),
    )
    catalogue_path = tmp_path / "catalogue.json"
    write_atomic(catalogue_path, json_bytes([doc.model_dump() for doc in documents]))
    monkeypatch.setattr(
        "finrag.evaluation.company_routing_experiment.load_control",
        lambda project: (chunks, config, questions, {}),
    )
    output = tmp_path / "artifacts/private/routed_results"
    result = run(tmp_path, catalogue_path, base_set, output)
    rows = read_jsonl(output / "predictions_top50.jsonl")
    assert len(rows) == 28
    assert result["aggregate"]["original"]["hit@5"]["denominator"] == 28
    for row in rows[:-1]:
        assert row["original"]["oracle_top50_and_metrics_identical"] is True
        assert row["cached_translation"]["oracle_top50_and_metrics_identical"] is True
    assert rows[-1]["original"]["route"]["document_ids"] == [documents[0].document_id]
    assert rows[-1]["original"]["expected_document_present"] is False
    assert rows[-1]["original"]["oracle_identity_comparable"] is False
    assert read_json(output / "comparison.json")["routing_expected_document_present"] == 27
    assert json.loads((output / "comparison.json").read_text())["split"] == "dev"
