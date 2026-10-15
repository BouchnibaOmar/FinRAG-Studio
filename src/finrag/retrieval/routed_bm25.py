from dataclasses import dataclass
from typing import Literal

from finrag.config import FinragError
from finrag.domain.chunks import ChunkRecord
from finrag.domain.documents import NonEmptyText, StrictRecord
from finrag.domain.retrieval import LexicalConfig, RetrievalHit
from finrag.retrieval.bm25 import BM25
from finrag.retrieval.company_router import CatalogueDocument, CompanyRoute, route_company


class CachedQueryTranslation(StrictRecord):
    original_question: NonEmptyText
    translated_question: NonEmptyText
    target_language: Literal["fr", "en"]
    provenance: Literal["openrouter", "local_correction"]


@dataclass
class RoutedRanking:
    route: CompanyRoute
    query_used: str
    translation_status: str
    hits: list[RetrievalHit]
    corpus_statistics: dict


def select_cached_query(
    question: str, documents: list[CatalogueDocument], translations: list[CachedQueryTranslation]
) -> tuple[str, str]:
    languages = {doc.language for doc in documents}
    if len(languages) != 1:
        return question, "mixed_document_languages_original"
    matches = [
        row
        for row in translations
        if row.original_question == question and row.target_language in languages
    ]
    if len(matches) > 1:
        raise FinragError("Multiple cached translations match the same question and language.")
    if not matches:
        return question, "no_compatible_cached_translation_original"
    return matches[0].translated_question, matches[0].provenance


class RoutedBM25:
    def __init__(
        self,
        chunks: list[ChunkRecord],
        catalogue: list[CatalogueDocument],
        config: LexicalConfig,
        translations: list[CachedQueryTranslation] | None = None,
    ):
        route_company("", catalogue)
        if {chunk.document_id for chunk in chunks} != {doc.document_id for doc in catalogue}:
            raise FinragError("Catalogue documents and passage documents must match exactly.")
        by_id = {doc.document_id: doc for doc in catalogue}
        for chunk in chunks:
            doc = by_id[chunk.document_id]
            if (chunk.source_filename, chunk.source_sha256) != (
                doc.source_filename,
                doc.source_sha256,
            ):
                raise FinragError("Catalogue source identity differs from the passage metadata.")
        self.chunks = chunks
        self.catalogue = catalogue
        self.config = config
        self.translations = translations or []
        self.indexes: dict[tuple[str, ...], BM25] = {}

    def search(self, question: str, use_cached_translation: bool = False) -> RoutedRanking:
        route = route_company(question, self.catalogue)
        key = tuple(route.document_ids)
        if not key:
            raise FinragError("Recognized company has no available report in the catalogue.")
        if key not in self.indexes:
            self.indexes[key] = BM25(
                [chunk for chunk in self.chunks if chunk.document_id in key], self.config
            )
        index = self.indexes[key]
        query, status = question, "original_requested"
        if use_cached_translation:
            query, status = select_cached_query(
                question,
                [doc for doc in self.catalogue if doc.document_id in key],
                self.translations,
            )
        return RoutedRanking(
            route,
            query,
            status,
            index.search(query, len(index.chunks)),
            {
                "passages": len(index.chunks),
                "average_length": index.average_length,
                "total_lexical_tokens": sum(index.lengths),
            },
        )
