from datetime import datetime
from typing import Literal, Self

from pydantic import model_validator

from finrag.domain.documents import (
    NonEmptyText,
    NonNegativeInt,
    PositiveInt,
    Sha256,
    SourceIdentity,
    StrictRecord,
)


class DocumentManifest(SourceIdentity):
    byte_size: PositiveInt
    physical_page_count: PositiveInt
    text_page_count: PositiveInt
    blank_page_count: NonNegativeInt
    extractor_name: Literal["pypdf"]
    extractor_version: NonEmptyText
    extraction_timestamp: datetime
    extraction_status: Literal["extracted"]

    @model_validator(mode="after")
    def page_counts_and_timestamp(self) -> Self:
        if self.text_page_count + self.blank_page_count != self.physical_page_count:
            raise ValueError("text and blank page counts must cover every physical page")
        if self.extraction_timestamp.tzinfo is None:
            raise ValueError("extraction_timestamp must have a timezone")
        return self


class DatasetCounts(StrictRecord):
    questions: NonNegativeInt
    dev: NonNegativeInt
    test: NonNegativeInt
    answerable: NonNegativeInt
    unanswerable: NonNegativeInt
    candidate: NonNegativeInt
    verified: NonNegativeInt
    evidence_pages: NonNegativeInt
    evidence_excerpts: NonNegativeInt


class Transformation(StrictRecord):
    question_id: NonEmptyText
    original_value: NonEmptyText
    new_value: NonEmptyText
    reason: NonEmptyText
    version: NonEmptyText


class DatasetManifest(StrictRecord):
    dataset_name: Literal["finrag"]
    dataset_version: Literal["finrag_v1"]
    schema_version: Literal["1.0"]
    creation_timestamp: datetime
    review_status: Literal["externally_reviewed_ai_assisted"]
    counts: DatasetCounts
    source_artifacts: dict[str, str]
    source_hashes: dict[str, Sha256]
    imported_hashes: dict[str, Sha256]
    transformation_history: list[Transformation]
    limitations: list[NonEmptyText]
    prototype_baseline_reference: str

    @model_validator(mode="after")
    def timezone_required(self) -> Self:
        if self.creation_timestamp.tzinfo is None:
            raise ValueError("creation_timestamp must have a timezone")
        return self
