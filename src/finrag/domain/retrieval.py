from typing import Literal

from pydantic import Field

from finrag.domain.chunks import ChunkingConfig
from finrag.domain.documents import (
    DocumentId,
    NonEmptyText,
    PositiveInt,
    SourceIdentity,
    StrictRecord,
)


class LexicalConfig(StrictRecord):
    configuration_id: Literal["A", "B", "C"]
    chunking: ChunkingConfig
    k1: float = Field(default=1.2, gt=0, allow_inf_nan=False)
    b: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    tokenizer_version: Literal["unicode-financial-v1"] = "unicode-financial-v1"
    top_k: Literal[10] = 10


class RetrievalHit(SourceIdentity):
    rank: PositiveInt
    lexical_score: float = Field(ge=0, allow_inf_nan=False)
    physical_page_number: PositiveInt
    chunk_id: NonEmptyText
    text: str
    configuration_id: Literal["A", "B", "C"]


class PageTarget(StrictRecord):
    document_id: DocumentId
    physical_page_number: PositiveInt


class LexicalPrediction(StrictRecord):
    question_id: NonEmptyText
    split: Literal["dev"]
    configuration_id: Literal["A", "B", "C"]
    answerable: bool
    expected_pages: list[PageTarget]
    retrieved: list[RetrievalHit]
