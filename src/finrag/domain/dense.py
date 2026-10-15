from typing import Literal, Self

from pydantic import Field, model_validator

from finrag.domain.chunks import ChunkRecord
from finrag.domain.documents import (
    NonEmptyText,
    NonNegativeInt,
    PositiveInt,
    Sha256,
    SourceIdentity,
    StrictRecord,
)
from finrag.domain.retrieval import PageTarget


class ModelChunk(ChunkRecord):
    parent_chunk_id: NonEmptyText
    child_index_on_parent: NonNegativeInt
    source_character_start: NonNegativeInt
    source_character_end: PositiveInt
    tokenizer_revision: NonEmptyText
    token_count: PositiveInt
    maximum_length: PositiveInt
    overlap_tokens: NonNegativeInt
    split_version: Literal["e5-child-v1"]


class DenseHit(SourceIdentity):
    rank: PositiveInt
    score: float = Field(allow_inf_nan=False)
    physical_page_number: PositiveInt
    chunk_id: NonEmptyText
    configuration_id: NonEmptyText
    text: str
    component_ranks: dict[str, PositiveInt] = Field(default_factory=dict)


class DensePrediction(StrictRecord):
    question_id: NonEmptyText
    split: Literal["dev"]
    configuration_id: NonEmptyText
    method: Literal["bm25", "dense", "hybrid"]
    answerable: bool
    expected_pages: list[PageTarget]
    retrieved: list[DenseHit]


class FaissInput(StrictRecord):
    path: NonEmptyText
    sha256: Sha256


class FaissRequest(StrictRecord):
    root: NonEmptyText
    fingerprint: dict
    inputs: dict[Literal["chunks", "passages", "queries"], FaissInput]
    query_count: PositiveInt
    top_k: Literal[50]
    threads: Literal[4]

    @model_validator(mode="after")
    def complete_inputs(self) -> Self:
        if set(self.inputs) != {"chunks", "passages", "queries"}:
            raise ValueError("FAISS request requires chunks, passages and queries")
        return self
