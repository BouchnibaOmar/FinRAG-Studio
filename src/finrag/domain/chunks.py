from typing import Literal, Self

from pydantic import Field, model_validator

from finrag.domain.documents import (
    NonEmptyText,
    NonNegativeInt,
    PositiveInt,
    Sha256,
    SourceIdentity,
    StrictRecord,
)


class ChunkingConfig(StrictRecord):
    method: Literal["fixed_window"] = "fixed_window"
    maximum_size: PositiveInt = 1250
    overlap: NonNegativeInt = 0
    unit: Literal["characters"] = "characters"
    version: Literal["characters-v1"] = "characters-v1"

    @model_validator(mode="after")
    def overlap_smaller_than_window(self) -> Self:
        if self.overlap >= self.maximum_size:
            raise ValueError("overlap must be smaller than maximum_size")
        return self


class ChunkRecord(SourceIdentity):
    chunk_id: NonEmptyText
    physical_page_number: PositiveInt
    chunk_index_on_page: NonNegativeInt
    # A whitespace-only window can occur inside a nonblank page; keep exact text.
    text: str = Field(min_length=1)
    chunking_config: ChunkingConfig
    content_sha256: Sha256
