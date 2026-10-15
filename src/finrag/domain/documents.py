from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
DocumentId = Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonEmptyText = Annotated[str, Field(min_length=1, pattern=r"\S")]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SourceIdentity(StrictRecord):
    document_id: DocumentId
    source_filename: NonEmptyText
    source_sha256: Sha256

    @field_validator("source_filename")
    @classmethod
    def filename_only(cls, value: str) -> str:
        if value in (".", "..") or any(character in value for character in ("/", "\\", ":")):
            raise ValueError("source_filename must be a filename, not a path")
        return value

    @model_validator(mode="after")
    def matching_document_hash(self) -> Self:
        if self.document_id != f"sha256:{self.source_sha256}":
            raise ValueError("document_id does not match source_sha256")
        return self


class PageRecord(SourceIdentity):
    physical_page_number: PositiveInt
    text: str
    is_blank: bool
    extraction_status: Literal["extracted", "blank"]

    @model_validator(mode="after")
    def consistent_blank_state(self) -> Self:
        blank = not self.text.strip()
        if self.is_blank != blank or (self.extraction_status == "blank") != blank:
            raise ValueError("blank status must agree with extracted text")
        return self
