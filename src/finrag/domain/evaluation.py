from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from finrag.domain.documents import (
    DocumentId,
    NonEmptyText,
    PositiveInt,
    SourceIdentity,
    StrictRecord,
)


class Evidence(StrictRecord):
    physical_page_number: PositiveInt
    text: NonEmptyText


class FinancialScope(StrictRecord):
    kind: NonEmptyText
    label: str | None = None
    entities: list["FinancialScope"] | None = None


class FinancialPeriod(StrictRecord):
    values: list[int | str] = Field(default_factory=list)
    report_fiscal_year: int | None = None
    corpus_years: list[int] | None = None
    plan_horizon: int | str | None = None
    meaning: NonEmptyText


class SourceValue(StrictRecord):
    value: NonEmptyText
    literal: NonEmptyText
    page_number: PositiveInt
    unit: str | None
    currency: str | None
    period: int | str | None
    scope: FinancialScope
    label: NonEmptyText
    evidence_text: NonEmptyText


class CalculatedResult(StrictRecord):
    value: NonEmptyText
    unit: str | None
    currency: str | None
    interpretation: NonEmptyText


class EvaluationQuestion(StrictRecord):
    question_id: Annotated[str, Field(pattern=r"^finrag_v1_[a-z]+_[0-9]{2}$")]
    document_id: DocumentId
    company: NonEmptyText
    language: Literal["fr", "en"]
    category: Literal["factual", "numerical", "comparative", "definition_policy", "unanswerable"]
    difficulty: Literal["easy", "medium", "hard"]
    question: NonEmptyText
    reference_answer: NonEmptyText
    relevant_pages: list[PositiveInt]
    evidence: list[Evidence]
    relevant_chunk_ids: list[str]
    answerable: bool
    annotation_status: Literal["candidate", "verified"]
    split: Literal["dev", "test"]
    unit: str | None
    currency: str | None
    period: FinancialPeriod
    scope: FinancialScope
    calculation: str | None
    source_values: list[SourceValue]
    context_annotation_status: Literal["candidate"]
    evidence_bundle_pages: list[PositiveInt]
    operand_1: SourceValue | None = None
    operand_2: SourceValue | None = None
    operation: Literal["subtract"] | None = None
    calculated_result: CalculatedResult | None = None
    rounding_rule: str | None = None
    corpus_years: list[int] | None = None
    requested_year: int | None = None
    possible_confusion: str | None = None
    reason_unanswerable: str | None = None

    @model_validator(mode="after")
    def evidence_matches_answerability(self) -> Self:
        if len(self.relevant_pages) != len(set(self.relevant_pages)):
            raise ValueError("relevant_pages contains duplicates")
        if len(self.relevant_chunk_ids) != len(set(self.relevant_chunk_ids)):
            raise ValueError("relevant_chunk_ids contains duplicates")
        if len(self.evidence_bundle_pages) != len(set(self.evidence_bundle_pages)):
            raise ValueError("evidence_bundle_pages contains duplicates")
        if self.answerable:
            if not self.relevant_pages or not self.evidence or not self.evidence_bundle_pages:
                raise ValueError("answerable questions require positive evidence")
            if self.category == "unanswerable":
                raise ValueError("answerable question cannot use unanswerable category")
            if set(self.relevant_pages) != {item.physical_page_number for item in self.evidence}:
                raise ValueError("evidence does not cover exactly the relevant physical pages")
        elif (
            self.relevant_pages
            or self.evidence
            or self.relevant_chunk_ids
            or self.evidence_bundle_pages
            or self.source_values
            or self.calculation is not None
            or self.operand_1 is not None
            or self.operand_2 is not None
            or self.calculated_result is not None
            or self.operation is not None
            or self.category != "unanswerable"
        ):
            raise ValueError("unanswerable questions must not contain positive evidence")
        return self


class EvidencePageMapping(SourceIdentity):
    bundle_page_number: PositiveInt
    original_physical_page: PositiveInt
    company: NonEmptyText
    question_ids: list[NonEmptyText]
