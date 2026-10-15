from dataclasses import dataclass
from typing import Literal

from finrag.config import FinragError
from finrag.domain.documents import NonEmptyText, PageRecord, PositiveInt, SourceIdentity

CONTEXT_CHARACTER_LIMIT = 20_000
CONTEXT_SEPARATOR = "\n\n"


class ContextPassage(SourceIdentity):
    chunk_id: NonEmptyText
    physical_page_number: PositiveInt
    text: str


@dataclass(frozen=True)
class ConstructedContext:
    mode: str
    text: str
    units: list[dict]
    represented_chunk_ids: list[str]
    omitted_units: list[dict]
    stop_reason: str | None
    first_unit_exceeds_limit: bool


def build_context(
    passages: list[ContextPassage],
    pages: dict[tuple[str, int], PageRecord],
    mode: Literal["passages", "pages"],
) -> ConstructedContext:
    if mode not in ("passages", "pages"):
        raise FinragError("Context mode must be passages or pages.")
    proposed, seen = [], set()
    for passage in passages:
        key = (passage.document_id, passage.physical_page_number)
        if key not in pages:
            raise FinragError(f"Extracted page missing for {key}.")
        page = pages[key]
        if (
            page.document_id != passage.document_id
            or page.physical_page_number != passage.physical_page_number
            or page.source_sha256 != passage.source_sha256
            or page.source_filename != passage.source_filename
            or passage.text not in page.text
        ):
            raise FinragError(f"Passage/page text or provenance mismatch: {passage.chunk_id}.")
        identity = key if mode == "pages" else passage.chunk_id
        if identity in seen:
            continue
        seen.add(identity)
        proposed.append(
            page.model_dump(mode="json") if mode == "pages" else passage.model_dump(mode="json")
        )
    accepted, omitted = [], []
    characters, reason = 0, None
    first_oversized = False
    for index, unit in enumerate(proposed):
        added = len(unit["text"]) + (len(CONTEXT_SEPARATOR) if accepted else 0)
        if characters + added > CONTEXT_CHARACTER_LIMIT:
            reason = "next_whole_unit_exceeds_character_limit"
            first_oversized = index == 0 and len(unit["text"]) > CONTEXT_CHARACTER_LIMIT
            omitted = [
                {key: value for key, value in row.items() if key != "text"}
                | {
                    "text_characters": len(row["text"]),
                    "reason": reason if offset == 0 else "not_added_after_budget_stop",
                }
                for offset, row in enumerate(proposed[index:])
            ]
            break
        accepted.append(unit)
        characters += added
    text = CONTEXT_SEPARATOR.join(unit["text"] for unit in accepted)
    assert len(text) == characters <= CONTEXT_CHARACTER_LIMIT
    represented = []
    for passage in passages:
        if (
            any(
                unit["document_id"] == passage.document_id
                and unit["physical_page_number"] == passage.physical_page_number
                and (mode == "pages" or unit["chunk_id"] == passage.chunk_id)
                for unit in accepted
            )
            and passage.chunk_id not in represented
        ):
            represented.append(passage.chunk_id)
    return ConstructedContext(mode, text, accepted, represented, omitted, reason, first_oversized)
