from finrag.config import FinragError
from finrag.domain.chunks import ChunkRecord
from finrag.domain.dense import ModelChunk
from finrag.ingestion.hashing import hash_bytes
from finrag.ingestion.manifest import json_bytes
from finrag.retrieval.e5 import PREFIXES, prefixed, token_count


def split_for_model(
    parents: list[ChunkRecord],
    tokenizer,
    revision: str,
    limit: int = 512,
    overlap: int = 32,
) -> list[ModelChunk]:
    if overlap < 0 or overlap >= limit - 8:
        raise FinragError("Tokenizer-aware overlap must leave room for content and special tokens.")
    policy = {
        "revision": revision,
        "maximum_length": limit,
        "overlap_tokens": overlap,
        "version": "e5-child-v1",
        "prefix": PREFIXES["passage"],
    }
    policy_hash = hash_bytes(json_bytes(policy))
    output = []
    for parent in parents:
        base = parent.chunk_index_on_page * (
            parent.chunking_config.maximum_size - parent.chunking_config.overlap
        )
        start, index = 0, 0
        while start < len(parent.text):
            tail = parent.text[start:]
            end = len(parent.text)
            if token_count(tokenizer, tail, "passage") > limit:
                offsets = tokenizer(
                    prefixed(tail, "passage"),
                    add_special_tokens=True,
                    truncation=False,
                    return_offsets_mapping=True,
                )["offset_mapping"]
                valid = [
                    finish - len(PREFIXES["passage"])
                    for begin, finish in offsets[: limit - 1]
                    if finish > len(PREFIXES["passage"])
                ]
                if not valid:
                    raise FinragError("Tokenizer cannot yield a positive source-text window.")
                end = start + max(valid)
                while (
                    end > start
                    and token_count(tokenizer, parent.text[start:end], "passage") > limit
                ):
                    end -= 1
            if end <= start:
                raise FinragError("Tokenizer-aware splitting failed to advance.")
            text = parent.text[start:end]
            count = token_count(tokenizer, text, "passage")
            digest = hash_bytes(text.encode("utf-8"))
            whole_parent = start == 0 and end == len(parent.text)
            identifier = (
                parent.chunk_id
                if whole_parent
                else f"{parent.chunk_id}:e5-child-v1:{policy_hash}:{start}-{end}:{digest}"
            )
            output.append(
                ModelChunk(
                    **{
                        **parent.model_dump(),
                        "chunk_id": identifier,
                        "text": text,
                        "content_sha256": digest,
                    },
                    parent_chunk_id=parent.chunk_id,
                    child_index_on_parent=index,
                    source_character_start=base + start,
                    source_character_end=base + end,
                    tokenizer_revision=revision,
                    token_count=count,
                    maximum_length=limit,
                    overlap_tokens=overlap,
                    split_version="e5-child-v1",
                )
            )
            if end == len(parent.text):
                break
            next_start = end
            if overlap:
                offsets = tokenizer(
                    prefixed(text, "passage"),
                    add_special_tokens=True,
                    truncation=False,
                    return_offsets_mapping=True,
                )["offset_mapping"]
                body = [
                    (begin - len(PREFIXES["passage"]), finish - len(PREFIXES["passage"]))
                    for begin, finish in offsets
                    if finish > len(PREFIXES["passage"])
                ]
                if body:
                    next_start = start + max(0, body[max(0, len(body) - overlap)][0])
            start = max(start + 1, min(end, next_start))
            index += 1
    validate_model_chunks(parents, output, tokenizer, revision, limit, overlap)
    return output


def validate_model_chunks(
    parents: list[ChunkRecord],
    children: list[ModelChunk],
    tokenizer,
    revision: str,
    limit: int,
    overlap: int = 32,
) -> None:
    by_parent = {parent.chunk_id: parent for parent in parents}
    grouped = {identifier: [] for identifier in by_parent}
    if len({child.chunk_id for child in children}) != len(children):
        raise FinragError("Tokenizer-aware child IDs are duplicated.")
    for child in children:
        if child.parent_chunk_id not in by_parent:
            raise FinragError("Unknown child parent identity.")
        grouped[child.parent_chunk_id].append(child)
    for identifier, parent in by_parent.items():
        base = parent.chunk_index_on_page * (
            parent.chunking_config.maximum_size - parent.chunking_config.overlap
        )
        covered = 0
        for index, child in enumerate(grouped[identifier]):
            start, end = child.source_character_start - base, child.source_character_end - base
            policy = {
                "revision": revision,
                "maximum_length": limit,
                "overlap_tokens": overlap,
                "version": "e5-child-v1",
                "prefix": PREFIXES["passage"],
            }
            expected_id = (
                parent.chunk_id
                if start == 0 and end == len(parent.text)
                else (
                    f"{parent.chunk_id}:e5-child-v1:{hash_bytes(json_bytes(policy))}:"
                    f"{start}-{end}:{hash_bytes(child.text.encode('utf-8'))}"
                )
            )
            if (
                start < 0
                or start > covered
                or end <= covered
                or end > len(parent.text)
                or child.chunk_id != expected_id
                or child.overlap_tokens != overlap
                or child.chunking_config != parent.chunking_config
                or child.chunk_index_on_page != parent.chunk_index_on_page
                or child.child_index_on_parent != index
                or child.text != parent.text[start:end]
                or child.content_sha256 != hash_bytes(child.text.encode("utf-8"))
                or child.document_id != parent.document_id
                or child.source_filename != parent.source_filename
                or child.source_sha256 != parent.source_sha256
                or child.physical_page_number != parent.physical_page_number
                or child.tokenizer_revision != revision
                or child.maximum_length != limit
                or child.token_count != token_count(tokenizer, child.text, "passage")
                or child.token_count > limit
            ):
                raise FinragError(
                    "Child offsets, token limit or source provenance are inconsistent."
                )
            covered = max(covered, end)
        if covered != len(parent.text):
            raise FinragError("Tokenizer-aware children lose source characters.")
