from finrag.domain.chunks import ChunkingConfig, ChunkRecord
from finrag.domain.documents import PageRecord
from finrag.ingestion.hashing import hash_bytes
from finrag.ingestion.manifest import json_bytes, validate_page_sequence


def chunk_pages(pages: list[PageRecord], config: ChunkingConfig) -> list[ChunkRecord]:
    validate_page_sequence(pages)
    config_hash = hash_bytes(json_bytes(config.model_dump(mode="json")))
    chunks = []
    step = config.maximum_size - config.overlap
    for page in pages:
        if page.is_blank:
            continue
        position = 0
        index = 0
        while position < len(page.text):
            end = min(position + config.maximum_size, len(page.text))
            text = page.text[position:end]
            digest = hash_bytes(text.encode("utf-8"))
            identity = (
                f"{page.document_id}:p{page.physical_page_number}:c{index}"
                f":{config.version}:{config_hash}:{digest}"
            )
            chunks.append(
                ChunkRecord(
                    chunk_id=identity,
                    document_id=page.document_id,
                    source_filename=page.source_filename,
                    source_sha256=page.source_sha256,
                    physical_page_number=page.physical_page_number,
                    chunk_index_on_page=index,
                    text=text,
                    chunking_config=config,
                    content_sha256=digest,
                )
            )
            if end == len(page.text):
                break
            position += step
            index += 1
    return chunks
