import hashlib
from pathlib import Path

from finrag.config import filesystem_operation


def hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with filesystem_operation("read file for SHA-256", path), path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_id(source_sha256: str) -> str:
    if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        raise ValueError("Document identity requires a lowercase SHA-256 digest.")
    return f"sha256:{source_sha256}"
