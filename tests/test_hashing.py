from pathlib import Path

import pytest

from finrag.ingestion.hashing import document_id, hash_bytes, hash_file


def test_sha256_and_document_identity_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert hash_file(path) == hash_bytes(b"abc") == expected
    assert document_id(expected) == f"sha256:{expected}"


@pytest.mark.parametrize("digest", ["", "A" * 64, "g" * 64, "a" * 63])
def test_document_id_rejects_invalid_hash(digest: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        document_id(digest)
