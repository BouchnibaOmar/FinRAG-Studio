import os
import urllib.request
from pathlib import Path

from finrag.config import FinragError
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, read_json, write_atomic

MODEL_NAME = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "sentence_bert_config.json",
)


def download_model(output: Path, official_metadata: Path) -> dict:
    metadata = read_json(official_metadata)
    if metadata["sha"] != REVISION or metadata["cardData"]["license"] != "mit":
        raise FinragError("Official model revision or license differs from the approved model.")
    files = {row["rfilename"]: row for row in metadata["siblings"]}
    output.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        path = output / name
        row = files[name]
        expected = row.get("lfs", {}).get("sha256")
        if path.exists():
            if path.stat().st_size != row["size"] or (expected and hash_file(path) != expected):
                raise FinragError(f"{name}: existing model file is inconsistent.")
            continue
        temporary = output / (name + ".tmp")
        url = f"https://huggingface.co/{MODEL_NAME}/resolve/{REVISION}/{name}"
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        if temporary.stat().st_size != row["size"] or (
            expected and hash_file(temporary) != expected
        ):
            raise FinragError(f"{name}: downloaded model size/hash mismatch.")
        os.replace(temporary, path)
    manifest = {
        "model": MODEL_NAME,
        "revision": REVISION,
        "license": "MIT",
        "official_source": f"https://huggingface.co/{MODEL_NAME}/tree/{REVISION}",
        "files": {name: hash_file(output / name) for name in MODEL_FILES},
        "dimensions": 384,
        "maximum_length": 512,
    }
    path = output / "model_manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise FinragError("Local model manifest drifted.")
    if not path.exists():
        write_atomic(path, json_bytes(manifest))
    return manifest


def validate_model(output: Path) -> dict:
    manifest = read_json(output / "model_manifest.json")
    if (
        manifest["model"] != MODEL_NAME
        or manifest["revision"] != REVISION
        or set(manifest["files"]) != set(MODEL_FILES)
    ):
        raise FinragError("Local model manifest does not identify the approved complete snapshot.")
    if any(hash_file(output / name) != digest for name, digest in manifest["files"].items()):
        raise FinragError("Local model file hash changed.")
    return manifest
