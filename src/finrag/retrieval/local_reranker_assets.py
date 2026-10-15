import argparse
from pathlib import Path

from finrag.config import FinragError, private_output
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, read_json, write_atomic

MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "324cc40576b08b305b9c65a867c26c173a477ae2"
WEIGHT_SHA256 = "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286"
MAX_PAIR_TOKENS = 8192
FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "README.md",
)


def seal_model(directory: Path) -> dict:
    for name in FILES:
        metadata = directory / ".cache/huggingface/download" / (name + ".metadata")
        if metadata.read_text(encoding="utf-8").split("\n")[0] != REVISION:
            raise FinragError(f"Downloaded revision differs for {name}.")
    if hash_file(directory / "model.safetensors") != WEIGHT_SHA256:
        raise FinragError("Local BGE weights differ from the pinned official SHA-256.")
    config, tokenizer = (
        read_json(directory / "config.json"),
        read_json(directory / "tokenizer_config.json"),
    )
    if (
        config["architectures"] != ["XLMRobertaForSequenceClassification"]
        or config["max_position_embeddings"] != 8194
        or config["pad_token_id"] != 1
        or len(config["id2label"]) != 1
        or tokenizer["model_max_length"] != MAX_PAIR_TOKENS
    ):
        raise FinragError("Local BGE architecture or pair length differs from the pinned snapshot.")
    manifest = {
        "model": MODEL,
        "revision": REVISION,
        "license": "apache-2.0",
        "max_pair_tokens": MAX_PAIR_TOKENS,
        "files": {name: hash_file(directory / name) for name in FILES},
    }
    path = directory / "finrag_local_model.json"
    if path.exists():
        if read_json(path) != manifest:
            raise FinragError("Existing local BGE manifest differs; overwrite refused.")
    else:
        write_atomic(path, json_bytes(manifest))
    return manifest


def validate_model(directory: Path) -> dict:
    manifest = read_json(directory / "finrag_local_model.json")
    if (
        manifest["model"] != MODEL
        or manifest["revision"] != REVISION
        or manifest["max_pair_tokens"] != MAX_PAIR_TOKENS
        or set(manifest["files"]) != set(FILES)
        or manifest["files"]["model.safetensors"] != WEIGHT_SHA256
        or any(hash_file(directory / name) != digest for name, digest in manifest["files"].items())
    ):
        raise FinragError("Local BGE files or pinned manifest differ.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal an already downloaded pinned BGE snapshot")
    parser.add_argument("--model-directory", type=Path, required=True)
    args = parser.parse_args()
    directory = private_output(args.model_directory, Path.cwd())
    print(seal_model(directory))


if __name__ == "__main__":
    main()
