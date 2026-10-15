import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np

from finrag.config import FinragError
from finrag.retrieval.model_assets import validate_model

PREFIXES = {"query": "query: ", "passage": "passage: "}


def prefixed(text: str, kind: Literal["query", "passage"]) -> str:
    if kind not in PREFIXES:
        raise FinragError("E5 encoding requires query or passage input kind.")
    return PREFIXES[kind] + text


def token_count(tokenizer: Any, text: str, kind: Literal["query", "passage"]) -> int:
    return len(
        tokenizer(prefixed(text, kind), add_special_tokens=True, truncation=False)["input_ids"]
    )


def require_token_limit(
    tokenizer: Any, texts: list[str], kind: Literal["query", "passage"], limit: int
) -> list[int]:
    lengths = [token_count(tokenizer, text, kind) for text in texts]
    if any(length > limit for length in lengths):
        raise FinragError(
            f"E5 {kind} input exceeds the effective token limit; truncation is forbidden."
        )
    return lengths


class E5Encoder:
    def __init__(
        self, model_directory: Path, device: Literal["cpu", "cuda"] = "cpu", threads: int = 4
    ):
        if "faiss" in sys.modules:
            raise FinragError("E5 encoding must run separately from an imported FAISS runtime.")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.manifest = validate_model(model_directory)
        torch.set_num_threads(threads)
        torch.manual_seed(20261013)
        torch.use_deterministic_algorithms(True)
        if device == "cuda":
            if not torch.cuda.is_available():
                raise FinragError("CUDA is not available in the installed PyTorch runtime.")
            try:
                torch.ones((16, 16), device="cuda").matmul(
                    torch.ones((16, 16), device="cuda")
                ).sum().item()
            except RuntimeError as error:
                raise FinragError(
                    "Actual PyTorch GPU kernel failed; CUDA device is incompatible."
                ) from error
            torch.backends.cuda.matmul.allow_tf32 = False
        self.device = device
        self.threads = threads
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_directory),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        self.model = (
            AutoModel.from_pretrained(
                str(model_directory),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                attn_implementation="eager",
            )
            .to(device=device, dtype=torch.float32)
            .eval()
        )
        self.limit = min(
            512, self.model.config.max_position_embeddings, self.tokenizer.model_max_length
        )
        if self.model.config.hidden_size != 384:
            raise FinragError("Pinned E5 model has an unexpected embedding dimension.")

    def encode(self, texts: list[str], kind: Literal["query", "passage"]) -> "np.ndarray":
        import numpy as np
        import torch

        from finrag.retrieval.vector_index import normalize_vectors

        require_token_limit(self.tokenizer, texts, kind, self.limit)
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        inputs = self.tokenizer(
            [prefixed(text, kind) for text in texts],
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            weights = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1)
        return normalize_vectors(pooled.cpu().numpy().astype(np.float32), 384)
