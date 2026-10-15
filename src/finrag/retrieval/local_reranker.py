import math
from pathlib import Path
from typing import Any

from finrag.config import FinragError
from finrag.retrieval.candidate_reranking import RerankInput, input_identity
from finrag.retrieval.local_reranker_assets import MAX_PAIR_TOKENS


def count_pairs(inputs: list[RerankInput], tokenizer: Any) -> dict:
    pairs = []
    for source in inputs:
        for candidate in source.candidates:
            encoded = tokenizer(
                source.question,
                candidate.text,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            pairs.append(
                {
                    "input_id": input_identity(source),
                    "candidate_id": candidate.candidate_id,
                    "tokens": len(encoded["input_ids"]),
                }
            )
    return {
        "pair_limit": MAX_PAIR_TOKENS,
        "pairs": pairs,
        "maximum_tokens": max((p["tokens"] for p in pairs), default=0),
        "overflows": [p for p in pairs if p["tokens"] > MAX_PAIR_TOKENS],
    }


def hardware_sample(pairs: list[dict]) -> list[dict]:
    ordered = sorted(pairs, key=lambda p: (p["tokens"], p["input_id"], p["candidate_id"]))
    if len(ordered) < 9:
        raise FinragError("Hardware sample requires at least nine pairs.")
    middle = len(ordered) // 2
    positions = sorted(
        {
            0,
            1,
            2,
            middle - 1,
            middle,
            middle + 1,
            len(ordered) - 3,
            len(ordered) - 2,
            len(ordered) - 1,
        }
    )
    return [ordered[index] for index in positions]


def rank_scores(source: RerankInput, scores: list[float]) -> list[dict]:
    if len(scores) != len(source.candidates) or any(not math.isfinite(s) for s in scores):
        raise FinragError("Local reranker must return one finite score per candidate.")
    pairs = [
        (candidate.candidate_id, float(score))
        for candidate, score in zip(source.candidates, scores, strict=True)
    ]
    return [
        {"candidate_id": identifier, "score": score, "rank": rank}
        for rank, (identifier, score) in enumerate(sorted(pairs, key=lambda p: (-p[1], p[0])), 1)
    ]


def validate_local_record(source: RerankInput, record: dict, config: dict) -> list[dict]:
    if record["input_id"] != input_identity(source) or record["configuration"] != config:
        raise FinragError("Local cache input or configuration differs.")
    rows = record["ranking"]
    scores = {row["candidate_id"]: row["score"] for row in rows}
    expected = {candidate.candidate_id for candidate in source.candidates}
    if len(rows) != len(scores) or set(scores) != expected:
        raise FinragError("Local cache has duplicate, missing or unknown IDs.")
    ordered = rank_scores(source, [scores[c.candidate_id] for c in source.candidates])
    if rows != ordered:
        raise FinragError("Local cache score order differs.")
    return ordered


class LocalReranker:
    def __init__(self, directory: Path, device: str, dtype: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device not in ("cpu", "cuda") or dtype not in ("float32", "float16"):
            raise FinragError("Local reranker requires explicit CPU/CUDA and float32/float16.")
        if device == "cpu" and dtype != "float32":
            raise FinragError("CPU local reranking requires float32.")
        if device == "cuda" and not torch.cuda.is_available():
            raise FinragError("Requested CUDA is unavailable; no CPU fallback.")
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=getattr(torch, dtype),
            attn_implementation="eager",
        ).to(device)
        self.model.eval()
        if self.model.config.num_labels != 1:
            raise FinragError("BGE reranker must have one classification logit.")

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        torch = self.torch
        encoded = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            add_special_tokens=True,
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] > MAX_PAIR_TOKENS:
            raise FinragError("Pair length exceeds BGE limit; inference not launched.")
        with torch.inference_mode():
            logits = self.model(
                **{key: value.to(self.device) for key, value in encoded.items()}
            ).logits
        return logits.float().view(-1).cpu().tolist()

    def synchronize(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def gpu_memory(self, reset: bool = False) -> dict | None:
        if self.device == "cpu":
            return None
        if reset:
            self.torch.cuda.reset_peak_memory_stats()
        return {
            "allocated_bytes": self.torch.cuda.memory_allocated(),
            "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(),
        }
