import numpy as np

from finrag.config import FinragError
from finrag.retrieval.e5 import E5Encoder


def run_local_model_smoke(encoder: E5Encoder) -> dict:
    vectors = encoder.encode(["Bénéfice annuel fictif", "Fictional annual revenue"], "query")
    if vectors.shape != (2, 384) or not np.isfinite(vectors).all():
        raise FinragError("Real local E5 multilingual smoke test failed.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise FinragError("Real local E5 smoke vectors are not normalised.")
    return {
        "status": "passed",
        "model_revision": encoder.manifest["revision"],
        "device": encoder.device,
        "shape": [2, 384],
        "finite": True,
        "normalised": True,
        "inputs": "two fictional French/English queries",
        "evaluation_questions_ranked": 0,
    }
