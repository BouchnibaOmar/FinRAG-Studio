import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_bm25 import synthetic_chunks
from test_retrieval_metrics import evaluation_question, prediction

from finrag.chunking.model_chunker import split_for_model, validate_model_chunks
from finrag.cli.main import parser
from finrag.config import FinragError
from finrag.domain.dense import DenseHit
from finrag.evaluation.run_dense_comparison import choose_method, public_comparison
from finrag.retrieval.e5 import prefixed, require_token_limit
from finrag.retrieval.embedding_cache import EmbeddingCache
from finrag.retrieval.hybrid import reciprocal_rank_fusion
from finrag.retrieval.vector_index import DenseIndex, normalize_vectors


class CharacterTokenizer:
    def __call__(self, text, *, add_special_tokens, truncation, return_offsets_mapping=False):
        assert add_special_tokens is True and truncation is False
        result = {"input_ids": list(range(len(text) + 2))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(0, 0)] + [(i, i + 1) for i in range(len(text))] + [(0, 0)]
        return result


def model_chunks(texts):
    return split_for_model(synthetic_chunks(texts), CharacterTokenizer(), "synthetic-revision", 512)


def hit(identifier, rank=1, page=1):
    chunk = model_chunks(["fictional revenue"])[0]
    return DenseHit(
        rank=rank,
        score=0.7,
        chunk_id=identifier,
        physical_page_number=page,
        configuration_id="synthetic",
        text=chunk.text,
        document_id=chunk.document_id,
        source_filename=chunk.source_filename,
        source_sha256=chunk.source_sha256,
    )


def test_prefixes_and_limit_include_special_tokens():
    assert prefixed("bénéfice", "query") == "query: bénéfice"
    assert prefixed("revenue", "passage") == "passage: revenue"
    assert require_token_limit(CharacterTokenizer(), ["abc"], "query", 12) == [12]
    with pytest.raises(FinragError, match="limit"):
        require_token_limit(CharacterTokenizer(), ["abcd"], "query", 12)


@pytest.mark.parametrize("normalised", [True, False])
def test_separate_real_smoke_contract_with_synthetic_encoder(normalised):
    from finrag.evaluation.local_model_smoke import run_local_model_smoke

    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[:, 0] = 1 if normalised else 2
    encoder = SimpleNamespace(
        encode=lambda texts, kind: vectors, manifest={"revision": "synthetic"}, device="cpu"
    )
    if normalised:
        assert run_local_model_smoke(encoder)["evaluation_questions_ranked"] == 0
    else:
        with pytest.raises(FinragError, match="normalised"):
            run_local_model_smoke(encoder)


def test_child_offsets_cover_parent_and_keep_physical_page():
    parents = synthetic_chunks(["é" * 100])
    children = split_for_model(parents, CharacterTokenizer(), "revision", 40, 4)
    assert len(children) > 1
    assert all(child.token_count <= 40 and child.physical_page_number == 1 for child in children)
    assert children[0].source_character_start == 0
    assert children[-1].source_character_end == 100
    for left, right in zip(children, children[1:], strict=False):
        assert right.source_character_start <= left.source_character_end
    corrupted = [children[0].model_copy(update={"source_character_start": 1}), *children[1:]]
    with pytest.raises(FinragError, match="offsets"):
        validate_model_chunks(parents, corrupted, CharacterTokenizer(), "revision", 40, 4)
    with pytest.raises(FinragError, match="lose"):
        validate_model_chunks(parents, children[:-1], CharacterTokenizer(), "revision", 40, 4)


def test_fitting_chunks_keep_exact_ids_and_text():
    parents = synthetic_chunks(["fictional operating income"])
    children = split_for_model(parents, CharacterTokenizer(), "revision")
    assert children[0].chunk_id == parents[0].chunk_id
    assert children[0].text == parents[0].text


@pytest.mark.parametrize(
    "vectors", [[[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], [[1, 2, 3]]]
)
def test_invalid_vectors_rejected(vectors):
    with pytest.raises(FinragError):
        normalize_vectors(np.array(vectors), 2)


def test_cosine_mapping_ties_persistence_and_integrity(tmp_path):
    chunks = model_chunks(["first", "second", "third"])
    vectors = normalize_vectors(np.array([[0, 1], [1, 0], [1, 0]]), 2)
    index = DenseIndex(chunks, vectors, 2)
    query = normalize_vectors(np.array([[1, 0]]), 2)
    expected = sorted(chunk.chunk_id for chunk in chunks[1:])
    assert [row.chunk_id for row in index.search(query, 2)] == expected
    index.save(tmp_path / "index", {"revision": "synthetic"})
    restored = DenseIndex.load(tmp_path / "index", {"revision": "synthetic"})
    assert restored.search(query) == index.search(query)
    with pytest.raises(FinragError, match="hashes"):
        DenseIndex.load(tmp_path / "index", {"revision": "changed"})
    with (tmp_path / "index/rows.jsonl").open("ab") as output:
        output.write(b"tamper")
    with pytest.raises(FinragError, match="hashes"):
        DenseIndex.load(tmp_path / "index", {"revision": "synthetic"})


def test_checkpoint_resume_does_not_encode_completed_rows(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache", {"revision": "one"}, 3, 2)
    cache.store([1], normalize_vectors(np.array([[1, 0]]), 2))
    calls = []

    def encode(texts):
        calls.extend(texts)
        return normalize_vectors(np.array([[0, 1]] * len(texts)), 2)

    resumed = EmbeddingCache(tmp_path / "cache", {"revision": "one"}, 3, 2)
    vectors = resumed.resume(["a", "b", "c"], encode, 2, tmp_path / "progress.log")
    assert calls == ["a", "c"] and vectors[1].tolist() == [1, 0]
    resumed.resume(["a", "b", "c"], encode, 2, tmp_path / "progress.log")
    assert calls == ["a", "c"]
    with pytest.raises(FinragError, match="fingerprint"):
        EmbeddingCache(tmp_path / "cache", {"revision": "two"}, 3, 2)
    (tmp_path / "cache/batch-000000.npy").write_bytes(b"corrupt")
    with pytest.raises(FinragError, match="hash"):
        EmbeddingCache(tmp_path / "cache", {"revision": "one"}, 3, 2)


def test_rrf_fuses_ranks_deduplicates_and_breaks_ties():
    result = reciprocal_rank_fusion([hit("a"), hit("a", 2), hit("b", 3)], [hit("b"), hit("a", 3)])
    assert result[0].chunk_id == "a" and len(result) == 2
    assert result[0].score == pytest.approx(1 / 61 + 1 / 63)
    assert result[0].component_ranks == {"bm25": 1, "dense": 3}
    assert [row.chunk_id for row in reciprocal_rank_fusion([hit("b")], [hit("a")])] == ["a", "b"]
    with pytest.raises(FinragError, match="provenance"):
        reciprocal_rank_fusion([hit("a")], [hit("a", page=2)])


def test_ranking_modules_do_not_import_evaluation_or_labels():
    for filename in ("bm25.py", "e5.py", "hybrid.py", "vector_index.py"):
        tree = ast.parse((Path("src/finrag/retrieval") / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "evaluation" not in (node.module or "")
        assert "reference_answer" not in ast.unparse(tree)
        assert "relevant_pages" not in ast.unparse(tree)


def test_dense_cli_never_accepts_test_split():
    with pytest.raises(SystemExit):
        parser().parse_args(["run-dense-comparison", "--split", "test"])


def test_selection_and_public_report_exclude_private_text():
    groups = {
        name: {"overall": {"metrics": {"hit@5": 0.5, "mrr@10": 0.2, "hit@1": 0.1}}}
        for name in ("bm25", "dense", "hybrid")
    }
    assert choose_method(groups) == "bm25"
    question = SimpleNamespace(
        question_id="synthetic",
        answerable=False,
        evidence=[],
        reference_answer="SECRET",
        question="PRIVATE",
        relevant_pages=[],
    )
    predictions = {
        name: [SimpleNamespace(question_id="synthetic", retrieved=[hit("a")])] for name in groups
    }
    report = public_comparison([question], predictions, groups)
    serialised = json.dumps(report)
    assert (
        "SECRET" not in serialised
        and "PRIVATE" not in serialised
        and "fictional revenue" not in serialised
    )
    assert report["test_metrics_generated"] is False


def test_complete_synthetic_comparison_resumes_and_verifies(monkeypatch, tmp_path):
    from test_bm25 import lexical_config

    from finrag.evaluation import run_dense_comparison as runner
    from finrag.retrieval.bm25 import BM25

    monkeypatch.chdir(tmp_path)
    chunks = model_chunks(["fictional revenue", "fictional profit"])
    question = evaluation_question(chunks).model_copy(update={"question": "fictional revenue"})
    absent = evaluation_question(chunks, False).model_copy(
        update={"question_id": "finrag_v1_bnp_02", "question": "future profit"}
    )
    questions = [question, absent]
    config = lexical_config()
    lexical = BM25(chunks, config)
    reference = [prediction(q, lexical.search(q.question)) for q in questions]
    root = tmp_path / "artifacts/private/synthetic-comparison"
    root.mkdir(parents=True)
    (root / "sample_benchmark.json").write_text(
        json.dumps({"fingerprint": {}, "requires_manual_execution": False})
    )
    calls = []

    class SyntheticEncoder:
        def __init__(self, model, device):
            pass

        def encode(self, texts, kind):
            calls.append((kind, list(texts)))
            vectors = np.zeros((len(texts), 384), dtype=np.float32)
            for row, text in enumerate(texts):
                vectors[row, 0 if "revenue" in text else 1] = 1
            return vectors

    monkeypatch.setattr(runner, "E5Encoder", SyntheticEncoder)
    monkeypatch.setattr(runner, "private_output", lambda path, project: path)
    monkeypatch.setattr(
        runner,
        "prepare_dense_inputs",
        lambda encoder, path: (chunks, {"B_identities_preserved": True}),
    )
    monkeypatch.setattr(runner, "encoding_fingerprint", lambda *args: {})
    monkeypatch.setattr(runner, "load_dev_questions", lambda path: questions)
    monkeypatch.setattr(runner, "load_config", lambda path: config)
    monkeypatch.setattr(
        runner, "read_jsonl", lambda path: [row.model_dump(mode="json") for row in reference]
    )
    monkeypatch.setattr(
        runner,
        "hash_file",
        lambda path: "5a025fdf2336b4d15044a5f0d3d58472ff4381d9d892ecff2289497e70614cc5",
    )
    result = runner.run_comparison(tmp_path, root, "cpu", 2)
    assert result["independently_verified"] is True
    assert result["groups"]["dense"]["overall"]["answerable"] == 1
    assert result["groups"]["dense"]["overall"]["unanswerable"] == 1
    assert len(calls) == 2
    # The native index reader uses its own validated JSON reader.
    runner.run_comparison(tmp_path, root, "cpu", 2)
    assert len(calls) == 2
    report = json.loads((tmp_path / "experiments/phase2b/reports/dev_results.json").read_text())
    assert report["test_metrics_generated"] is False
    assert "fictional revenue" not in json.dumps(report)
    assert (root / "dense/predictions.jsonl").exists()
