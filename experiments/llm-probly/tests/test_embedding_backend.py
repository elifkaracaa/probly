"""Tests for the embedding backend of the semantic methods (B3).

The backend is part of ``NliSettings``: it changes the tag (cache file name) and the matrix
fingerprints, so embedding matrices can never be read as NLI matrices from the A5 cache.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from common.nli_matrix import (
    NliSettings,
    count_cached_questions,
    matrix_fingerprint,
    nli_cache_path,
    nli_settings,
    read_cached_matrices,
    set_nli_cache_file,
    set_nli_settings,
)
from common.semantic_scores import semantic_scores
from methods import _graph_common, _se_common, eig_val_laplacian
import pytest
from scripts.run_benchmark import parse_args
import torch

from probly.representer.embedder.huggingface import HFTextEmbedder
from probly.representer.semantic_clustering.huggingface import HFEmbeddingSemanticClusterer

ROOT = Path(__file__).resolve().parents[1]
VECTORS = {"Paris": [1.0, 0.0], "It is Paris.": [0.95, 0.312], "Lyon": [0.0, 1.0]}
ROW = {
    "id": 7,
    "question": "Capital of France?",
    "greedy_answer": "Paris",
    "samples": ["Paris", "It is Paris.", "Lyon"],
}


class FakeEmbeddingModel:
    """Sentence-transformer stand-in: looks the vectors up; unknown texts get their own direction."""

    def encode(self, sentences, **kwargs):
        return torch.tensor([VECTORS.get(s, [0.6, 0.8]) for s in sentences], dtype=torch.float32)


def fake_clusterer(threshold: float = 0.9) -> HFEmbeddingSemanticClusterer:
    return HFEmbeddingSemanticClusterer(HFTextEmbedder(FakeEmbeddingModel()), threshold=threshold)


@pytest.fixture
def fresh_state(monkeypatch):
    """Empty in-memory matrices and clusterers, a fake embedding model; settings restored afterwards."""
    monkeypatch.setattr(
        HFEmbeddingSemanticClusterer,
        "from_model_name",
        classmethod(lambda _cls, _name=None, *, threshold=0.9: fake_clusterer(threshold)),
    )
    monkeypatch.setattr(_se_common, "_matrices", {})
    monkeypatch.setattr(_se_common, "_disk", {})
    monkeypatch.setattr(_se_common, "_clusterers", {})
    monkeypatch.setattr(_graph_common, "_scores", {})
    old = nli_settings()
    yield
    set_nli_settings(old.model, use_question=old.use_question, backend=old.backend, threshold=old.threshold)
    set_nli_cache_file(None)


def test_semantic_scores_switch_backend_with_one_argument():
    scores = semantic_scores(["Paris", "It is Paris.", "Lyon", "Paris"], clusterer=fake_clusterer())

    assert scores["clusters"] == [0, 0, 1, 0]
    assert scores["num_semantic_sets"] == 2.0


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="semantic backend must be one of"):
        NliSettings(backend="bert")


def test_backend_and_threshold_change_the_tag_and_the_cache_file():
    nli = NliSettings()
    embedding = NliSettings(backend="embedding")
    data = Path("data/qwen0.5b/triviaqa_200.jsonl")

    assert nli.tag == "deberta-large"
    assert embedding.tag == "embedding-all-mpnet-base-v2@0.9"
    assert replace(embedding, threshold=0.8).tag == "embedding-all-mpnet-base-v2@0.8"
    assert nli_cache_path(data, embedding).name == "triviaqa_200.nli.embedding-all-mpnet-base-v2@0.9.npz"
    assert nli_cache_path(data, embedding) != nli_cache_path(data, nli)


def test_backend_and_threshold_change_the_fingerprint_but_nli_fingerprints_stay_as_before():
    nli = NliSettings()
    embedding = NliSettings(backend="embedding")

    assert matrix_fingerprint(ROW, embedding) != matrix_fingerprint(ROW, nli)
    assert matrix_fingerprint(ROW, replace(embedding, threshold=0.8)) != matrix_fingerprint(ROW, embedding)
    # the threshold is not used by NLI, so existing NLI caches stay valid whatever its value
    assert matrix_fingerprint(ROW, replace(nli, threshold=0.5)) == matrix_fingerprint(ROW, nli)


def test_an_embedding_run_never_writes_into_or_reads_from_the_nli_cache(fresh_state, tmp_path):  # noqa: ARG001
    """Elif's scenario: embedding run first, then an NLI run on the same data."""
    data = tmp_path / "triviaqa_3.jsonl"
    rows = [{**ROW, "id": i} for i in range(3)]

    embedding = set_nli_settings(backend="embedding")
    embedding_file = nli_cache_path(data, embedding)
    set_nli_cache_file(embedding_file)
    record = eig_val_laplacian.score(rows)
    set_nli_cache_file(None)

    assert isinstance(_se_common.get_clusterer(), HFEmbeddingSemanticClusterer)
    assert record["nli"] == embedding.tag
    assert not any(record["failed"])
    assert embedding_file.exists()
    nli = set_nli_settings(backend="nli")
    assert not nli_cache_path(data, nli).exists()  # the NLI cache was not touched
    assert count_cached_questions(embedding_file, nli, rows) == 0  # and the embedding file is useless for NLI
    assert all(_se_common.matrix_key(row) not in _se_common._matrices for row in rows)  # noqa: SLF001
    assert len(read_cached_matrices(embedding_file, embedding)) == 3  # but a second embedding run reuses it


def test_plugins_compute_embedding_matrices_with_the_settings_threshold(fresh_state):  # noqa: ARG001
    rows = [json.loads(line) for line in (ROOT / "data/qwen0.5b/triviaqa_200.jsonl").open()][:3]
    for row in rows:
        row["greedy_answer"], row["samples"] = "Paris", ["Paris", "It is Paris.", "Lyon"]

    set_nli_settings(backend="embedding", threshold=0.95)
    record = eig_val_laplacian.score(rows)
    clusterer = _se_common.get_clusterer()

    assert isinstance(clusterer, HFEmbeddingSemanticClusterer)
    assert clusterer.threshold == 0.95
    assert record["uncertainty"][0] == pytest.approx(record["uncertainty"][1])


def test_runner_has_the_backend_option():
    assert parse_args(["--config", "c.yaml"]).semantic_backend == "nli"
    assert parse_args(["--config", "c.yaml", "--semantic-backend", "embedding"]).semantic_backend == "embedding"
    with pytest.raises(SystemExit):
        parse_args(["--config", "c.yaml", "--semantic-backend", "bert"])
