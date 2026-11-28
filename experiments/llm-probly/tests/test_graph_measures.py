"""Tests for the graph measures (B1) on hand-made matrices and on the week-2 caches."""

from __future__ import annotations

import json
import math
from pathlib import Path

from common.graph_measures import (
    SPECTRAL_DECIMALS,
    deg_mat,
    eccentricity,
    eig_val_laplacian,
    graph_scores,
    normalized_laplacian,
    num_semantic_sets,
    similarity_matrix,
)
from common.nli_matrix import load_nli_cache, nli_cache_path, set_nli_cache_file, set_nli_settings
from methods import _graph_common, _se_common, deg_mat as deg_mat_plugin, eccentricity as eccentricity_plugin
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CONTRADICTION, NEUTRAL, ENTAILMENT = 0, 1, 2  # probly's label order


def blocks(*sizes: int) -> np.ndarray:
    """Similarity matrix of fully agreeing groups that do not agree with each other at all."""
    groups = np.repeat(np.arange(len(sizes)), sizes)
    return (groups[:, None] == groups[None, :]).astype(float)


def nli_from_groups(groups: list[int]) -> torch.Tensor:
    """Question matrix: same group entails, different groups contradict; index 0 = greedy answer."""
    n = len(groups)
    matrix = torch.zeros((n, n, 3))
    for a in range(n):
        for b in range(n):
            matrix[a, b, ENTAILMENT if groups[a] == groups[b] else CONTRADICTION] = 1.0
    matrix[torch.arange(n), torch.arange(n)] = float("nan")
    return matrix


def test_similarity_matrix_is_symmetric_entailment_mean_without_greedy():
    matrix = torch.full((3, 3, 3), float("nan"))
    matrix[1, 2] = torch.tensor([0.1, 0.1, 0.8])  # sample 0 -> sample 1
    matrix[2, 1] = torch.tensor([0.2, 0.4, 0.4])  # sample 1 -> sample 0
    matrix[0, 1] = matrix[1, 0] = matrix[0, 2] = matrix[2, 0] = torch.tensor([1.0, 0.0, 0.0])  # greedy

    w = similarity_matrix(matrix)

    np.testing.assert_allclose(w, [[1.0, 0.6], [0.6, 1.0]], atol=1e-6)
    np.testing.assert_allclose(similarity_matrix(matrix, self_similarity=0.5).diagonal(), [0.5, 0.5])


def test_laplacian_has_one_zero_eigenvalue_per_disconnected_group():
    eigenvalues = np.linalg.eigvalsh(normalized_laplacian(blocks(3, 2, 1)))

    assert np.sum(np.isclose(eigenvalues, 0.0, atol=1e-9)) == 3


@pytest.mark.parametrize("sizes", [(5,), (3, 2), (2, 2, 1), (1, 1, 1, 1)])
def test_eig_val_laplacian_counts_fully_separated_groups(sizes):
    assert eig_val_laplacian(blocks(*sizes)) == pytest.approx(len(sizes))


@pytest.mark.parametrize("cross", [0.0, 0.5, 0.9])
def test_eig_val_laplacian_falls_as_two_groups_agree_more(cross):
    w = blocks(3, 3)
    w[w == 0] = cross  # every answer of one group agrees `cross` with every answer of the other

    # Closed form for two equal groups: 1 + (1 - c) / (1 + c); 2 when separated, 1 when identical.
    assert eig_val_laplacian(w) == pytest.approx(1 + (1 - cross) / (1 + cross))


@pytest.mark.parametrize(
    ("sizes", "expected_eig_val", "expected_eccentricity"),
    [
        ((10,), 1.0, 0.0),
        ((5, 5), 2.0, 1.0),
        ((4, 3, 3), 3.0, round(math.sqrt(2), SPECTRAL_DECIMALS)),  # c separated groups: sqrt(c - 1)
        ((1,) * 10, 10.0, 3.0),
    ],
)
def test_spectral_measures_ignore_eigensolver_noise(sizes, expected_eig_val, expected_eccentricity):
    """Noise of 1e-14 (more than different machines give) must not change the score, so ties stay ties."""
    exact = blocks(*sizes)
    rng = np.random.default_rng(0)
    for _ in range(20):
        noise = rng.normal(0.0, 1e-14, exact.shape)
        noisy = np.clip(exact + (noise + noise.T) / 2, 0.0, 1.0)
        np.fill_diagonal(noisy, 1.0)
        assert eig_val_laplacian(noisy) == expected_eig_val
        assert eccentricity(noisy) == expected_eccentricity


def test_deg_mat_known_values():
    assert deg_mat(blocks(4)) == pytest.approx(0.0)  # everyone agrees
    assert deg_mat(blocks(1, 1, 1, 1)) == pytest.approx(0.75)  # nobody agrees: 1 - 1/K
    assert deg_mat(blocks(2, 2)) == pytest.approx(0.5)


def test_eccentricity_zero_when_all_agree_and_grows_with_groups():
    assert eccentricity(blocks(5)) == pytest.approx(0.0, abs=1e-9)
    assert eccentricity(blocks(3, 2)) > 0.0
    assert eccentricity(blocks(2, 2, 1)) > eccentricity(blocks(3, 2))


def test_num_semantic_sets_counts_distinct_ids():
    assert num_semantic_sets(torch.tensor([0, 0, 1, 2, 1])) == 3.0
    assert num_semantic_sets(np.array([0, 0, 0])) == 1.0


def test_graph_scores_from_an_nli_matrix():
    scores = graph_scores(nli_from_groups([0, 0, 0, 1, 1]))  # greedy + samples in groups (0, 0, 1, 1)

    assert scores["num_semantic_sets"] == 2.0
    assert scores["eig_val_laplacian"] == pytest.approx(2.0)
    assert scores["deg_mat"] == pytest.approx(0.5)
    assert scores["eccentricity"] > 0.0


def reference_lm_polygraph(w: np.ndarray) -> tuple[float, float, float]:
    """LM-Polygraph's formulas for DegMat, EigValLaplacian and Eccentricity, written out with matrices."""
    k = len(w)
    d = np.diag(w.sum(axis=1))
    d_inv_sqrt = np.linalg.inv(np.sqrt(d))
    laplacian = np.eye(k) - d_inv_sqrt @ w @ d_inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    degmat = np.trace(k - d) / k**2
    eig = sum(max(0, 1 - lam) for lam in eigenvalues)
    kept = eigenvectors[:, eigenvalues < 0.9].T
    ecc = np.linalg.norm([np.linalg.norm(x - x.mean(0), 2) for x in kept], 2)
    return degmat, eig, ecc


def test_matches_lm_polygraph_formulas_on_random_matrices():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.uniform(0, 1, size=(10, 10))
        w = (a + a.T) / 2
        np.fill_diagonal(w, 1.0)

        expected = reference_lm_polygraph(w)

        assert (deg_mat(w), eig_val_laplacian(w), eccentricity(w)) == pytest.approx(expected)


def test_plugins_read_the_persistent_nli_cache_without_nli_calls(monkeypatch):
    def no_nli(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the NLI model must not run when the cache file matches")

    monkeypatch.setattr(_se_common, "compute_matrices", no_nli)
    monkeypatch.setattr(_se_common, "_matrices", {})
    monkeypatch.setattr(_se_common, "_disk", {})
    monkeypatch.setattr(_graph_common, "_scores", {})
    data = ROOT / "data/qwen0.5b/triviaqa_200.jsonl"
    rows = [json.loads(line) for line in data.open()]
    settings = set_nli_settings("deberta-large", use_question=False)
    path = nli_cache_path(data, settings)
    set_nli_cache_file(path)
    try:
        record = deg_mat_plugin.score(rows)
    finally:
        set_nli_cache_file(None)

    assert record["ids"] == [r["id"] for r in rows]
    assert not any(record["failed"])
    assert record["nli_pairs_per_question"] == 0.0
    assert record["nli"] == "deberta-large"
    cache = load_nli_cache(path)
    assert record["uncertainty"][0] == pytest.approx(deg_mat(similarity_matrix(cache["matrices"][0])))


def test_plugins_fall_back_to_computed_matrices_for_rows_without_a_cache(monkeypatch):
    computed = []

    def fake_matrix(row):
        computed.append(row["id"])
        return nli_from_groups([0, 0, 0, 1])

    monkeypatch.setattr("methods._se_common.prepare_matrices", lambda _rows: None)
    monkeypatch.setattr("methods._se_common.get_matrix", fake_matrix)
    monkeypatch.setattr(_graph_common, "_scores", {})
    rows = [
        {"id": i, "model": "m", "dataset": "nowhere", "greedy_answer": "a", "samples": ["a", "a", f"b{i}"]}
        for i in range(2)
    ]

    record = eccentricity_plugin.score(rows)

    assert computed == [0, 1]
    assert record["uncertainty"][0] > 0.0
    assert record["failed"] == [False, False]
    assert record["nli_pairs_per_question"] == 0.0  # the fake did not score pairs
