"""Tests for the sample-count curve and the early-stop rule (B2)."""

from __future__ import annotations

import json
from pathlib import Path

from common.nli_matrix import load_nli_cache, se_and_dse
from common.sample_count import (
    METHODS,
    draw_subsets,
    early_stop_count,
    early_stop_scores,
    k_curve,
    lexical_uncertainty,
    subsample_matrix,
    subset_scores,
)
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CONTRADICTION, NEUTRAL, ENTAILMENT = 0, 1, 2  # probly's label order


def matrix_from_groups(groups: list[int]) -> torch.Tensor:
    """Hand-made question matrix: texts with the same group entail each other, others contradict.

    ``groups[0]`` is the greedy answer, the rest are the samples.
    """
    n = len(groups)
    matrix = torch.zeros((n, n, 3))
    for a in range(n):
        for b in range(n):
            matrix[a, b, ENTAILMENT if groups[a] == groups[b] else CONTRADICTION] = 1.0
    matrix[torch.arange(n), torch.arange(n)] = float("nan")
    return matrix


def make_row(samples: list[str]) -> dict:
    """Data-cache row with equal sample likelihoods."""
    return {"samples": samples, "samples_ll_mean": [-0.5] * len(samples)}


def test_subsample_matrix_keeps_greedy_and_chosen_samples():
    matrix = torch.arange(5 * 5 * 3, dtype=torch.float32).reshape(5, 5, 3)

    sub = subsample_matrix(matrix, [1, 3])  # samples 1 and 3 are rows 2 and 4

    assert sub.shape == (3, 3, 3)
    torch.testing.assert_close(sub[0, 1], matrix[0, 2])
    torch.testing.assert_close(sub[1, 2], matrix[2, 4])
    torch.testing.assert_close(sub[2, 0], matrix[4, 0])


def test_subset_with_all_samples_equals_full_matrix():
    matrix = matrix_from_groups([0, 0, 1, 0, 2])
    row = make_row(["a", "b", "a", "c"])

    se, dse, _ = subset_scores(matrix, row, range(4))

    assert (se, dse) == pytest.approx(se_and_dse(matrix, row["samples_ll_mean"]))


def test_subset_sees_only_its_own_clusters():
    matrix = matrix_from_groups([0, 0, 0, 1, 1])  # samples: A, A, B, B
    row = make_row(["Paris", "Paris", "Lyon", "Lyon"])

    same = subset_scores(matrix, row, [0, 1])
    mixed = subset_scores(matrix, row, [1, 2])

    assert same[0] == pytest.approx(0.0)
    assert same[1] == pytest.approx(0.0)
    assert mixed[1] == pytest.approx(np.log(2))  # two clusters of one sample each
    assert same[2] == pytest.approx(-1.0)  # identical samples: ROUGE-L = 1
    assert mixed[2] == pytest.approx(0.0)  # no shared words


def test_lexical_uncertainty_needs_two_samples():
    with pytest.raises(ValueError, match="two samples"):
        lexical_uncertainty(["only one"])


def test_draw_subsets_are_sorted_unique_and_full_k_has_one_draw():
    rng = np.random.default_rng(0)

    subsets = draw_subsets(10, 4, 5, rng)

    assert len(subsets) == 5
    for keep in subsets:
        assert keep == sorted(keep)
        assert len(set(keep)) == 4
    assert draw_subsets(10, 10, 5, rng) == [list(range(10))]
    with pytest.raises(ValueError, match="between 2 and 10"):
        draw_subsets(10, 1, 5, rng)


def test_k_curve_shape():
    matrices = [matrix_from_groups([0, 0, 1, 0]), matrix_from_groups([0, 1, 2, 3])]
    rows = [make_row(["a", "b", "a"]), make_row(["a", "b", "c"])]

    curve = k_curve(matrices, rows, ks=[2, 3], n_draws=4, seed=0)

    assert curve[2].shape == (4, len(METHODS), 2)
    assert curve[3].shape == (1, len(METHODS), 2)


def test_early_stop_stops_when_first_samples_agree():
    agree_first = matrix_from_groups([0, 0, 0, 0, 1, 2])  # first 3 samples in one cluster
    disagree_first = matrix_from_groups([0, 0, 1, 0, 0, 0])  # sample 2 differs

    assert early_stop_count(agree_first, check_at=3) == 3
    assert early_stop_count(disagree_first, check_at=3) == 5
    with pytest.raises(ValueError, match="between 2 and 5"):
        early_stop_count(agree_first, check_at=6)


def test_early_stop_scores_use_only_the_generated_samples():
    matrices = [matrix_from_groups([0, 0, 0, 1, 2]), matrix_from_groups([0, 1, 2, 1, 1])]
    rows = [make_row(["a", "a", "b", "c"]), make_row(["a", "b", "a", "a"])]

    values, used = early_stop_scores(matrices, rows, check_at=2)

    assert used.tolist() == [2, 4]
    assert values[1, 0] == pytest.approx(0.0)  # stopped after two agreeing samples
    assert values[1, 1] > 0.0  # all 4 samples, two clusters


@pytest.mark.parametrize("model", ["qwen0.5b", "qwen1.5b"])
def test_all_ten_samples_reproduce_the_stored_results(model):
    """Equivalence: K = 10 from the cache gives the week-2 SE, discrete SE and lexical scores."""
    cache = load_nli_cache(ROOT / f"data/{model}/triviaqa_200.nli.npz")
    rows = [json.loads(line) for line in (ROOT / f"data/{model}/triviaqa_200.jsonl").open()]
    results = ROOT / f"results/{model}/triviaqa_200"

    values, used = early_stop_scores(cache["matrices"], rows, check_at=10)

    assert (used == 10).all()
    for m, method in enumerate(METHODS):
        stored = json.loads((results / f"{method}.json").read_text())
        assert stored["ids"] == [r["id"] for r in rows]
        np.testing.assert_allclose(values[m], stored["uncertainty"], atol=1e-5, err_msg=method)


def test_tied_makes_float_noise_equal_but_keeps_real_differences():
    from scripts.sample_count_curve import tied  # noqa: PLC0415

    log2 = float(np.log(2))
    values = np.array([log2, log2 + 4e-7, log2 - 3e-7, log2 + 2e-4, 0.0])

    out = tied(values)

    assert out[0] == out[1] == out[2]  # float noise: one tie
    assert out[3] > out[0]  # a real difference stays
    assert out[4] == 0.0
    assert tied(np.stack([values, values])).shape == (2, 5)
