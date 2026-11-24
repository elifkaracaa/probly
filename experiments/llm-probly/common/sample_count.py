"""Sample-count optimization for the semantic methods (B2).

Generation is the most expensive step of the semantic methods: every question
needs K sampled answers. This module answers two questions from the cache only,
without new generations and without new NLI calls:

1. How much quality do semantic entropy (SE), discrete SE and lexical similarity
   lose with fewer samples? :func:`k_curve` draws random subsets of K = 2..10 of
   the 10 cached samples and scores each subset.
2. Can we stop sampling early? :func:`early_stop_count` simulates the rule "after
   ``check_at`` samples, stop if they all fall into one semantic cluster".

Scores are passed through :func:`tied` before AUROC / PRR, so values that differ
only by float noise count as exact ties.

Every subset reads its NLI labels from the shared per-question matrix (A1), so a
subset of samples gives exactly the clusters the clusterer would give on those
samples alone.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from methods.lexical_similarity import rouge_l
import numpy as np
import torch

from common.nli_matrix import GREEDY_INDEX, sample_block, se_and_dse
from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer

if TYPE_CHECKING:
    from collections.abc import Sequence

TIE_TOLERANCE = 1e-5
"""Scores closer than this are treated as exactly equal before AUROC / PRR.

Discrete SE takes only a few distinct values (for example log 2 for two equal
clusters of samples); two different values are always at least 2e-4 apart for up
to 10 samples. probly computes them in float32, so the same value can come out
with noise of about 1e-7, and that noise would decide how ties are ordered and
move AUROC by up to 0.002 between machines. Rounding to a fixed number of
decimals is not enough: a value close to a rounding boundary can still fall on
either side.
"""


def tied(values: np.ndarray, tolerance: float = TIE_TOLERANCE) -> np.ndarray:
    """Make scores that differ by less than ``tolerance`` exactly equal.

    Sorted scores are grouped while consecutive gaps stay below ``tolerance``;
    every score in a group is replaced by the group's smallest score.

    Args:
        values: Scores, any shape; the last axis holds the questions.
        tolerance: Largest gap that still counts as float noise.

    Returns:
        Array of the same shape with near-equal scores made equal.
    """
    values = np.asarray(values, dtype=float)
    flat = values.reshape(-1, values.shape[-1])
    out = np.empty_like(flat)
    for i, row in enumerate(flat):
        order = np.argsort(row, kind="stable")
        ranked = row[order]
        group_start = np.concatenate(([True], np.diff(ranked) >= tolerance))
        snapped = ranked[np.flatnonzero(group_start)[np.cumsum(group_start) - 1]]
        out[i, order] = snapped
    return out.reshape(values.shape)


METHODS = ("semantic_entropy", "discrete_semantic_entropy", "lexical_similarity")
"""Methods scored by :func:`subset_scores`, in this order."""


def subsample_matrix(matrix: torch.Tensor, keep: Sequence[int]) -> torch.Tensor:
    """Keep the greedy answer and the chosen samples of a question matrix.

    Args:
        matrix: Question matrix of shape ``(K+1, K+1, 3)``; index 0 is the greedy answer.
        keep: Positions of the samples to keep, counted from 0 among the samples
            (sample ``k`` is row ``k + 1`` of ``matrix``). The order is kept.

    Returns:
        Matrix of shape ``(len(keep)+1, len(keep)+1, 3)`` with the greedy answer first.
    """
    index = torch.tensor([GREEDY_INDEX, *(int(k) + 1 for k in keep)], dtype=torch.long)
    return matrix[index][:, index]


def lexical_uncertainty(samples: Sequence[str]) -> float:
    """Minus the mean pairwise ROUGE-L of the samples (same as the lexical_similarity plugin).

    Args:
        samples: At least two sampled answers.

    Returns:
        Uncertainty in ``[-1, 0]``; higher means the samples share fewer words.
    """
    pairs = list(combinations(samples, 2))
    if not pairs:
        msg = "lexical similarity needs at least two samples."
        raise ValueError(msg)
    return -sum(rouge_l(a, b) for a, b in pairs) / len(pairs)


def subset_scores(matrix: torch.Tensor, row: dict, keep: Sequence[int]) -> tuple[float, float, float]:
    """Score one question with only the chosen samples.

    Args:
        matrix: Full question matrix ``(K+1, K+1, 3)`` in probly's label order.
        row: The question's row of the data cache (``samples`` and ``samples_ll_mean``).
        keep: Positions of the samples to use.

    Returns:
        Uncertainties in the order of :data:`METHODS`.
    """
    keep = list(keep)
    se, dse = se_and_dse(subsample_matrix(matrix, keep), [row["samples_ll_mean"][k] for k in keep])
    return se, dse, lexical_uncertainty([row["samples"][k] for k in keep])


def draw_subsets(n_samples: int, k: int, n_draws: int, rng: np.random.Generator) -> list[list[int]]:
    """Random subsets of ``k`` out of ``n_samples`` sample positions, each sorted.

    Sorting keeps the generation order, so the greedy clusterer sees the samples
    in the order they were produced. With ``k == n_samples`` there is only one
    subset, so a single draw is returned.

    Args:
        n_samples: Number of cached samples per question.
        k: Subset size, ``2 <= k <= n_samples``.
        n_draws: Number of random subsets.
        rng: Random number generator.

    Returns:
        List of sorted position lists.
    """
    if not 2 <= k <= n_samples:
        msg = f"k must be between 2 and {n_samples}, got {k}."
        raise ValueError(msg)
    if k == n_samples:
        return [list(range(n_samples))]
    return [sorted(rng.choice(n_samples, size=k, replace=False).tolist()) for _ in range(n_draws)]


def k_curve(
    matrices: Sequence[torch.Tensor],
    rows: Sequence[dict],
    ks: Sequence[int],
    n_draws: int = 20,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """Uncertainties of every method for every subset size.

    For each K, the same random subset positions are used for every question in a
    draw, so a draw behaves like a run of the benchmark with K samples.

    Args:
        matrices: One question matrix per row (probly's label order).
        rows: Rows of the data cache, same order as ``matrices``.
        ks: Subset sizes to try.
        n_draws: Random subsets per K.
        seed: Seed of the subset draws.

    Returns:
        ``{K: array of shape (n_draws_K, len(METHODS), n_questions)}``.
    """
    n_samples = len(rows[0]["samples"])
    rng = np.random.default_rng(seed)
    curve = {}
    for k in ks:
        subsets = draw_subsets(n_samples, k, n_draws, rng)
        values = np.empty((len(subsets), len(METHODS), len(rows)))
        for d, keep in enumerate(subsets):
            for q, (matrix, row) in enumerate(zip(matrices, rows, strict=True)):
                values[d, :, q] = subset_scores(matrix, row, keep)
        curve[k] = values
    return curve


def early_stop_count(matrix: torch.Tensor, check_at: int) -> int:
    """Number of samples the early-stop rule would generate for one question.

    Rule: generate ``check_at`` samples; if they all fall into one semantic
    cluster, stop there, otherwise generate all samples.

    Args:
        matrix: Question matrix ``(K+1, K+1, 3)`` in probly's label order.
        check_at: Number of samples after which the rule checks agreement.

    Returns:
        ``check_at`` if the first ``check_at`` samples agree, else ``K``.
    """
    n_samples = matrix.shape[0] - 1
    if not 2 <= check_at <= n_samples:
        msg = f"check_at must be between 2 and {n_samples}, got {check_at}."
        raise ValueError(msg)
    first = sample_block(subsample_matrix(matrix, range(check_at)))
    group_ids = HFGreedySemanticClusterer.cluster_nli_matrix(first)
    return check_at if int(group_ids.max()) == 0 else n_samples


def early_stop_scores(
    matrices: Sequence[torch.Tensor],
    rows: Sequence[dict],
    check_at: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scores and sample counts when every question uses the early-stop rule.

    Args:
        matrices: One question matrix per row.
        rows: Rows of the data cache, same order.
        check_at: See :func:`early_stop_count`.

    Returns:
        ``(values, used)``: ``values`` has shape ``(len(METHODS), n_questions)``,
        ``used`` holds the number of samples generated per question.
    """
    used = np.array([early_stop_count(m, check_at) for m in matrices])
    values = np.empty((len(METHODS), len(rows)))
    for q, (matrix, row, n) in enumerate(zip(matrices, rows, used, strict=True)):
        values[:, q] = subset_scores(matrix, row, range(n))
    return values, used
