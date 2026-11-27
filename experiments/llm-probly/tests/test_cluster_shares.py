"""Tests for the cluster-share features (D2): largest cluster share and greedy cluster share."""

from __future__ import annotations

from common import nli_matrix
from common.io import validate_scores
from common.nli_matrix import NliSettings, cluster_shares
from common.registry import REGISTRY, check_registry
from methods import _se_common, greedy_cluster_share, largest_cluster_share
import pytest
import torch


def matrix_from_groups(groups: list[str]) -> torch.Tensor:
    """NLI matrix where texts of the same group entail each other and all others contradict."""
    n = len(groups)
    matrix = torch.zeros((n, n, 3))
    for a in range(n):
        for b in range(n):
            matrix[a, b, 2 if groups[a] == groups[b] else 0] = 1.0
    return matrix


def test_shares_of_a_mixed_question():
    # greedy answer "a", then 10 samples: 2 x a, 5 x b, 3 x c
    matrix = matrix_from_groups(["a", "a", "b", "b", "a", "b", "c", "b", "c", "b", "c"])
    largest, greedy = cluster_shares(matrix)
    assert largest == pytest.approx(0.5)  # the 5 samples of "b"
    assert greedy == pytest.approx(0.2)  # the 2 samples of "a", like the greedy answer


def test_greedy_answer_no_sample_agrees_with():
    matrix = matrix_from_groups(["x", "a", "a", "a", "b"])
    assert cluster_shares(matrix) == (pytest.approx(0.75), 0.0)


def test_all_samples_agree():
    matrix = matrix_from_groups(["a"] * 11)
    assert cluster_shares(matrix) == (1.0, 1.0)


def test_neutral_in_both_directions_is_not_the_same_meaning():
    """Same rule as SE: neutral both ways keeps two answers apart."""
    matrix = matrix_from_groups(["a", "a", "a"])
    matrix[0, 2] = torch.tensor([0.0, 1.0, 0.0])
    matrix[2, 0] = torch.tensor([0.0, 1.0, 0.0])
    assert cluster_shares(matrix)[1] == pytest.approx(0.5)


class FirstLetterNli:
    """Stand-in NLI model: answers with the same first letter entail each other."""

    def pairwise_nli_many(self, groups, contexts=None, deduplicate=True):
        """One matrix per group (probly order: contradiction, neutral, entailment)."""
        del contexts, deduplicate
        return [matrix_from_groups([text[0] for text in group]) for group in groups]


def test_plugins_give_valid_records(monkeypatch):
    settings = NliSettings()
    monkeypatch.setattr(nli_matrix, "_settings", settings)
    monkeypatch.setattr(nli_matrix, "_cache_file", None)
    monkeypatch.setattr(_se_common, "_clusterers", {settings.hf_name: FirstLetterNli()})
    monkeypatch.setattr(_se_common, "_matrices", {})
    monkeypatch.setattr(_se_common, "_disk", {})
    monkeypatch.setattr(_se_common, "_nli_seconds", {})
    rows = [
        {
            "id": 0,
            "question": "Q0?",
            "greedy_answer": "Paris",
            "samples": ["Paris", "Pau", "Lyon", "Lille"],
            "samples_ll_mean": [-0.1] * 4,
            "model": "m",
        },
        {
            "id": 1,
            "question": "Q1?",
            "greedy_answer": "Rome",
            "samples": ["Milan", "Milan", "Milan", "Naples"],
            "samples_ll_mean": [-0.1] * 4,
            "model": "m",
        },
    ]
    largest = largest_cluster_share.score(rows)
    greedy = greedy_cluster_share.score(rows)

    assert validate_scores(largest, [0, 1]) == []
    assert validate_scores(greedy, [0, 1]) == []
    assert largest["uncertainty"] == pytest.approx([0.5, 0.25])  # 1 - 2/4, 1 - 3/4
    assert greedy["uncertainty"] == pytest.approx([0.5, 1.0])  # 1 - 2/4, no sample like "Rome"
    assert greedy["nli_pairs_per_question"] == 0  # the second plugin reuses the matrices


def test_registered_like_the_other_matrix_methods():
    assert check_registry() == []
    for name in ("largest_cluster_share", "greedy_cluster_share"):
        spec = REGISTRY[name]
        assert spec.uses_nli
        assert spec.cost == REGISTRY["discrete_semantic_entropy"].cost
