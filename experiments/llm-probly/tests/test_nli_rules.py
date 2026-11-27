"""Known-answer tests for the equivalence rules and greedy clustering in common/nli_matrix.py."""

from __future__ import annotations

from common.nli_matrix import RULES, as_partition, greedy_clusters, same_meaning
import pytest
import torch

CONTRADICTION, NEUTRAL, ENTAILMENT = 0, 1, 2  # probly's label order


def pair(ab: int, ba: int) -> torch.Tensor:
    """A 2-text matrix with one-hot labels for a->b and b->a."""
    matrix = torch.zeros((2, 2, 3))
    matrix[0, 0, ENTAILMENT] = matrix[1, 1, ENTAILMENT] = 1.0
    matrix[0, 1, ab] = 1.0
    matrix[1, 0, ba] = 1.0
    return matrix


@pytest.mark.parametrize(
    ("ab", "ba", "expected"),
    [
        (ENTAILMENT, ENTAILMENT, {"probly": True, "strict": True, "lenient": True}),
        (ENTAILMENT, NEUTRAL, {"probly": True, "strict": False, "lenient": True}),
        (NEUTRAL, NEUTRAL, {"probly": False, "strict": False, "lenient": True}),
        (ENTAILMENT, CONTRADICTION, {"probly": False, "strict": False, "lenient": False}),
    ],
)
def test_same_meaning_rules(ab, ba, expected):
    assert set(expected) == set(RULES)
    assert {rule: same_meaning(pair(ab, ba), 0, 1, rule) for rule in RULES} == expected


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="rule must be one of"):
        same_meaning(pair(ENTAILMENT, ENTAILMENT), 0, 1, "other")


def test_greedy_clusters_join_the_first_matching_cluster():
    texts = ["Paris", "Paris", "Lyon", "Paris"]  # index 0 = greedy answer
    matrix = torch.zeros((4, 4, 3))
    for a in range(4):
        for b in range(4):
            matrix[a, b, ENTAILMENT if texts[a] == texts[b] else CONTRADICTION] = 1.0
    assert greedy_clusters(matrix, range(1, 4)) == [[1, 3], [2]]
    assert greedy_clusters(matrix, range(4)) == [[0, 1, 3], [2]]


def test_as_partition_ignores_order_and_ids():
    assert as_partition([[1, 3], [2]]) == as_partition([[2], [3, 1]])
    assert as_partition([[1, 2], [3]]) != as_partition([[1], [2, 3]])
