"""Tests for the A4 hand-label tooling: pair selection, the same-meaning rule, the labelling loop."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from common.nli_labels import (
    candidate_pairs,
    load_labels,
    normalize,
    select_pairs,
    stratum_shares,
    weighted_agreement,
)
from common.nli_matrix import same_meaning
import pytest
from scripts.label_nli_pairs import label_pairs
import torch

if TYPE_CHECKING:
    from pathlib import Path

CONTRADICTION, NEUTRAL, ENTAILMENT = torch.eye(3)


def matrix_with(pairs: dict[tuple[int, int], torch.Tensor], n: int) -> torch.Tensor:
    """(n, n, 3) matrix, entailment everywhere except the given ordered pairs."""
    matrix = ENTAILMENT.repeat(n, n, 1)
    for (a, b), label in pairs.items():
        matrix[a, b] = label
    return matrix


def test_same_meaning_uses_probly_rule() -> None:
    matrix = matrix_with({(0, 2): NEUTRAL, (2, 0): NEUTRAL, (1, 2): CONTRADICTION, (0, 1): NEUTRAL}, 3)

    assert same_meaning(matrix, 0, 1)  # entailment one way, neutral the other
    assert not same_meaning(matrix, 0, 2)  # neutral both ways
    assert not same_meaning(matrix, 2, 1)  # contradiction in one direction


def test_strict_rule_needs_entailment_both_ways() -> None:
    matrix = matrix_with({(0, 1): NEUTRAL}, 3)  # 0 -> 1 neutral, 1 -> 0 entailment

    assert same_meaning(matrix, 0, 1, rule="probly")
    assert not same_meaning(matrix, 0, 1, rule="strict")
    assert same_meaning(matrix, 0, 2, rule="strict")
    with pytest.raises(ValueError, match="rule"):
        same_meaning(matrix, 0, 1, rule="lenient")


def test_normalize() -> None:
    assert normalize("  Paris. ") == normalize("paris") == "paris"
    assert normalize("U.S.  Navy") == "us navy"


def test_candidate_pairs_skip_trivial_and_long_answers_and_set_strata() -> None:
    row = {"id": 7, "question": "Q?", "greedy_answer": "Paris", "samples": ["paris.", "Lyon", "x" * 200]}
    base = matrix_with({(0, 2): NEUTRAL, (2, 0): NEUTRAL}, 4)  # base: Paris != Lyon
    large = matrix_with({}, 4)  # large: everything the same

    pairs = candidate_pairs("m", [row], [base], [large])

    assert [(p["index_a"], p["index_b"]) for p in pairs] == [(0, 2)]  # "paris." duplicates "Paris"; long one left out
    assert pairs[0]["stratum"] == "disagree"
    assert (pairs[0]["base_same"], pairs[0]["large_same"]) == (False, True)


def test_select_pairs_balances_strata_one_pair_per_question() -> None:
    candidates = [
        {"id": q, "stratum": stratum, "index_a": 0, "index_b": k}
        for q in range(30)
        for k, stratum in enumerate(["disagree", "both_same", "both_different"], start=1)
    ]

    picked = select_pairs(candidates, per_stratum=5, seed=1)

    assert Counter(p["stratum"] for p in picked) == {"disagree": 5, "both_same": 5, "both_different": 5}
    assert len({p["id"] for p in picked}) == 15
    assert picked == select_pairs(candidates, per_stratum=5, seed=1)


def test_label_pairs_saves_after_each_answer_and_resumes(tmp_path: Path) -> None:
    pairs = [
        {"model": "m", "id": i, "index_a": 0, "index_b": 1, "question": "Q?", "a": "A", "b": "B"} for i in range(4)
    ]
    path = tmp_path / "labels.json"
    answers = iter(["y", "maybe", "n", "s", "q"])  # "maybe" is not valid and is asked again

    assert label_pairs(pairs, "elif", path, ask=lambda _: next(answers)) == 3
    labels = load_labels(path)
    assert [label["same_meaning"] for label in labels] == [True, False, None]
    assert labels[0]["labeller"] == "elif"

    assert label_pairs(pairs, "elif", path, ask=lambda _: "y") == 1  # only the 4th pair is left
    assert len(load_labels(path)) == 4


def test_stratum_shares() -> None:
    candidates = [{"stratum": "both_different"}] * 6 + [{"stratum": "both_same"}] * 3 + [{"stratum": "disagree"}]

    assert stratum_shares(candidates) == pytest.approx({"both_different": 0.6, "both_same": 0.3, "disagree": 0.1})


def test_weighted_agreement_weights_strata_by_their_real_share() -> None:
    labels = [
        {"model": "m", "stratum": "disagree", "same_meaning": False},
        {"model": "m", "stratum": "disagree", "same_meaning": False},
        {"model": "m", "stratum": "both_same", "same_meaning": True},
        {"model": "m", "stratum": "both_same", "same_meaning": None},  # skipped pair, left out
    ]
    decisions = [True, True, True, False]  # wrong on both disagree pairs, right on the both_same pair
    shares = {"m": {"disagree": 0.1, "both_same": 0.9, "both_different": 0.0}}

    # plain accuracy would be 1/3; weighted: 0.1 * 0 + 0.9 * 1
    assert weighted_agreement(labels, decisions, shares) == pytest.approx(0.9)


def test_weighted_agreement_ignores_strata_and_models_without_labels() -> None:
    labels = [{"model": "m", "stratum": "both_same", "same_meaning": True}]
    shares = {
        "m": {"disagree": 0.1, "both_same": 0.2, "both_different": 0.7},
        "other": {"disagree": 0.1, "both_same": 0.2, "both_different": 0.7},
    }

    assert weighted_agreement(labels, [True], shares) == pytest.approx(1.0)  # not 0.2 / 2
    with pytest.raises(ValueError, match="no labelled pair"):
        weighted_agreement([], [], shares)
