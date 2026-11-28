"""Tests for selector.rank (D1, step 3: AUROC and PRR with bootstrap intervals)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from selector import rank as rank_module
from selector.rank import bootstrap_indices, rank_methods
from selector.table import ScoreTable


def make_table(columns: dict[str, list[float]], correct: list[bool]) -> ScoreTable:
    """Table with the given method scores and labels."""
    ids = list(range(len(correct)))
    return ScoreTable(
        model="m",
        run="r",
        scores=pd.DataFrame(columns, index=pd.Index(ids)),
        correct=pd.Series(correct, index=ids, name="correct"),
        info=pd.DataFrame(),
    )


def three_methods(n: int = 100) -> ScoreTable:
    """A perfect, a noisy and a useless method on the same questions (30 % correct)."""
    rng = np.random.default_rng(0)
    correct = [i < 0.3 * n for i in range(n)]
    wrong = np.array([not c for c in correct], dtype=float)
    return make_table(
        {
            "perfect": list(wrong + 0.01 * rng.random(n)),  # wrong answers always more uncertain
            "noisy": list(wrong + rng.normal(0, 0.8, n)),
            "random": list(rng.random(n)),
        },
        correct,
    )


def test_ranking_is_sorted_by_auroc_and_has_all_columns() -> None:
    ranking = rank_methods(three_methods(), n_boot=200)
    assert list(ranking.index) == ["perfect", "noisy", "random"]
    assert list(ranking.columns) == ["auroc", "auroc_low", "auroc_high", "prr", "prr_low", "prr_high"]


def test_perfect_method_scores_one_and_random_is_near_chance() -> None:
    ranking = rank_methods(three_methods(), n_boot=200)
    assert ranking.loc["perfect", "auroc"] == pytest.approx(1.0)
    assert ranking.loc["perfect", "prr"] == pytest.approx(1.0)
    assert ranking.loc["random", "auroc_low"] < 0.5 < ranking.loc["random", "auroc_high"]


def test_intervals_contain_the_point_estimate() -> None:
    ranking = rank_methods(three_methods(), n_boot=200)
    for metric in ("auroc", "prr"):
        assert (ranking[f"{metric}_low"] <= ranking[metric]).all()
        assert (ranking[metric] <= ranking[f"{metric}_high"]).all()


def test_same_seed_gives_same_intervals() -> None:
    table = three_methods()
    pd.testing.assert_frame_equal(rank_methods(table, n_boot=100, seed=3), rank_methods(table, n_boot=100, seed=3))


def test_resamples_with_one_class_are_dropped() -> None:
    correct = np.array([True] + [False] * 9)  # one correct answer: many resamples miss it
    resamples = bootstrap_indices(correct, n_boot=200, seed=0)
    assert 0 < len(resamples) < 200
    assert all(correct[idx].any() and not correct[idx].all() for idx in resamples)


def test_single_class_table_raises() -> None:
    with pytest.raises(ValueError, match="one class"):
        rank_methods(make_table({"msp": [0.1, 0.2, 0.3]}, [True, True, True]), n_boot=20)


def test_table_without_methods_raises() -> None:
    with pytest.raises(ValueError, match="no candidate methods"):
        rank_methods(make_table({}, [True, False]), n_boot=20)


def test_main_prints_one_line_per_method(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(rank_module, "load_score_table", lambda *_args, **_kwargs: three_methods())
    monkeypatch.setattr(sys, "argv", ["rank", "--calibration-size", "50", "--n-boot", "50"])
    rank_module.main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "m / r: calibration part, 50 questions, 15 correct, 50 bootstrap resamples"
    assert lines[1].split()[0] == "method"
    assert lines[2].split()[0] == "perfect"  # best method first
    assert {line.split()[0] for line in lines[2:]} == {"perfect", "noisy", "random"}
