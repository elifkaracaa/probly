"""Tests for selector.split (step 2: calibration / held-out split)."""

from __future__ import annotations

import sys

import pandas as pd
import pytest
from selector import split as split_module
from selector.split import make_split, subset
from selector.table import ScoreTable


def make_table(n: int = 200, n_correct: int = 40) -> ScoreTable:
    """Table with one method and ``n_correct`` correct answers out of ``n``."""
    ids = list(range(n))
    correct = pd.Series([i < n_correct for i in ids], index=ids, name="correct")
    scores = pd.DataFrame({"msp": [float(i) for i in ids]}, index=pd.Index(ids))
    return ScoreTable(model="m", run="r", scores=scores, correct=correct, info=pd.DataFrame())


def test_parts_are_disjoint_and_cover_all_questions() -> None:
    split = make_split(make_table(), calibration_size=100)
    assert len(split.calibration) == 100
    assert len(split.held_out) == 100
    assert not set(split.calibration) & set(split.held_out)
    assert sorted(split.calibration + split.held_out) == list(range(200))


def test_split_keeps_the_share_of_correct_answers() -> None:
    table = make_table(n=200, n_correct=40)  # 20 % correct
    split = make_split(table, calibration_size=100)
    assert table.correct.loc[split.calibration].sum() == 20
    assert table.correct.loc[split.held_out].sum() == 20


def test_same_seed_same_split_other_seed_other_split() -> None:
    table = make_table()
    assert make_split(table, seed=0) == make_split(table, seed=0)
    assert make_split(table, seed=0).calibration != make_split(table, seed=1).calibration


def test_too_few_correct_answers_raise() -> None:
    with pytest.raises(ValueError, match="at least"):
        make_split(make_table(n=200, n_correct=6), calibration_size=100)


def test_bad_size_raises() -> None:
    with pytest.raises(ValueError, match="calibration_size"):
        make_split(make_table(n=200), calibration_size=200)


def test_subset_keeps_scores_and_labels_aligned() -> None:
    table = make_table()
    part = subset(table, [3, 50, 7])
    assert list(part.scores.index) == [3, 50, 7]
    assert list(part.correct.index) == [3, 50, 7]
    assert part.scores.loc[50, "msp"] == 50.0


def test_main_prints_sizes_and_share_of_correct_answers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(split_module, "load_score_table", lambda *_args, **_kwargs: make_table())
    monkeypatch.setattr(sys, "argv", ["split", "--calibration-size", "100", "--seed", "3"])
    split_module.main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "m / r, seed 3"
    assert "calibration" in lines[1]
    assert "100 questions" in lines[1]
    assert "20 correct (20.0%)" in lines[1]
    assert "held-out" in lines[2]
    assert "20 correct (20.0%)" in lines[2]
