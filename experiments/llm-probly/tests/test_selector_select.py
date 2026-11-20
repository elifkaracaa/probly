"""Tests for selector.select (D1, step 4: the selection rule)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from selector.profile import load_profile, save_profile
from selector.select import build_profile, filter_candidates, paired_differences, select_method
from selector.split import Split
from selector.table import ScoreTable

if TYPE_CHECKING:
    from pathlib import Path

REGISTRY = {
    "strong_costly": {"family": "self_eval", "access": "logprobs", "cost": 10.0},
    "strong_cheap": {"family": "baseline", "access": "logprobs", "cost": 1.0},
    "weak_cheap": {"family": "baseline", "access": "logprobs", "cost": 1.0},
    "text_method": {"family": "agreement", "access": "text_only", "cost": 5.0},
}


def make_table(n: int = 200) -> ScoreTable:
    """Two good methods (same signal, different noise), one weak one, one text-only one."""
    rng = np.random.default_rng(0)
    correct = np.arange(n) < n // 2
    wrong = (~correct).astype(float)
    signal = wrong + rng.normal(0, 0.5, n)
    columns = {
        "strong_costly": signal + rng.normal(0, 0.05, n),
        "strong_cheap": signal + rng.normal(0, 0.05, n),
        "weak_cheap": wrong + rng.normal(0, 3.0, n),
        "text_method": wrong + rng.normal(0, 1.0, n),
    }
    ids = list(range(n))
    info = pd.DataFrame({"probability": [False] * 4, "seconds_per_question": [0.0] * 4}, index=list(columns))
    return ScoreTable(
        model="m",
        run="triviaqa_200",
        scores=pd.DataFrame(columns, index=ids),
        correct=pd.Series(correct, index=ids, name="correct"),
        info=info,
    )


def test_filter_by_access_budget_and_registry() -> None:
    methods = [*REGISTRY, "unknown"]
    candidates, excluded = filter_candidates(methods, REGISTRY, access="text_only")
    assert candidates == ["text_method"]
    assert "needs logprobs" in excluded["strong_cheap"]
    assert "not in the registry" in excluded["unknown"]
    candidates, excluded = filter_candidates(methods, REGISTRY, access="white_box", max_cost=5)
    assert candidates == ["strong_cheap", "weak_cheap", "text_method"]
    assert "budget" in excluded["strong_costly"]


def test_identical_method_is_tied_and_clearly_worse_method_is_not() -> None:
    table = make_table()
    diffs = paired_differences(table, "strong_costly", ["strong_cheap", "weak_cheap"], n_boot=200)
    assert diffs["strong_cheap"]["low"] < 0 < diffs["strong_cheap"]["high"]
    assert diffs["weak_cheap"]["high"] < 0


def test_cheapest_tied_method_is_chosen() -> None:
    selection = select_method(make_table(), REGISTRY, n_boot=200)
    assert {selection.best, selection.chosen} <= {"strong_costly", "strong_cheap"}
    assert selection.chosen == "strong_cheap"
    assert "weak_cheap" not in selection.tied
    assert selection.reason


def test_budget_changes_the_candidates() -> None:
    selection = select_method(make_table(), REGISTRY, access="text_only", n_boot=200)
    assert selection.chosen == "text_method"
    assert set(selection.excluded) == {"strong_costly", "strong_cheap", "weak_cheap"}


def test_no_candidate_left_raises() -> None:
    with pytest.raises(ValueError, match="no candidate"):
        select_method(make_table(), REGISTRY, access="text_only", max_cost=1, n_boot=50)


def test_selection_becomes_a_valid_profile(tmp_path: Path) -> None:
    table = make_table()
    selection = select_method(table, REGISTRY, n_boot=200)
    split = Split(calibration=list(range(100)), held_out=list(range(100, 200)), seed=0)
    profile = build_profile(selection, table, split, REGISTRY, model_id="org/m")
    assert profile.method.name == selection.chosen
    assert profile.method.cost == 1.0
    assert set(profile.candidates) == set(REGISTRY)
    assert load_profile(save_profile(profile, tmp_path / "m.json")) == profile
