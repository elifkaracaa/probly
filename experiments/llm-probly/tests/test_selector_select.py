"""Tests for selector.select (D1, step 4: the selection rule)."""

from __future__ import annotations

from dataclasses import replace
import json
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from selector import select as select_module
from selector.profile import load_profile, save_profile
from selector.select import (
    Selection,
    build_profile,
    explain,
    filter_candidates,
    paired_differences,
    print_selection,
    select_method,
)
from selector.split import Split
from selector.table import ScoreTable

if TYPE_CHECKING:
    from pathlib import Path

    from selector.profile import Profile

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
    info = pd.DataFrame({"probability": [False] * 4, "seconds_per_question": [0.0] * 4}, index=pd.Index(list(columns)))
    return ScoreTable(
        model="m",
        run="triviaqa_200",
        scores=pd.DataFrame(columns, index=pd.Index(ids)),
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


def test_methods_with_many_failed_questions_are_excluded() -> None:
    table = make_table()
    info = table.info.assign(failed_fraction=[0.0, 0.07, 0.0, 0.0])
    selection = select_method(replace(table, info=info), REGISTRY, n_boot=200)
    assert "strong_cheap" not in selection.ranking.index
    assert "failed on 7%" in selection.excluded["strong_cheap"]
    assert selection.chosen == "strong_costly"


def test_selection_becomes_a_valid_profile(tmp_path: Path) -> None:
    table = make_table()
    selection = select_method(table, REGISTRY, n_boot=200)
    split = Split(calibration=list(range(0, 200, 2)), held_out=list(range(1, 200, 2)), seed=0)
    profile = build_profile(selection, table, split, REGISTRY, model_id="org/m", n_boot=200)
    assert profile.held_out_metrics is not None
    assert profile.held_out_metrics.auroc_low <= profile.held_out_metrics.auroc <= profile.held_out_metrics.auroc_high
    assert profile.method.name == selection.chosen
    assert profile.method.cost == 1.0
    assert set(profile.candidates) == set(REGISTRY)
    assert load_profile(save_profile(profile, tmp_path / "m.json")) == profile


def test_unknown_access_level_raises() -> None:
    with pytest.raises(ValueError, match="access must be one of"):
        filter_candidates(list(REGISTRY), REGISTRY, access="cloud")


def make_selection(chosen: str, tied: list[str]) -> Selection:
    """A selection of two methods: ``a`` (best, cost 10) and ``b`` (cost 1)."""
    return Selection(
        model="m",
        chosen=chosen,
        best="a",
        tied=tied,
        differences={"b": {"diff": -0.01, "low": -0.05, "high": 0.03}},
        ranking=pd.DataFrame({"auroc": [0.90, 0.89]}, index=pd.Index(["a", "b"])),
    )


SMALL_REGISTRY = {"a": {"cost": 10.0}, "b": {"cost": 1.0}}


def test_explain_best_method_without_ties() -> None:
    reason = explain(make_selection("a", tied=[]), SMALL_REGISTRY)
    assert "highest AUROC (0.900)" in reason
    assert "every other method is significantly worse" in reason


def test_explain_best_method_when_tied_methods_are_not_cheaper() -> None:
    registry = {"a": {"cost": 1.0}, "b": {"cost": 10.0}}
    reason = explain(make_selection("a", tied=["b"]), registry)
    assert "the tied methods are not cheaper" in reason


def test_explain_best_method_when_a_cheaper_method_is_tied() -> None:
    # Not what select_method returns (it would pick b), but explain must still give a sentence.
    reason = explain(make_selection("a", tied=["b"]), SMALL_REGISTRY)
    assert reason == "a has the highest AUROC (0.900)."


def test_explain_cheaper_tied_method_gives_difference_and_cost_ratio() -> None:
    reason = explain(make_selection("b", tied=["b"]), SMALL_REGISTRY)
    assert "b (AUROC 0.890) is not significantly worse than the best, a" in reason
    assert "[-0.050, +0.030]" in reason
    assert "costs 10x less" in reason


def test_build_profile_reads_the_model_id_from_the_result_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    table = make_table()
    selection = select_method(table, REGISTRY, n_boot=50)
    folder = tmp_path / table.model / table.run
    folder.mkdir(parents=True)
    (folder / f"{selection.chosen}.json").write_text(json.dumps({"model": "org/from-file"}))
    monkeypatch.setattr(select_module, "RESULTS_ROOT", tmp_path)
    split = Split(calibration=list(range(0, 200, 2)), held_out=list(range(1, 200, 2)), seed=0)
    profile = build_profile(selection, table, split, REGISTRY, n_boot=50)
    assert profile.model_id == "org/from-file"


def test_print_selection_marks_best_tied_and_excluded_methods(capsys: pytest.CaptureFixture[str]) -> None:
    selection = select_method(make_table(), REGISTRY, access="white_box", max_cost=5, n_boot=100)
    selection.excluded["strong_costly"] = "cost 10 > budget 5"
    print_selection(selection, REGISTRY)
    out = capsys.readouterr().out
    assert f"m: chosen {selection.chosen}" in out
    assert "(best)" in out
    assert "yes" in out or "no" in out
    assert "excluded strong_costly: cost 10 > budget 5" in out


def run_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *args: str) -> None:
    """Run ``selector.select.main`` on the in-memory table, with the test registry and no real files."""
    monkeypatch.setattr(select_module, "load_score_table", lambda *_args, **_kwargs: make_table())
    monkeypatch.setattr(select_module, "DEFAULT_REGISTRY", REGISTRY)
    monkeypatch.setattr(select_module, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["select", "--models", "m", "--n-boot", "50", *args])
    select_module.main()


def test_main_prints_the_choice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_main(monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert "m: chosen " in out
    assert "saved" not in out


def test_main_with_save_calibrates_and_writes_the_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "m" / "triviaqa_200"
    folder.mkdir(parents=True)
    for method in REGISTRY:
        (folder / f"{method}.json").write_text(json.dumps({"model": "org/m"}))
    saved: list[Profile] = []

    def fake_save(profile: Profile) -> Path:
        saved.append(profile)
        return tmp_path / "m.json"

    monkeypatch.setattr(select_module, "save_profile", fake_save)
    run_main(monkeypatch, tmp_path, "--save")
    out = capsys.readouterr().out
    assert "calibration: " in out
    assert f"saved {tmp_path / 'm.json'}" in out
    assert len(saved) == 1
    assert saved[0].model == "m"
