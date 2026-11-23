"""Tests for selector.combine (D2: logistic-regression stacking)."""

from __future__ import annotations

from common.evaluate import auroc
import numpy as np
import pandas as pd
import pytest
from selector.combine import (
    compare_stack,
    cross_validated_comparison,
    fit_stack,
    stack_uncertainty,
    stack_weights,
)
from selector.split import subset
from selector.table import ScoreTable


def make_table(n: int = 400, seed: int = 0) -> ScoreTable:
    """Two methods that each see a different half of the signal, plus a useless one."""
    rng = np.random.default_rng(seed)
    correct = rng.random(n) < 0.4
    wrong = (~correct).astype(float)
    part = rng.random(n) < 0.5  # which questions each method can "see"
    columns = {
        "view_a": wrong * part + rng.normal(0, 0.4, n),
        "view_b": wrong * (~part) + rng.normal(0, 0.4, n),
        "noise": rng.normal(0, 1, n),
    }
    ids = list(range(n))
    return ScoreTable(
        model="m",
        run="r",
        scores=pd.DataFrame(columns, index=pd.Index(ids)),
        correct=pd.Series(correct, index=ids, name="correct"),
        info=pd.DataFrame(),
    )


def test_stack_combines_two_complementary_methods() -> None:
    table = make_table()
    train, test = subset(table, list(range(200))), subset(table, list(range(200, 400)))
    methods = ["view_a", "view_b", "noise"]
    u = stack_uncertainty(fit_stack(train, methods), test, methods)
    y = test.correct.to_numpy(dtype=bool)
    assert u.shape == (200,)
    assert u.min() >= 0
    assert u.max() <= 1
    single = max(auroc(test.scores[m].to_numpy(dtype=float), y) for m in methods)
    assert auroc(u, y) > single + 0.05


def test_informative_methods_get_negative_weights_noise_a_small_one() -> None:
    table = make_table()
    methods = ["view_a", "view_b", "noise"]
    weights = stack_weights(fit_stack(table, methods), methods)
    assert set(weights) == set(methods)
    assert weights["view_a"] < 0
    assert weights["view_b"] < 0
    assert abs(weights["noise"]) < min(abs(weights["view_a"]), abs(weights["view_b"]))


def test_cross_validated_comparison_is_deterministic_and_favours_the_stack_here() -> None:
    table = make_table(n=200)
    methods = ["view_a", "view_b", "noise"]
    first = cross_validated_comparison(table, methods, seed=1)
    assert first == cross_validated_comparison(table, methods, seed=1)
    stack, best = first
    assert 0 <= best <= 1
    assert stack > best


def test_compare_stack_reports_both_parts() -> None:
    table = make_table()
    report = compare_stack(subset(table, list(range(200))), subset(table, list(range(200, 400))), n_boot=200)
    assert report.best in {"view_a", "view_b"}
    assert report.methods == ["view_a", "view_b", "noise"]
    d = report.held_out_difference
    assert d["low"] <= d["diff"] <= d["high"]
    assert d["diff"] == pytest.approx(report.held_out_stack_auroc - report.held_out_best_auroc)
    assert report.cv_stack_auroc > report.cv_best_auroc
