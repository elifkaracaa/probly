"""Tests for selector.evaluate (held-out comparison of the selector with fixed methods)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from selector import evaluate
from selector.evaluate import ModelResult, common_methods, evaluate_model, split_robustness, to_markdown
from selector.table import ScoreTable

if TYPE_CHECKING:
    from collections.abc import Callable


def make_table(n: int = 200, seed: int = 0) -> ScoreTable:
    """Synthetic scores: msp is informative and cheap, p_true_zeroshot a bit worse, verbalized_2s noise."""
    rng = np.random.default_rng(seed)
    correct = rng.random(n) < 0.5
    wrong = (~correct).astype(float)
    columns = {
        "msp": wrong + rng.normal(0, 0.6, n),
        "p_true_zeroshot": wrong + rng.normal(0, 0.9, n),
        "verbalized_2s": rng.normal(0, 1, n),
    }
    ids = list(range(n))
    info = pd.DataFrame({"probability": [False, True, True]}, index=pd.Index(list(columns)))
    return ScoreTable(
        model="m",
        run="triviaqa_200",
        scores=pd.DataFrame(columns, index=pd.Index(ids)),
        correct=pd.Series(correct, index=ids, name="correct"),
        info=info,
    )


def result(model: str, chosen: str, aurocs: dict[str, float]) -> ModelResult:
    """A ModelResult with the oracle taken from ``aurocs``."""
    oracle = max(aurocs, key=lambda m: aurocs[m])
    return ModelResult(
        model=model,
        chosen=chosen,
        chosen_auroc=aurocs[chosen],
        chosen_cost=1.0,
        fixed_auroc=aurocs,
        oracle=oracle,
        difference={"diff": aurocs[chosen] - aurocs[oracle], "low": -0.1, "high": 0.0},
    )


@pytest.fixture
def fake_load(monkeypatch: pytest.MonkeyPatch) -> Callable[..., ScoreTable]:
    """Make evaluate_model read the synthetic table instead of the results folder."""

    def load(model: str, dataset: str = "triviaqa", n: int = 200) -> ScoreTable:  # noqa: ARG001
        return make_table(n)

    monkeypatch.setattr(evaluate, "load_score_table", load)
    return load


def test_selector_picks_the_strong_cheap_method_and_matches_the_oracle(fake_load: Callable[..., ScoreTable]) -> None:
    del fake_load
    r = evaluate_model("m", n_boot=200)
    assert r.chosen == "msp"
    assert r.oracle == "msp"
    assert r.difference == {"diff": 0.0, "low": 0.0, "high": 0.0}
    assert set(r.fixed_auroc) == {"msp", "p_true_zeroshot", "verbalized_2s"}
    assert r.chosen_auroc == r.fixed_auroc["msp"]


def test_common_methods_only_keeps_methods_of_every_model_best_first() -> None:
    results = [
        result("a", "msp", {"msp": 0.8, "p_true_zeroshot": 0.7, "semantic_entropy": 0.9}),
        result("b", "msp", {"msp": 0.6, "p_true_zeroshot": 0.75}),
    ]
    assert common_methods(results) == ["p_true_zeroshot", "msp"]


def test_markdown_has_one_row_per_model_and_per_fixed_method() -> None:
    results = [
        result("a", "msp", {"msp": 0.8, "p_true_zeroshot": 0.7}),
        result("b", "p_true_zeroshot", {"msp": 0.6, "p_true_zeroshot": 0.75}),
    ]
    report = to_markdown(results, "triviaqa", 200, 100)
    assert "| a | msp |" in report
    assert "| b | p_true_zeroshot |" in report
    assert "**Selector (per model)** | **0.775**" in report
    assert "| always p_true_zeroshot | 0.725 |" in report
    assert "| always msp | 0.700 |" in report


def test_split_robustness_counts_choices_and_compares_over_splits() -> None:
    runs = {
        0: [
            result("a", "msp", {"msp": 0.8, "p_true_zeroshot": 0.7}),
            result("b", "msp", {"msp": 0.6, "p_true_zeroshot": 0.8}),
        ],
        1: [
            result("a", "p_true_zeroshot", {"msp": 0.7, "p_true_zeroshot": 0.9}),
            result("b", "msp", {"msp": 0.7, "p_true_zeroshot": 0.7}),
        ],
    }
    report = split_robustness(runs)
    assert "Over 2 random splits (seeds 0-1)" in report
    assert "| a | msp (1), p_true_zeroshot (1) |" in report
    assert "| b | msp (2) |" in report
    # selector means: split 0 = 0.7, split 1 = 0.8; always msp: 0.7, 0.7 -> differences 0.0 and +0.1
    assert "| always msp | 0.700 (0.000) | +0.050 [+0.000, +0.100] |" in report
