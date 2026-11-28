"""Tests for the zero-shot and brainstorm p(True) plugins (methods/p_true_*.py), with a stub evaluator.

No model is loaded: ``HFSelfEvaluator`` is replaced by a stub that returns fixed probabilities, so the tests
check only what the plugins add on top of the library (score contract, failed rows, result fields).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from methods import p_true_brainstorm, p_true_zeroshot
import numpy as np
import pytest

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# The plugins only pass the model and tokenizer on to the evaluator, so plain objects are enough.
MODEL = cast("PreTrainedModel", object())
TOKENIZER = cast("PreTrainedTokenizerBase", object())


def make_rows(samples: list[list[str] | None] | None = None, n: int = 3) -> list[dict[str, Any]]:
    """Return ``n`` cached rows; ``samples`` adds the sampled answers that brainstorm needs."""
    rows: list[dict[str, Any]] = [
        {"id": i, "model": "fake-model", "question": f"Q{i}?", "greedy_answer": f"A{i}"} for i in range(n)
    ]
    if samples is not None:
        for row, sample in zip(rows, samples, strict=True):
            row["samples"] = sample
    return rows


class StubEvaluator:
    """Stand-in for HFSelfEvaluator: returns the given p_true and coverage and records how it was called."""

    def __init__(self, p_true: list[float], coverage: list[float] | None = None) -> None:
        self.p_true = p_true
        self.coverage = coverage if coverage is not None else [0.9] * len(p_true)
        self.init_kwargs: dict[str, Any] = {}
        self.evaluate_args: tuple = ()

    def factory(self, model: object, tok: object, **kwargs: Any) -> StubEvaluator:  # noqa: ANN401
        self.init_kwargs = {"model": model, "tok": tok, **kwargs}
        return self

    def evaluate(self, *args: Any) -> SimpleNamespace:  # noqa: ANN401
        self.evaluate_args = args
        return SimpleNamespace(p_true=np.array(self.p_true), coverage=np.array(self.coverage))


@pytest.mark.parametrize("plugin", [p_true_zeroshot, p_true_brainstorm])
def test_plugins_need_the_model(plugin):
    with pytest.raises(ValueError, match="needs the model"):
        plugin.score(make_rows(samples=[["x"]] * 3))
    with pytest.raises(ValueError, match="needs the model"):
        plugin.score(make_rows(samples=[["x"]] * 3), model=MODEL)


def test_zeroshot_uncertainty_is_one_minus_p_true(monkeypatch):
    stub = StubEvaluator([0.9, 0.25, 0.5], coverage=[0.99, 0.8, 0.1])
    monkeypatch.setattr(p_true_zeroshot, "HFSelfEvaluator", stub.factory)
    result = p_true_zeroshot.score(make_rows(), model=MODEL, tok=TOKENIZER)
    assert result["uncertainty"] == pytest.approx([0.1, 0.75, 0.5])
    assert result["failed"] == [False, False, False]
    assert result["coverage"] == pytest.approx([0.99, 0.8, 0.1])
    assert result["ids"] == [0, 1, 2]
    assert result["method"] == "p_true_zeroshot"
    assert result["model"] == "fake-model"
    assert result["answer"] == "greedy"
    assert result["probability"] is True
    assert result["seconds_per_question"] >= 0.0
    assert stub.init_kwargs == {"model": MODEL, "tok": TOKENIZER, "mode": "plain_ab", "batch_size": 8}
    assert stub.evaluate_args == (["Q0?", "Q1?", "Q2?"], ["A0", "A1", "A2"])


def test_zeroshot_nan_probability_is_failed_with_uncertainty_one(monkeypatch):
    stub = StubEvaluator([0.8, math.nan, 0.0])
    monkeypatch.setattr(p_true_zeroshot, "HFSelfEvaluator", stub.factory)
    result = p_true_zeroshot.score(make_rows(), model=MODEL, tok=TOKENIZER)
    assert result["uncertainty"] == pytest.approx([0.2, 1.0, 1.0])
    assert result["failed"] == [False, True, False]  # p_true = 0 is a valid answer, not a failure


def test_brainstorm_passes_the_cached_samples_to_the_prompt(monkeypatch):
    stub = StubEvaluator([0.7, 0.4])
    monkeypatch.setattr(p_true_brainstorm, "HFSelfEvaluator", stub.factory)
    rows = make_rows(samples=[["a", "b"], ["c"], ["d"]])
    result = p_true_brainstorm.score(rows[:2], model=MODEL, tok=TOKENIZER)
    assert stub.evaluate_args == (["Q0?", "Q1?"], ["A0", "A1"], [["a", "b"], ["c"]])
    assert result["uncertainty"] == pytest.approx([0.3, 0.6])
    assert result["failed"] == [False, False]
    assert result["method"] == "p_true_brainstorm"
    assert result["mode"] == "plain_ab"
    assert stub.init_kwargs["mode"] == "plain_ab"


def test_brainstorm_rows_without_samples_are_failed_and_not_evaluated(monkeypatch):
    stub = StubEvaluator([0.6], coverage=[0.95])  # only the one row with samples reaches the evaluator
    monkeypatch.setattr(p_true_brainstorm, "HFSelfEvaluator", stub.factory)
    rows = make_rows(samples=[["a"], None, []])  # None and an empty list both mean "no samples"
    result = p_true_brainstorm.score(rows, model=MODEL, tok=TOKENIZER)
    assert stub.evaluate_args == (["Q0?"], ["A0"], [["a"]])
    assert result["uncertainty"] == pytest.approx([0.4, 1.0, 1.0])
    assert result["failed"] == [False, True, True]
    assert result["coverage"][0] == pytest.approx(0.95)
    assert math.isnan(result["coverage"][1])
    assert math.isnan(result["coverage"][2])


def test_brainstorm_nan_probability_is_failed(monkeypatch):
    stub = StubEvaluator([math.nan, 0.5])
    monkeypatch.setattr(p_true_brainstorm, "HFSelfEvaluator", stub.factory)
    rows = make_rows(samples=[["a"], ["b"]], n=2)
    result = p_true_brainstorm.score(rows, model=MODEL, tok=TOKENIZER)
    assert result["uncertainty"] == pytest.approx([1.0, 0.5])
    assert result["failed"] == [True, False]
