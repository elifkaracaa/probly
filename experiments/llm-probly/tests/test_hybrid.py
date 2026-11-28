"""Tests for the hybrid CoCoA scores (C5) on hand-made matrices and the two plugins."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from common import nli_matrix
from common.hybrid import cocoa, cocoa_record, greedy_similarities, inconsistency
from common.io import validate_scores
from common.nli_matrix import NliSettings
from common.registry import REGISTRY
from methods import _se_common, cocoa_msp, cocoa_p_true
import pytest
from scripts.hybrid_from_results import parse_run
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

CONTRADICTION, NEUTRAL, ENTAILMENT = 0, 1, 2  # probly's label order


def matrix_from_groups(groups: list[str]) -> torch.Tensor:
    """NLI matrix where texts of the same group entail each other and all others contradict."""
    n = len(groups)
    matrix = torch.zeros((n, n, 3))
    for a in range(n):
        for b in range(n):
            matrix[a, b, ENTAILMENT if groups[a] == groups[b] else CONTRADICTION] = 1.0
    matrix[torch.arange(n), torch.arange(n)] = float("nan")  # as in the cache: the diagonal is not scored
    return matrix


def test_similarities_are_the_mean_of_both_directions():
    matrix = matrix_from_groups(["a", "a", "b"])
    matrix[0, 1] = torch.tensor([0.0, 0.5, 0.5])  # greedy -> sample 1 only half entailed
    assert greedy_similarities(matrix) == pytest.approx([0.75, 0.0])


def test_diagonal_nan_does_not_leak():
    assert all(math.isfinite(s) for s in greedy_similarities(matrix_from_groups(["a", "a", "a"])))


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        (["a", "a", "a", "a", "a"], 0.0),  # every sample repeats the greedy answer
        (["a", "b", "b", "b", "b"], 1.0),  # no sample agrees with the greedy answer
        (["a", "a", "b", "a", "c"], 0.5),
    ],
)
def test_inconsistency(groups, expected):
    assert inconsistency(matrix_from_groups(groups)) == pytest.approx(expected)


def test_cocoa_multiplies_confidence_and_inconsistency():
    matrix = matrix_from_groups(["a", "a", "b", "a", "c"])
    assert cocoa(0.8, matrix) == pytest.approx(0.4)
    assert cocoa(0.0, matrix) == 0.0  # fully confident answer
    assert cocoa(2.0, matrix_from_groups(["a", "a", "a"])) == 0.0  # samples all agree


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_cocoa_rejects_invalid_confidence(value):
    with pytest.raises(ValueError, match="confidence_uncertainty"):
        cocoa(value, matrix_from_groups(["a", "a"]))


def test_matrix_without_samples_is_rejected():
    with pytest.raises(ValueError, match="no samples"):
        greedy_similarities(torch.zeros((1, 1, 3)))
    with pytest.raises(ValueError, match="matrix"):
        greedy_similarities(torch.zeros((2, 3, 3)))


def test_record_marks_failed_ingredient_and_checks_ids():
    rows = [{"id": 0, "model": "m"}, {"id": 1, "model": "m"}]
    matrices = {0: matrix_from_groups(["a", "b"]), 1: matrix_from_groups(["a", "a"])}
    confidence = {
        "method": "msp",
        "ids": [0, 1],
        "uncertainty": [1.0, 0.3],
        "failed": [True, False],
        "seconds_per_question": 0.5,
    }
    record = cocoa_record("cocoa_x", rows, confidence, lambda row: matrices[row["id"]])
    assert validate_scores(record, [0, 1]) == []
    assert record["failed"] == [True, False]
    assert record["uncertainty"] == pytest.approx([1.0, 0.0])
    assert record["confidence"] == "msp"
    assert record["seconds_per_question"] == 0.5

    with pytest.raises(ValueError, match="other question ids"):
        cocoa_record("cocoa_x", rows, {**confidence, "ids": [1, 0]}, lambda row: matrices[row["id"]])


class FirstLetterNli:
    """Stand-in NLI model: answers with the same first letter entail each other."""

    def pairwise_nli_many(self, groups, contexts=None, deduplicate=True):
        """One matrix per group (probly order: contradiction, neutral, entailment)."""
        del contexts, deduplicate
        return [matrix_from_groups([text[0] for text in group]) for group in groups]


ROWS = [
    {
        "id": 0,
        "question": "Q0?",
        "greedy_answer": "Paris",
        "greedy_token_logprobs": [-0.5, -0.5],
        "samples": ["Paris", "Pau", "Lyon", "Lille"],
        "samples_ll_mean": [-0.1] * 4,
        "model": "m",
    },
    {
        "id": 1,
        "question": "Q1?",
        "greedy_answer": "Rome",
        "greedy_token_logprobs": [-2.0],
        "samples": ["Milan", "Milan", "Rome", "Naples"],
        "samples_ll_mean": [-0.1] * 4,
        "model": "m",
    },
]


@pytest.fixture
def fake_nli(monkeypatch):
    settings = NliSettings()
    monkeypatch.setattr(nli_matrix, "_settings", settings)
    monkeypatch.setattr(nli_matrix, "_cache_file", None)
    monkeypatch.setattr(_se_common, "_clusterers", {settings.hf_name: FirstLetterNli()})
    monkeypatch.setattr(_se_common, "_matrices", {})
    monkeypatch.setattr(_se_common, "_disk", {})
    monkeypatch.setattr(_se_common, "_nli_seconds", {})


@pytest.mark.usefixtures("fake_nli")
def test_cocoa_msp_plugin():
    record = cocoa_msp.score(ROWS)
    assert validate_scores(record, [0, 1]) == []
    # -log P: 1.0 and 2.0; inconsistency: 1 - 2/4 and 1 - 1/4
    assert record["uncertainty"] == pytest.approx([0.5, 1.5])
    assert record["nli"] == NliSettings().tag


@pytest.mark.usefixtures("fake_nli")
def test_cocoa_p_true_plugin_uses_the_p_true_record(monkeypatch):
    def fake_p_true(rows, model, tok):
        del model, tok
        return {
            "method": "p_true_zeroshot",
            "ids": [r["id"] for r in rows],
            "uncertainty": [0.2, 0.4],
            "failed": [False, False],
            "seconds_per_question": 0.1,
        }

    monkeypatch.setattr(cocoa_p_true.p_true_zeroshot, "score", fake_p_true)
    fake_model = cast("PreTrainedModel", object())  # never used: p(True) is replaced above
    record = cocoa_p_true.score(ROWS, model=fake_model, tok=cast("PreTrainedTokenizerBase", object()))
    assert validate_scores(record, [0, 1]) == []
    assert record["uncertainty"] == pytest.approx([0.1, 0.3])
    assert record["confidence"] == "p_true_zeroshot"


def test_registry_entries():
    assert REGISTRY["cocoa_msp"].family == "hybrid"
    assert not REGISTRY["cocoa_msp"].needs_model
    assert REGISTRY["cocoa_p_true"].needs_model
    assert REGISTRY["cocoa_p_true"].cost == REGISTRY["cocoa_msp"].cost + 1


def test_parse_run():
    assert parse_run("qwen1.5b/triviaqa_200") == ("qwen1.5b", "triviaqa", 200)
    assert parse_run("smollm2-1.7b/mmlu_200") == ("smollm2-1.7b", "mmlu", 200)
    with pytest.raises(ValueError, match="run must look like"):
        parse_run("qwen1.5b")
