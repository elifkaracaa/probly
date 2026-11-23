"""Tests for decomposed semantic entropy (B4)."""

from __future__ import annotations

from argparse import Namespace
import math
from typing import Any, cast

from common.decomposed_se import clean_answer, clean_clarification, decompose, entropy
from common.io import validate_scores
from common.registry import REGISTRY
from methods import decomposed_semantic_entropy as plugin
import numpy as np
import pytest
from scripts.run_benchmark import choose_methods
import torch

CONTRADICTION, NEUTRAL, ENTAILMENT = 0, 1, 2  # probly's label order


def test_same_answers_everywhere_give_zero():
    assert decompose([0] * 15, 3, 5) == {"total": 0.0, "aleatoric": 0.0, "epistemic": 0.0}


def test_each_reading_certain_but_different_is_purely_aleatoric():
    parts = decompose([0] * 5 + [1] * 5 + [2] * 5, 3, 5)  # the question is ambiguous, the model is sure per reading

    assert parts["total"] == pytest.approx(math.log(3))
    assert parts["aleatoric"] == pytest.approx(math.log(3))
    assert parts["epistemic"] == pytest.approx(0.0)


def test_same_split_in_every_reading_is_purely_epistemic():
    parts = decompose([0, 0, 1, 1] * 3, 3, 4)  # every reading gets the same 50/50 split: the model does not know

    assert parts["total"] == pytest.approx(math.log(2))
    assert parts["aleatoric"] == pytest.approx(0.0)
    assert parts["epistemic"] == pytest.approx(math.log(2))


def test_total_is_aleatoric_plus_epistemic():
    rng = np.random.default_rng(0)
    for _ in range(50):
        parts = decompose(rng.integers(0, 4, size=15).tolist(), 3, 5)
        assert parts["total"] == pytest.approx(parts["aleatoric"] + parts["epistemic"])
        assert parts["aleatoric"] >= 0.0


def test_probly_decomposition_matches_the_formulas_with_hou_et_al_naming():
    ids = [0, 0, 1, 2, 2, 0, 1, 1, 1, 2, 0, 0, 0, 0, 3]
    p = np.stack([np.bincount(row, minlength=4) / 5 for row in np.reshape(ids, (3, 5))])
    expected_total = entropy(p.mean(axis=0))
    expected_epistemic = np.mean([entropy(row) for row in p])  # mean entropy per reading

    parts = decompose(ids, 3, 5)

    assert parts["total"] == pytest.approx(expected_total)
    assert parts["epistemic"] == pytest.approx(expected_epistemic)
    assert parts["aleatoric"] == pytest.approx(expected_total - expected_epistemic)


def test_decompose_checks_the_number_of_answers():
    with pytest.raises(ValueError, match="expected 3 x 5 cluster ids, got 14"):
        decompose([0] * 14, 3, 5)


def test_cleaning_keeps_the_first_line_and_falls_back():
    assert clean_clarification('"Which city is the capital of France?"\nMore text', "q") == (
        "Which city is the capital of France?"
    )
    assert clean_clarification("   ", "original question") == "original question"
    assert clean_answer("Paris\nParis is the capital") == "Paris"
    assert clean_answer("") == "(no answer)"


def nli_from_first_letters(texts: list[str]) -> torch.Tensor:
    """Fake NLI: texts with the same first letter entail each other, others contradict."""
    n = len(texts)
    matrix = torch.zeros((n, n, 3))
    for a in range(n):
        for b in range(n):
            matrix[a, b, ENTAILMENT if texts[a][0] == texts[b][0] else CONTRADICTION] = 1.0
    matrix[torch.arange(n), torch.arange(n)] = float("nan")
    return matrix


def test_plugin_record_with_fake_generation(monkeypatch):
    generated = [
        {"clarifications": ["c1", "c2", "c3"], "answers": ["Paris"] * 5 + ["Lyon"] * 5 + ["Nice"] * 5},
        {"clarifications": ["c1", "c2", "c3"], "answers": ["Paris", "Lyon"] * 7 + ["Paris"]},
    ]
    monkeypatch.setattr(plugin, "clarify_and_answer", lambda _model, _tok, questions: generated[: len(questions)])
    monkeypatch.setattr("methods._se_common.prepare_matrices", lambda _rows: None)
    monkeypatch.setattr(
        "methods._se_common.get_matrix", lambda row: nli_from_first_letters([row["greedy_answer"], *row["samples"]])
    )
    rows = [{"id": i, "model": "m", "question": f"q{i}", "greedy_answer": "Paris"} for i in range(2)]

    record = plugin.score(rows, model=cast("Any", object()), tok=cast("Any", object()))

    assert validate_scores(record, expected_ids=[0, 1]) == []
    assert record["uncertainty"][0] == pytest.approx(math.log(3))
    assert record["aleatoric"][0] == pytest.approx(math.log(3))  # each reading gets one answer: ambiguous question
    assert record["epistemic"][1] > record["aleatoric"][1]  # mixed answers in every reading: the model does not know
    assert record["details"][0]["clusters"] == [0] * 5 + [1] * 5 + [2] * 5


def test_plugin_needs_the_model():
    with pytest.raises(ValueError, match="needs the model and tokenizer"):
        plugin.score([{"id": 0, "question": "q", "greedy_answer": "a"}])


def test_runner_runs_it_only_when_named():
    args = Namespace(methods=None, skip_model_methods=False)

    everything = choose_methods(dict(REGISTRY), {"methods": "all"}, args)
    named = choose_methods(dict(REGISTRY), {"methods": "all"}, Namespace(methods=plugin.NAME, skip_model_methods=False))

    assert plugin.NAME not in everything
    assert "semantic_entropy" in everything
    assert set(named) == {plugin.NAME}
