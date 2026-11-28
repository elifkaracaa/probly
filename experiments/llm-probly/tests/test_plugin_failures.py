"""Failure path of the plugins that read the shared NLI matrix.

If one question cannot be scored, the plugin does not stop: that question gets the placeholder
uncertainty 1.0 with failed=True (CONTRACT.md, rule 3), and the other questions are scored normally.
"""

from __future__ import annotations

from methods import _graph_common, _se_common, deg_mat, semantic_entropy
import pytest
import torch

ENTAILMENT = 2  # probly's label order: contradiction, neutral, entailment


def make_rows(n: int) -> list[dict]:
    """Questions with a greedy answer and three samples each (they differ, so every row has its own key)."""
    return [
        {
            "id": i,
            "model": "m",
            "dataset": "nowhere",
            "greedy_answer": "a",
            "samples": ["a", "a", f"b{i}"],
            "samples_ll_mean": [-0.5, -0.5, -0.5],
        }
        for i in range(n)
    ]


def break_second_question(monkeypatch):
    """Matrices without the NLI model: all answers agree, except that question 1 cannot be scored."""

    def get_matrix(row):
        if row["id"] == 1:
            msg = "no NLI matrix for this question"
            raise RuntimeError(msg)
        matrix = torch.zeros((4, 4, 3))
        matrix[..., ENTAILMENT] = 1.0
        return matrix

    monkeypatch.setattr(_se_common, "prepare_matrices", lambda _rows: None)
    monkeypatch.setattr(_se_common, "get_matrix", get_matrix)
    monkeypatch.setattr(_graph_common, "_scores", {})


def test_graph_plugin_marks_only_the_broken_question_as_failed(monkeypatch):
    break_second_question(monkeypatch)

    record = deg_mat.score(make_rows(3))

    assert record["failed"] == [False, True, False]
    assert record["uncertainty"] == pytest.approx([0.0, 1.0, 0.0])


def test_semantic_entropy_marks_only_the_broken_question_as_failed(monkeypatch):
    break_second_question(monkeypatch)
    rows = make_rows(3)
    monkeypatch.setattr(_se_common, "_nli_seconds", {_se_common.matrix_key(row): 0.0 for row in rows})

    record = semantic_entropy.score(rows)

    assert record["failed"] == [False, True, False]
    assert record["uncertainty"] == pytest.approx([0.0, 1.0, 0.0])
