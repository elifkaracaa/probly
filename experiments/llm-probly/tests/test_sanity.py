"""Tests for the shared sanity test of reflexive methods (common/sanity.py)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from common.sanity import SANITY, run_sanity, save_sanity
import pytest

if TYPE_CHECKING:
    from pathlib import Path

RIGHT_ANSWERS = {right for _, right, _ in SANITY}


def knows_the_answers(_question: str, answer: str, _samples: list[str] | None) -> float:
    """A perfect method: confident in every correct answer, doubtful about every wrong one."""
    return 0.9 if answer in RIGHT_ANSWERS else 0.1


def test_sanity_set_has_a_right_and_a_wrong_answer_per_question():
    assert len(SANITY) >= 5
    for question, right, wrong in SANITY:
        assert question
        assert right != wrong


def test_perfect_method_gets_auroc_one_and_a_clear_gap():
    result = run_sanity(knows_the_answers)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["gap"] == pytest.approx(0.8)
    assert result["n_distinct"] == 2
    assert len(result["confidences"]) == 2 * len(SANITY)


def test_each_question_is_asked_twice_with_the_right_answer_first():
    calls = []

    def record(question: str, answer: str, samples: list[str] | None) -> float:
        calls.append((question, answer, samples))
        return 0.5

    run_sanity(record)
    expected = []
    for question, right, wrong in SANITY:
        expected += [(question, right, None), (question, wrong, None)]
    assert calls == expected


def test_constant_method_gets_auroc_one_half_and_no_gap():
    result = run_sanity(lambda *_args: 0.7)
    assert result["auroc"] == pytest.approx(0.5)
    assert result["gap"] == pytest.approx(0.0)
    assert result["n_distinct"] == 1


def test_inverted_method_gets_auroc_zero_and_a_negative_gap():
    result = run_sanity(lambda _q, answer, _s: 0.1 if answer in RIGHT_ANSWERS else 0.9)
    assert result["auroc"] == pytest.approx(0.0)
    assert result["gap"] == pytest.approx(-0.8)


def test_samples_are_passed_per_question_and_missing_ones_are_none():
    first_question = SANITY[0][0]
    seen = {}

    def record(question: str, _answer: str, samples: list[str] | None) -> float:
        seen[question] = samples
        return 0.5

    run_sanity(record, samples={first_question: ["Paris", "Lyon"]})
    assert seen[first_question] == ["Paris", "Lyon"]
    assert all(value is None for question, value in seen.items() if question != first_question)


def test_save_sanity_writes_a_json_named_after_method_and_short_model_name(tmp_path: Path):
    result = run_sanity(knows_the_answers)
    folder = tmp_path / "results" / "sanity"  # the folder does not exist yet
    path = save_sanity(result, "p_true_zeroshot", "org/Qwen2.5-0.5B-Instruct", folder)
    assert path == folder / "p_true_zeroshot_Qwen2.5-0.5B-Instruct.json"
    saved = json.loads(path.read_text())
    assert saved["method"] == "p_true_zeroshot"
    assert saved["model"] == "org/Qwen2.5-0.5B-Instruct"
    assert saved["auroc"] == pytest.approx(1.0)
    assert len(saved["confidences"]) == 2 * len(SANITY)
