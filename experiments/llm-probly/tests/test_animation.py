"""Tests for the animated README demo (demo/animation.py), with fake rows and score files."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from demo import animation
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

ROWS = [
    {
        "id": 0,
        "question": "In which year was Sputnik launched?",
        "greedy_answer": "1957.",
        "correct": True,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    {
        "id": 1,
        "question": "Who wrote The Joy Luck Club?",
        "greedy_answer": "Pu Songling",
        "correct": False,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    {
        "id": 2,
        "question": "What is the capital of France?",
        "greedy_answer": "Paris",
        "correct": True,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
]


def _score_file(folder: Path, method: str, values: list[float]) -> None:
    record = {
        "method": method,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "answer": "greedy",
        "ids": [0, 1, 2],
        "uncertainty": values,
        "failed": [False, False, False],
        "seconds_per_question": 0.0,
    }
    (folder / f"{method}.json").write_text(json.dumps(record))


@pytest.fixture
def fake_results(tmp_path, monkeypatch):
    folder = tmp_path / "results" / "qwen1.5b" / "triviaqa_3"
    folder.mkdir(parents=True)
    _score_file(folder, "p_true_zeroshot", [0.1, 0.9, 0.2])
    _score_file(folder, "likelihood_mean", [0.2, 0.8, 0.1])
    monkeypatch.setattr(animation, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(animation, "load_data", lambda _run, _n, _dataset="triviaqa": ROWS)
    monkeypatch.setattr(animation, "cache_path", lambda *_args, **_kwargs: tmp_path / "missing.meta.json")
    return folder


def test_percentile_counts_ties_half():
    values = np.array([0.1, 0.2, 0.2, 0.4])
    assert animation.percentile(values, 0.2) == pytest.approx((1 + 0.5 * 2) / 4)
    assert animation.percentile(values, 0.1) == pytest.approx(0.5 / 4)


def test_clean_collapses_whitespace_and_cuts():
    assert animation.clean("a   b\nc", 10) == "a b c"
    assert animation.clean("abcdefghij", 5) == "abcd…"


def test_results_dir_falls_back_to_the_flat_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(animation, "RESULTS_ROOT", tmp_path)
    assert animation.results_dir("qwen0.5b", 50, "triviaqa") == tmp_path / "qwen0.5b"
    (tmp_path / "qwen0.5b" / "triviaqa_50").mkdir(parents=True)
    assert animation.results_dir("qwen0.5b", 50, "triviaqa") == tmp_path / "qwen0.5b" / "triviaqa_50"


@pytest.mark.usefixtures("fake_results")
def test_collect_scenes_reads_rows_and_scores():
    scenes, footer = animation.collect_scenes("qwen1.5b", 3, [0, 1])
    assert [s["id"] for s in scenes] == [0, 1]
    assert scenes[1]["correct"] is False
    labels = [label for _, label, _, _ in scenes[0]["scores"]]
    assert labels == ["p(True) zero-shot", "likelihood (mean)"]
    assert footer.startswith("Qwen2.5-1.5B-Instruct")


@pytest.mark.usefixtures("fake_results")
def test_collect_scenes_rejects_unknown_ids():
    with pytest.raises(SystemExit, match="ids not in the data"):
        animation.collect_scenes("qwen1.5b", 3, [7])


@pytest.mark.usefixtures("fake_results")
def test_write_animation_writes_both_themes(tmp_path):
    out = tmp_path / "out"
    paths = animation.write_animation("qwen1.5b", 3, [0, 1], out, "_test")
    assert sorted(p.name for p in paths) == [
        "llm_uncertainty_demo_test_dark.svg",
        "llm_uncertainty_demo_test_light.svg",
    ]
    svg = (out / "llm_uncertainty_demo_test_dark.svg").read_text()
    assert svg.startswith("<svg")
    assert "Qwen2.5-1.5B-Instruct" in svg  # footer
    assert 'repeatCount="indefinite"' in svg
