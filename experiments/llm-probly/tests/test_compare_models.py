"""Smoke tests for the model comparison script (scripts/compare_models.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts import compare_models as script

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_main_writes_one_comparison_for_all_models(
    tmp_path: Path,
    rows: list[dict],
    write_scores: Callable[..., None],
    run_results_script: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = tmp_path / "results"
    perfect = [0.0 if r["correct"] else 1.0 for r in rows]
    for model in ("qwen1.5b", "qwen0.5b"):
        write_scores(results / model / "triviaqa_50", "perfect", perfect)

    run_results_script(script, results)

    report = (results / "comparison_triviaqa_50.md").read_text()
    assert report.startswith("# triviaqa, 50 questions")
    assert report.index("qwen0.5b") < report.index("qwen1.5b")  # small to large
    assert "unknown" in report  # no meta file -> precision is unknown
    assert "saved" in capsys.readouterr().out


def test_main_stops_when_there_are_no_results(tmp_path: Path, run_results_script: Callable[..., None]) -> None:
    with pytest.raises(SystemExit, match="run run_benchmark first"):
        run_results_script(script, tmp_path / "results")
