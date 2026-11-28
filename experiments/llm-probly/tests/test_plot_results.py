"""Smoke tests for the plotting script (scripts/plot_results.py)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts import plot_results as script

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest


def perfect_scores(rows: list[dict]) -> list[float]:
    return [0.0 if r["correct"] else 1.0 for r in rows]


def test_main_writes_size_and_rejection_plots(
    tmp_path: Path,
    rows: list[dict],
    write_scores: Callable[..., None],
    run_results_script: Callable[..., None],
) -> None:
    results = tmp_path / "results"
    for model in ("qwen0.5b", "qwen1.5b"):
        write_scores(results / model / "triviaqa_50", "perfect", perfect_scores(rows))

    run_results_script(script, results, "--curves-model", "qwen0.5b")

    plots = sorted(p.name for p in (results / "plots").glob("*.png"))
    assert plots == [
        "triviaqa_50_auroc_vs_size.png",
        "triviaqa_50_prr_vs_size.png",
        "triviaqa_50_qwen0.5b_rejection.png",
    ]


def test_main_skips_rejection_curves_for_a_model_without_results(
    tmp_path: Path,
    rows: list[dict],
    write_scores: Callable[..., None],
    run_results_script: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = tmp_path / "results"
    write_scores(results / "qwen0.5b" / "triviaqa_50", "perfect", perfect_scores(rows))

    run_results_script(script, results, "--curves-model", "qwen7b")

    assert not list((results / "plots").glob("*_rejection.png"))
    assert "rejection curves skipped" in capsys.readouterr().out


def test_load_method_scores_skips_hidden_and_non_score_files(
    tmp_path: Path, rows: list[dict], write_scores: Callable[..., None]
) -> None:
    write_scores(tmp_path, "perfect", perfect_scores(rows))
    write_scores(tmp_path, "dummy_random", perfect_scores(rows))  # test plugin: hidden in plots
    (tmp_path / "meta.json").write_text(json.dumps({"note": "not a score file"}))

    assert list(script.load_method_scores(tmp_path)) == ["perfect"]
