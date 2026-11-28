"""Smoke tests for the model comparison script (scripts/compare_models.py)."""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

from common import summary
from common.data import save_jsonl
import pytest

from scripts import compare_models as script

if TYPE_CHECKING:
    from pathlib import Path

ROWS = [{"id": i, "correct": i < 12} for i in range(50)]


def write_scores(folder: Path, method: str, uncertainty: list[float]) -> None:
    record = {
        "method": method,
        "model": "m",
        "answer": "greedy",
        "ids": [r["id"] for r in ROWS],
        "uncertainty": uncertainty,
        "failed": [False] * len(ROWS),
        "seconds_per_question": 0.5,
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{method}.json").write_text(json.dumps(record))


def run_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, results: Path) -> None:
    # Everything is read from and written to tmp_path, never to the real data/ and results/.
    data = tmp_path / "data"
    data.mkdir()
    save_jsonl(ROWS, data / "labels.jsonl")
    monkeypatch.setattr(script, "RESULTS_ROOT", results)
    monkeypatch.setattr(script, "find_models", functools.partial(summary.find_models, results_root=results))
    monkeypatch.setattr(script, "compare_models", functools.partial(summary.compare_models, results_root=results))
    monkeypatch.setattr(script, "cache_path", lambda *_args, suffix=".jsonl": data / f"labels{suffix}")
    monkeypatch.setattr("sys.argv", ["compare_models", "--n", "50", "--n-boot", "20"])
    script.main()


def test_main_writes_one_comparison_for_all_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    perfect = [0.0 if r["correct"] else 1.0 for r in ROWS]
    for model in ("qwen1.5b", "qwen0.5b"):
        write_scores(results / model / "triviaqa_50", "perfect", perfect)

    run_script(monkeypatch, tmp_path, results)

    report = (results / "comparison_triviaqa_50.md").read_text()
    assert report.startswith("# triviaqa, 50 questions")
    assert report.index("qwen0.5b") < report.index("qwen1.5b")  # small to large
    assert "unknown" in report  # no meta file -> precision is unknown
    assert "saved" in capsys.readouterr().out


def test_main_stops_when_there_are_no_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="run run_benchmark first"):
        run_script(monkeypatch, tmp_path, tmp_path / "results")
