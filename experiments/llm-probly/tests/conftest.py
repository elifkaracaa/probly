"""Shared fixtures for the smoke tests of scripts that read results/ and data/."""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

from common import summary
from common.data import save_jsonl
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import ModuleType


@pytest.fixture
def rows() -> list[dict]:
    """Fifty labelled questions; the first twelve are answered correctly."""
    return [{"id": i, "correct": i < 12} for i in range(50)]


@pytest.fixture
def write_scores(rows: list[dict]) -> Callable[[Path, str, list[float]], None]:
    """Return a function that writes one score file in the benchmark format."""

    def write(folder: Path, method: str, uncertainty: list[float]) -> None:
        record = {
            "method": method,
            "model": "m",
            "answer": "greedy",
            "ids": [r["id"] for r in rows],
            "uncertainty": uncertainty,
            "failed": [False] * len(rows),
            "seconds_per_question": 0.5,
        }
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{method}.json").write_text(json.dumps(record))

    return write


@pytest.fixture
def run_results_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rows: list[dict]) -> Callable[..., None]:
    """Return a function that runs a script's main() on files under tmp_path only.

    The script's RESULTS_ROOT, find_models, compare_models and cache_path are pointed
    at tmp_path, so the real data/ and results/ folders are never read or written.
    """

    def run(script: ModuleType, results: Path, *args: str) -> None:
        data = tmp_path / "data"
        data.mkdir(exist_ok=True)
        save_jsonl(rows, data / "labels.jsonl")
        monkeypatch.setattr(script, "RESULTS_ROOT", results)
        monkeypatch.setattr(script, "find_models", functools.partial(summary.find_models, results_root=results))
        monkeypatch.setattr(script, "compare_models", functools.partial(summary.compare_models, results_root=results))
        monkeypatch.setattr(script, "cache_path", lambda *_args, suffix=".jsonl": data / f"labels{suffix}")
        monkeypatch.setattr("sys.argv", ["script", "--n", str(len(rows)), "--n-boot", "20", *args])
        script.main()

    return run
