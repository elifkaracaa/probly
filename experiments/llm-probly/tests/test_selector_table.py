"""Tests for selector.table (D1, step 1: loading one model's scores)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from selector.table import load_score_table

if TYPE_CHECKING:
    from pathlib import Path

ROWS = [{"id": i, "correct": i < 4} for i in range(10)]


def write(folder: Path, name: str, uncertainty: list[float], ids: list[int] | None = None, **extra: object) -> None:
    """Save a score record that follows CONTRACT.md."""
    ids = ids if ids is not None else [row["id"] for row in ROWS]
    record = {
        "method": name,
        "model": "m",
        "answer": "greedy",
        "ids": ids,
        "uncertainty": uncertainty,
        "failed": [False] * len(ids),
        "seconds_per_question": 0.5,
        **extra,
    }
    (folder / f"{name}.json").write_text(json.dumps(record))


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """Empty results folder ``<tmp>/m/triviaqa_10``."""
    folder = tmp_path / "m" / "triviaqa_10"
    folder.mkdir(parents=True)
    return folder


def test_loads_methods_as_columns_aligned_with_labels(run: Path) -> None:
    write(run, "msp", [float(i) for i in range(10)])
    write(run, "p_true_zeroshot", [0.1] * 10, probability=True)
    table = load_score_table("m", n=10, results_root=run.parents[1], rows=ROWS)
    assert table.methods == ["msp", "p_true_zeroshot"]
    assert list(table.scores.index) == list(range(10))
    assert table.scores.loc[7, "msp"] == 7.0
    assert table.correct.sum() == 4
    assert bool(table.info.loc["p_true_zeroshot", "probability"]) is True
    assert bool(table.info.loc["msp", "probability"]) is False


def test_dummy_and_invalid_files_are_skipped(run: Path) -> None:
    write(run, "msp", [0.0] * 10)
    write(run, "dummy_random", [0.5] * 10)
    write(run, "broken", [0.0] * 10, ids=list(range(9, -1, -1)))  # ids in the wrong order
    table = load_score_table("m", n=10, results_root=run.parents[1], rows=ROWS)
    assert table.methods == ["msp"]
    assert set(table.skipped) == {"dummy_random", "broken"}


def test_missing_labels_raise(run: Path) -> None:
    write(run, "msp", [0.0] * 10)
    with pytest.raises(ValueError, match="correct"):
        load_score_table("m", n=10, results_root=run.parents[1], rows=[{"id": i} for i in range(10)])


def test_missing_run_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_score_table("m", n=10, results_root=tmp_path, rows=ROWS)
