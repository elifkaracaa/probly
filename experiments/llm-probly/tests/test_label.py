"""Smoke tests for the labelling script (scripts/label.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from common.data import load_jsonl, save_jsonl
import pytest

from scripts import label

if TYPE_CHECKING:
    from pathlib import Path

QUESTIONS = [
    {"id": 0, "question": "Capital of France?", "aliases": ["Paris"]},
    {"id": 1, "question": "Capital of Italy?", "aliases": ["Rome", "Roma"]},
]


def make_rows() -> list[dict]:
    return [
        {"id": 0, "question": "Capital of France?", "greedy_answer": "Paris", "samples": ["Paris", "Lyon"]},
        {"id": 1, "question": "Capital of Italy?", "greedy_answer": "Milan", "samples": ["Rome", "Roma"]},
    ]


def run_label(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    # No dataset download and no write to data/: both are replaced for this test only.
    monkeypatch.setattr(label, "cache_path", lambda *_args, **_kwargs: path)
    monkeypatch.setattr(label, "load_questions", lambda *_args, **_kwargs: QUESTIONS)
    monkeypatch.setattr("sys.argv", ["label", "--n", "2"])
    label.main()


def test_main_labels_rows_and_rewrites_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "triviaqa_2.jsonl"
    save_jsonl(make_rows(), path)

    run_label(monkeypatch, path)

    rows = load_jsonl(path)
    assert [r["correct"] for r in rows] == [True, False]
    assert [r["samples_correct"] for r in rows] == [[True, False], [True, True]]
    assert rows[1]["aliases"] == ["Rome", "Roma"]
    assert "greedy accuracy 1/2" in capsys.readouterr().out


def test_main_rejects_rows_from_another_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = make_rows()
    rows[0]["question"] = "Something else?"
    path = tmp_path / "triviaqa_2.jsonl"
    save_jsonl(rows, path)

    with pytest.raises(ValueError, match="does not match"):
        run_label(monkeypatch, path)
