from __future__ import annotations

from typing import TYPE_CHECKING

from common.data import save_jsonl
import pytest
from scripts.check_prefix import compare_prefix, format_report

from scripts import check_prefix

if TYPE_CHECKING:
    from pathlib import Path


def row(i: int, answer: str = "a", correct: bool = True) -> dict:
    return {
        "id": i,
        "question": f"q{i}",
        "aliases": [f"x{i}"],
        "greedy_answer": answer,
        "samples": [answer] * 10,
        "correct": correct,
    }


def test_identical_prefix():
    small = [row(i) for i in range(3)]
    large = [row(i) for i in range(5)]
    report = compare_prefix(small, large)
    assert report.same_questions
    assert report.n_compared == 3
    assert (report.same_greedy, report.same_samples, report.same_correct) == (3, 3, 3)


def test_different_question_is_found():
    small = [row(i) for i in range(3)]
    large = [row(0), row(7), row(2), row(3)]
    report = compare_prefix(small, large)
    assert not report.same_questions
    assert report.first_mismatch == 1
    assert "DIFFERENT" in format_report(report, "s", "l")


def test_changed_answers_are_counted_but_questions_still_match():
    small = [row(0), row(1, answer="b", correct=False)]
    large = [row(0), row(1, answer="c", correct=True), row(2)]
    report = compare_prefix(small, large)
    assert report.same_questions
    assert report.same_greedy == 1
    assert report.same_correct == 1


def test_partial_large_cache_compares_what_exists():
    small = [row(i) for i in range(5)]
    report = compare_prefix(small, [row(0), row(1)])
    assert report.n_compared == 2
    assert report.same_questions


def test_unlabelled_rows_are_not_compared_for_correctness():
    small = [row(0)]
    unlabelled = row(0)
    del unlabelled["correct"]
    report = compare_prefix(small, [unlabelled])
    assert report.n_labelled == 0
    assert "correctness" not in format_report(report, "s", "l")


def run_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> int | str | None:
    # Cache files live in tmp_path as <n>.jsonl / <n>.partial.jsonl, never in the real data/.
    monkeypatch.setattr(
        check_prefix, "cache_path", lambda _model, n, _dataset, suffix=".jsonl": tmp_path / f"{n}{suffix}"
    )
    monkeypatch.setattr("sys.argv", ["check_prefix", "--model", "m", "--small", "2", "--large", "3"])
    with pytest.raises(SystemExit) as stop:
        check_prefix.main()
    return stop.value.code


def test_main_reads_the_partial_large_cache_and_exits_0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    save_jsonl([row(i) for i in range(2)], tmp_path / "2.jsonl")
    save_jsonl([row(i) for i in range(3)], tmp_path / "3.partial.jsonl")  # generation still running

    assert run_main(monkeypatch, tmp_path) == 0
    out = capsys.readouterr().out
    assert "3.partial.jsonl: first 2 rows vs 2.jsonl" in out
    assert "questions: identical" in out


def test_main_exits_1_when_the_questions_differ(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    save_jsonl([row(0), row(1)], tmp_path / "2.jsonl")
    save_jsonl([row(0), row(7), row(2)], tmp_path / "3.jsonl")

    assert run_main(monkeypatch, tmp_path) == 1


def test_main_exits_2_without_a_cache_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_main(monkeypatch, tmp_path) == 2
    assert "no cache file" in capsys.readouterr().out
