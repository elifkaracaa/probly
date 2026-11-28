"""Smoke test for the method comparison script (scripts/diff_methods.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from common.data import save_jsonl

from scripts import diff_methods as script

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest


def test_main_names_the_better_method(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rows: list[dict],
    write_scores: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = tmp_path / "results"
    perfect = [0.0 if r["correct"] else 1.0 for r in rows]
    write_scores(results / "m" / "triviaqa_50", "perfect", perfect)
    write_scores(results / "m" / "triviaqa_50", "reversed", [1.0 - u for u in perfect])
    save_jsonl(rows, tmp_path / "labels.jsonl")
    monkeypatch.setattr(script, "RESULTS_ROOT", results)
    monkeypatch.setattr(script, "cache_path", lambda *_args, **_kwargs: tmp_path / "labels.jsonl")
    monkeypatch.setattr("sys.argv", ["diff_methods", "perfect", "reversed", "--run", "m/triviaqa_50", "--n-boot", "50"])

    script.main()

    out = capsys.readouterr().out
    assert out.startswith("On m/triviaqa_50, perfect is better than reversed: AUROC 1.000 vs 0.000")
