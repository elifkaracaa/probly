"""Tests for the paired report of scripts/c4_verbalized_prompt.py (no model)."""

from __future__ import annotations

import json

import pytest
from scripts.c4_verbalized_prompt import paired_report

IDS = list(range(40))
CORRECT = [i % 2 == 0 for i in IDS]
PERFECT = [0.0 if c else 1.0 for c in CORRECT]  # AUROC 1
RANDOMISH = [((i * 7) % 11) / 10 for i in IDS]


def write(path, uncertainty):
    record = {"method": path.stem, "model": "m", "answer": "greedy", "ids": IDS, "uncertainty": uncertainty}
    record.update({"failed": [False] * len(IDS), "seconds_per_question": 0.1})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def test_reference_is_the_saved_week1_file(tmp_path):
    write(tmp_path / "verbalized_2s.json", RANDOMISH)
    write(tmp_path / "c4" / "verbalized_2s_anchored.json", PERFECT)
    lines = paired_report(tmp_path, IDS, CORRECT)
    assert "week 1 = verbalized_2s.json" in lines[0]
    assert len(lines) == 2  # only anchored was saved
    assert lines[1].strip().startswith("anchored - week 1: +")


def test_plain100_rerun_is_preferred_and_both_prompts_are_compared(tmp_path):
    write(tmp_path / "verbalized_2s.json", PERFECT)  # would give a negative difference
    write(tmp_path / "c4" / "verbalized_2s_plain100.json", RANDOMISH)
    write(tmp_path / "c4" / "verbalized_2s_plain10.json", RANDOMISH)
    write(tmp_path / "c4" / "verbalized_2s_anchored.json", PERFECT)
    lines = paired_report(tmp_path, IDS, CORRECT)
    assert "verbalized_2s_plain100.json" in lines[0]
    assert [line.split(":")[0].strip() for line in lines[1:]] == [
        "plain10 - week 1",
        "anchored - week 1",
        "anchored - plain10",
    ]
    assert lines[1].strip().startswith("plain10 - week 1: +0.000")


def test_other_question_ids_are_rejected(tmp_path):
    write(tmp_path / "verbalized_2s.json", RANDOMISH)
    write(tmp_path / "c4" / "verbalized_2s_anchored.json", PERFECT)
    with pytest.raises(ValueError, match=r"id|ids"):
        paired_report(tmp_path, IDS[::-1], CORRECT)
