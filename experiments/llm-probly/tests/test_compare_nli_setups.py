"""Smoke tests for scripts/compare_nli_setups.py, run on saved matrices (``--reuse``), without an NLI model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from common.nli_matrix import NLI_MODELS, load_nli_cache, question_texts, save_nli_cache
import pytest
import torch

from scripts import compare_nli_setups

if TYPE_CHECKING:
    from pathlib import Path

ANSWERS = [
    ["Paris", "Paris", "Paris"],
    ["Rome", "Milan", "Naples"],
    ["Bern", "Bern", "Bern"],
    ["Oslo", "Lima", "Kiev"],
    ["Rio", "Rio", "Rio"],
    ["Cairo", "Doha", "Accra"],
    ["Tokyo", "Tokyo", "Tokyo"],
    ["Quito", "Sofia", "Hanoi"],
]


def fake_matrix(texts: list[str]) -> torch.Tensor:
    """NLI matrix in probly order (contradiction, neutral, entailment): same first letter = entailment."""
    n = len(texts)
    matrix = torch.zeros((n, n, 3))
    for a in range(n):
        for b in range(n):
            matrix[a, b, 2 if texts[a][0] == texts[b][0] else 0] = 1.0
    return matrix


def rows_for_test() -> list[dict]:
    """Eight questions: the ones with agreeing samples are right, the others wrong."""
    return [
        {
            "id": i,
            "question": f"Q{i}?",
            "greedy_answer": a[0],
            "samples": a,
            "samples_ll_mean": [-0.1, -0.5, -1.0],
            "correct": len(set(a)) == 1,
        }
        for i, a in enumerate(ANSWERS)
    ]


def save_setups(folder: Path, rows: list[dict], n: int | None = None) -> None:
    """Write the saved matrices of all four setups for model ``m`` (the files ``--reuse`` reads)."""
    rows = rows[:n]
    for nli, hf_name in NLI_MODELS.items():
        for use_question in (False, True):
            name = compare_nli_setups.setup_name(nli, use_question=use_question)
            save_nli_cache(
                folder / f"m_{name}.nli.npz",
                [fake_matrix(question_texts(r)) for r in rows],
                list(range(len(rows))),
                nli_model=hf_name,
                use_question=use_question,
                seconds_per_question=0.25,
            )


def test_setup_name():
    assert compare_nli_setups.setup_name("deberta-large", use_question=True) == "deberta-large+question"
    assert compare_nli_setups.setup_name("deberta-base", use_question=False) == "deberta-base"


def test_load_saved_rejects_a_file_with_too_few_questions(tmp_path):
    save_setups(tmp_path, rows_for_test(), n=3)
    path = tmp_path / "m_deberta-base.nli.npz"
    matrices, seconds = compare_nli_setups.load_saved(path, 2)
    assert len(matrices) == 2
    assert seconds == pytest.approx(0.25)
    with pytest.raises(ValueError, match="run without --reuse"):
        compare_nli_setups.load_saved(path, 8)


def test_main_with_reuse_writes_the_table(tmp_path, monkeypatch, capsys):
    rows = rows_for_test()
    out_dir = tmp_path / "results" / "nli_setups"
    out_dir.mkdir(parents=True)
    save_setups(out_dir, rows)
    labels = [  # question 0: indices 1 and 2 are both "Paris"; question 1: "Rome" vs "Milan"
        {"model": "m", "id": 0, "index_a": 1, "index_b": 2, "stratum": "both_same", "same_meaning": True},
        {"model": "m", "id": 1, "index_a": 1, "index_b": 2, "stratum": "both_different", "same_meaning": False},
    ]
    monkeypatch.setattr(compare_nli_setups, "ROOT", tmp_path)
    monkeypatch.setattr(compare_nli_setups, "OUT_DIR", out_dir)
    monkeypatch.setattr(compare_nli_setups, "MODELS", ("m",))
    monkeypatch.setattr(
        compare_nli_setups, "shares_per_model", lambda: {"m": {"both_same": 0.9, "both_different": 0.1}}
    )
    monkeypatch.setattr(compare_nli_setups, "load_jsonl", lambda _: rows)
    monkeypatch.setattr(compare_nli_setups, "load_labels", lambda: labels)

    compare_nli_setups.main(0, reuse=True)

    table = (tmp_path / "results" / "nli_setups.md").read_text(encoding="utf-8")
    assert table in capsys.readouterr().out
    lines = [line for line in table.splitlines() if line.startswith("| deberta-")]
    assert [line.split(" | ")[0] for line in lines] == [
        "| deberta-base",
        "| deberta-base+question",
        "| deberta-large",
        "| deberta-large+question",
    ]
    assert lines[0].startswith("| deberta-base | 1.000 | 1.000 |")  # both labels matched by both rules
    assert "reference" in lines[0]
    assert all(line.endswith("| 0.25 |") for line in lines)
    assert len(load_nli_cache(out_dir / "m_deberta-base.nli.npz")["matrices"]) == 8  # --reuse does not rewrite
