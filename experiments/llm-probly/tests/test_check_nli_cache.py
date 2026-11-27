"""Smoke tests for scripts/check_nli_cache.py: cached clusters vs probly's clusterer, without an NLI model."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

from common.nli_matrix import NliSettings, nli_cache_path, question_texts, save_nli_cache
import numpy as np
import pytest
import torch

from scripts import check_nli_cache

if TYPE_CHECKING:
    from pathlib import Path

ANSWERS = [["Paris", "Paris", "Lyon"], ["Rome", "Milan", "Naples"], ["Bern", "Bern", "Bern"]]


def probly_matrix(texts: list[str]) -> torch.Tensor:
    """NLI matrix in probly order (contradiction, neutral, entailment): same first letter = entailment."""
    n = len(texts)
    matrix = torch.zeros((n, n, 3))
    for a in range(n):
        for b in range(n):
            matrix[a, b, 2 if texts[a][0] == texts[b][0] else 0] = 1.0
    return matrix


class FakeClusterer:
    """Clusters samples by first letter, the same rule probly_matrix encodes."""

    def _cluster_row(self, samples: np.ndarray) -> np.ndarray:
        """One cluster id per sample, numbered in order of first appearance."""
        first: dict[str, int] = {}
        return np.array([first.setdefault(str(s)[0], len(first)) for s in samples])


def write_data(root: Path, nli_model: str) -> None:
    """A 3-question data cache and its NLI cache for ``nli_model`` under ``root/data/qwen0.5b``."""
    rows = [{"id": i, "question": f"Q{i}?", "greedy_answer": a[0], "samples": a} for i, a in enumerate(ANSWERS)]
    folder = root / "data" / "qwen0.5b"
    folder.mkdir(parents=True)
    data = folder / "triviaqa_3.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    settings = NliSettings(nli_model)
    save_nli_cache(
        nli_cache_path(data, settings),
        [probly_matrix(question_texts(r)) for r in rows],
        list(range(len(rows))),
        nli_model=settings.hf_name,
        use_question=False,
        seconds_per_question=0.1,
    )


@pytest.mark.parametrize(
    ("nli_model", "argv"),
    [("deberta-base", []), ("deberta-large", ["--nli-model", "deberta-large"])],
)
def test_main_reports_full_agreement_on_a_consistent_cache(tmp_path, monkeypatch, capsys, nli_model, argv):
    write_data(tmp_path, nli_model)
    monkeypatch.setattr(check_nli_cache, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_nli_cache, "HFGreedySemanticClusterer", SimpleNamespace(from_model_name=lambda _: FakeClusterer())
    )
    monkeypatch.setattr(sys, "argv", ["check_nli_cache", "--n", "3", *argv])

    check_nli_cache.main()

    out = capsys.readouterr().out
    assert f"cache microsoft/{nli_model}-mnli" in out
    assert "ids match: True   texts per question match: True" in out
    assert "DIFF" not in out
    for rule in ("probly", "strict", "lenient"):
        assert f"  {rule:<8} 3/3" in out
