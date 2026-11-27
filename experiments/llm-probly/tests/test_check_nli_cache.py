"""Tests for scripts/check_nli_cache.py: equivalence rules on cached matrices and a smoke run of main()."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from common.nli_matrix import question_texts, save_nli_cache
import numpy as np
import pytest
from scripts.check_nli_cache import CONTRADICT, ENTAIL, NEUTRAL, as_partition, clusters_from_cache, same_meaning
import torch

from scripts import check_nli_cache


def file_pair(ab: int, ba: int) -> np.ndarray:
    """A 2-text matrix in the cache file's label order with one-hot labels for a->b and b->a."""
    mat = np.zeros((2, 2, 3))
    mat[0, 1, ab] = 1.0
    mat[1, 0, ba] = 1.0
    return mat


@pytest.mark.parametrize(
    ("ab", "ba", "expected"),
    [
        (ENTAIL, ENTAIL, {"probly": True, "strict": True, "lenient": True}),
        (ENTAIL, NEUTRAL, {"probly": True, "strict": False, "lenient": True}),
        (NEUTRAL, NEUTRAL, {"probly": False, "strict": False, "lenient": True}),
        (ENTAIL, CONTRADICT, {"probly": False, "strict": False, "lenient": False}),
    ],
)
def test_same_meaning_rules(ab, ba, expected):
    mat = file_pair(ab, ba)
    assert {rule: same_meaning(mat, 0, 1, rule) for rule in expected} == expected


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="other"):
        same_meaning(file_pair(ENTAIL, ENTAIL), 0, 1, "other")


def test_clusters_from_cache_is_greedy_and_partition_ignores_order():
    texts = ["Paris", "Paris", "Lyon", "Paris"]  # index 0 = greedy answer
    mat = np.zeros((4, 4, 3))
    for a in range(4):
        for b in range(4):
            mat[a, b, ENTAIL if texts[a] == texts[b] else CONTRADICT] = 1.0
    clusters = clusters_from_cache(mat, range(1, 4), "probly")
    assert clusters == [[1, 3], [2]]
    assert as_partition(clusters) == as_partition([[2], [3, 1]])


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


def test_main_reports_full_agreement_on_a_consistent_cache(tmp_path, monkeypatch, capsys):
    answers = [["Paris", "Paris", "Lyon"], ["Rome", "Milan", "Naples"], ["Bern", "Bern", "Bern"]]
    rows = [
        {"id": i, "question": f"Q{i}?", "greedy_answer": a[0], "samples": a, "correct": True}
        for i, a in enumerate(answers)
    ]
    folder = tmp_path / "data" / "qwen0.5b"
    folder.mkdir(parents=True)
    (folder / "triviaqa_3.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    save_nli_cache(
        folder / "triviaqa_3.nli.npz",
        [probly_matrix(question_texts(r)) for r in rows],
        list(range(len(rows))),
        nli_model="microsoft/deberta-base-mnli",
        use_question=False,
        seconds_per_question=0.1,
    )
    monkeypatch.setattr(check_nli_cache, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_nli_cache, "HFGreedySemanticClusterer", SimpleNamespace(from_model_name=lambda _: FakeClusterer())
    )
    monkeypatch.setattr(sys, "argv", ["check_nli_cache", "--n", "3"])

    check_nli_cache.main()

    out = capsys.readouterr().out
    assert "ids match: True   texts per question match: True" in out
    assert "DIFF" not in out
    for rule in ("probly", "strict", "lenient"):
        assert f"  {rule:<8} 3/3" in out
