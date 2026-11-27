"""Smoke tests for scripts/embedding_vs_nli.py with fake NLI and embedding backends (no model downloads)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from common.nli_matrix import question_texts
import torch

from scripts import embedding_vs_nli

ANSWERS = [
    ["Paris", "Paris", "Paris"],
    ["Rome", "Milan", "Naples"],
    ["Bern", "Bern", "Bern"],
    ["Oslo", "Lima", "Kiev"],
    ["Rio", "Rio", "Rio"],
    ["Cairo", "Doha", "Accra"],
]


def fake_matrix(texts: list[str]) -> torch.Tensor:
    """NLI matrix in probly order (contradiction, neutral, entailment): same first letter = entailment."""
    n = len(texts)
    matrix = torch.zeros((n, n, 3))
    for a in range(n):
        for b in range(n):
            matrix[a, b, 2 if texts[a][0] == texts[b][0] else 0] = 1.0
    return matrix


class FakeEmbeddingClusterer:
    """Stands in for the embedding backend; gives the same matrices as the fake NLI model."""

    def __init__(self) -> None:
        """Start with the library's default threshold."""
        self.threshold = 0.9

    def pairwise_nli_many(
        self, groups: list[list[str]], contexts: list[str | None] | None = None, deduplicate: bool = True
    ) -> list[torch.Tensor]:
        """One matrix per question."""
        del contexts, deduplicate
        return [fake_matrix(group) for group in groups]


def rows_for_test() -> list[dict]:
    """Six questions: the ones with agreeing samples are right, the others wrong."""
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


def test_same_partition_ignores_cluster_ids():
    assert embedding_vs_nli.same_partition([0, 0, 1], [1, 1, 0])
    assert not embedding_vs_nli.same_partition([0, 0, 1], [0, 1, 1])
    assert not embedding_vs_nli.same_partition([0, 0, 0], [0, 1, 2])


def test_main_writes_both_reports_and_picks_the_threshold_on_the_first_model(tmp_path, monkeypatch, capsys):
    rows = rows_for_test()
    monkeypatch.setattr(embedding_vs_nli, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(embedding_vs_nli, "load_jsonl", lambda _: rows)
    monkeypatch.setattr(
        embedding_vs_nli,
        "HFEmbeddingSemanticClusterer",
        SimpleNamespace(from_model_name=FakeEmbeddingClusterer),
    )
    monkeypatch.setattr(
        embedding_vs_nli, "benchmark_nli_matrices", lambda rs, _: [fake_matrix(question_texts(r)) for r in rs]
    )
    monkeypatch.setattr(embedding_vs_nli, "nli_seconds", lambda _: 0.5)
    argv = ["embedding_vs_nli", "--models", "m1", "m2", "--thresholds", "0.8", "0.9", "--timing-questions", "2"]
    monkeypatch.setattr(sys, "argv", argv)

    embedding_vs_nli.main()

    for model in ("m1", "m2"):
        folder = tmp_path / model / "triviaqa_200" / "analysis"
        record = json.loads((folder / "embedding_backend.json").read_text())
        assert record["selected_on"] == "m1"
        assert record["threshold"] == 0.8  # equal AUROC at both thresholds: the first one wins
        assert record["seconds_nli"] == 0.5
        assert {r["same_as_nli"] for r in record["embedding"].values()} == {1.0}
        assert record["nli"]["auroc"]["semantic_entropy"] == 1.0  # agreeing samples are exactly the right answers
        markdown = (folder / "embedding_backend.md").read_text()
        assert ("(held-out model)" in markdown) == (model == "m2")
    assert "wrote" in capsys.readouterr().out
