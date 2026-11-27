"""Smoke tests for scripts/build_nli_cache.py: the file it writes is a valid, reusable NLI cache."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

from common.nli_matrix import (
    NliSettings,
    load_nli_cache,
    matrix_fingerprint,
    question_texts,
    read_cached_matrices,
)
import pytest
import torch

from scripts import build_nli_cache

if TYPE_CHECKING:
    from pathlib import Path

NLI_MODEL = "microsoft/deberta-large-mnli"


def fake_matrix(texts: list[str]) -> torch.Tensor:
    """NLI matrix in probly order (contradiction, neutral, entailment): same first letter = entailment."""
    n = len(texts)
    matrix = torch.zeros((n, n, 3))
    for a in range(n):
        for b in range(n):
            matrix[a, b, 2 if texts[a][0] == texts[b][0] else 0] = 1.0
    return matrix


class FakeModel:
    """Records where the script moves the NLI model."""

    def __init__(self) -> None:
        """Start without a device."""
        self.moved_to: dict = {}

    def to(self, **kwargs: object) -> FakeModel:
        """Remember device and dtype."""
        self.moved_to = kwargs
        return self


class FakeClusterer:
    """Stands in for probly's clusterer; records every batch of questions it gets."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.model = FakeModel()
        self.calls: list[dict] = []

    def pairwise_nli_many(
        self, groups: list[list[str]], contexts: list[str | None] | None = None, deduplicate: bool = True
    ) -> list[torch.Tensor]:
        """Return one fake matrix per question."""
        self.calls.append({"n": len(groups), "contexts": contexts, "deduplicate": deduplicate})
        return [fake_matrix(group) for group in groups]


def rows_for_test() -> list[dict]:
    """Three questions with different answer patterns."""
    answers = [["Paris", "Paris", "Lyon"], ["Rome", "Milan", "Naples"], ["Bern", "Bern", "Bern"]]
    return [
        {"id": i, "question": f"Q{i}?", "greedy_answer": a[0], "samples": a, "samples_ll_mean": [-0.1, -0.5, -1.0]}
        for i, a in enumerate(answers)
    ]


def write_rows(path: Path, rows: list[dict]) -> Path:
    """Write rows as a JSONL data cache."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeClusterer:
    """Replace the NLI model and the device choice of the script for this test."""
    clusterer = FakeClusterer()
    monkeypatch.setattr(
        build_nli_cache, "HFGreedySemanticClusterer", SimpleNamespace(from_model_name=lambda *_, **__: clusterer)
    )
    monkeypatch.setattr(build_nli_cache, "pick_device", lambda: "cpu")
    return clusterer


def run_build(data: Path, out: Path, *, use_question: bool = False, deduplicate: bool = True, limit: int = 0) -> None:
    """Call build() with a small batch size and two questions per batch."""
    build_nli_cache.build(
        str(data),
        NLI_MODEL,
        str(out),
        use_question=use_question,
        deduplicate=deduplicate,
        batch_size=4,
        limit=limit,
        questions_per_batch=2,
    )


@pytest.mark.usefixtures("fake")
def test_build_writes_a_cache_the_runner_can_reuse(tmp_path, capsys):
    rows = rows_for_test()
    out = tmp_path / "triviaqa_3.nli.deberta-large.npz"
    run_build(write_rows(tmp_path / "triviaqa_3.jsonl", rows), out)

    cache = load_nli_cache(out)
    settings = NliSettings("deberta-large")
    assert cache["ids"] == [0, 1, 2]
    assert cache["nli_model"] == NLI_MODEL
    assert cache["deduplicate"] is True
    assert cache["fingerprints"] == [matrix_fingerprint(r, settings) for r in rows]
    assert len(read_cached_matrices(out, settings)) == 3  # the runner would reuse every question
    for got, row in zip(cache["matrices"], rows, strict=True):
        torch.testing.assert_close(got, fake_matrix(question_texts(row)))
    assert "saved" in capsys.readouterr().out


def test_build_batches_questions_and_passes_the_settings(tmp_path, fake):
    data = write_rows(tmp_path / "d.jsonl", rows_for_test())
    run_build(data, tmp_path / "out.npz", use_question=True, deduplicate=False)

    assert [call["n"] for call in fake.calls] == [2, 1]  # questions_per_batch=2
    assert fake.calls[0]["contexts"] == ["Q0?", "Q1?"]
    assert all(call["deduplicate"] is False for call in fake.calls)
    assert fake.model.moved_to == {"device": "cpu", "dtype": torch.float32}
    assert load_nli_cache(tmp_path / "out.npz")["use_question"] is True


@pytest.mark.usefixtures("fake")
def test_build_limit_keeps_the_first_rows(tmp_path):
    run_build(write_rows(tmp_path / "d.jsonl", rows_for_test()), tmp_path / "out.npz", limit=2)
    assert load_nli_cache(tmp_path / "out.npz")["ids"] == [0, 1]


@pytest.mark.usefixtures("fake")
def test_build_rejects_rows_without_answers(tmp_path):
    rows = [{"id": 0, "question": "Q?", "greedy_answer": "Paris"}]
    with pytest.raises(KeyError, match="samples"):
        run_build(write_rows(tmp_path / "d.jsonl", rows), tmp_path / "out.npz")
