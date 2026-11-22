"""Tests for the persistent NLI cache (A5): reuse, invalidation and the runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cli
from common import io, nli_matrix
from common.data import cache_path, load_jsonl
from common.nli_matrix import (
    NliSettings,
    count_cached_questions,
    load_nli_cache,
    matrix_fingerprint,
    nli_cache_path,
    read_cached_matrices,
    save_nli_cache,
)
from methods import _se_common, discrete_semantic_entropy, semantic_entropy
import pytest
import torch

from scripts import run_benchmark

if TYPE_CHECKING:
    from pathlib import Path


class FakeClusterer:
    """Stands in for the NLI model: answers that start with the same letter entail each other."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.groups: list[list[str]] = []

    def pairwise_nli_many(
        self, groups: list[list[str]], contexts: list[str | None] | None = None, deduplicate: bool = True
    ) -> list[torch.Tensor]:
        """Return one matrix per group (probly order: contradiction, neutral, entailment)."""
        del contexts, deduplicate
        self.groups += groups
        matrices = []
        for group in groups:
            n = len(group)
            matrix = torch.zeros((n, n, 3))
            for a in range(n):
                for b in range(n):
                    matrix[a, b, 2 if group[a][0] == group[b][0] else 0] = 1.0
            matrices.append(matrix)
        return matrices


class NoNli:
    """Fails the test if the NLI model is called at all."""

    def pairwise_nli_many(self, *args: object, **kwargs: object) -> list[torch.Tensor]:
        """Raise: the cache should have answered."""
        msg = "NLI model called although the cache should be used"
        raise AssertionError(msg)


def rows_for_test() -> list[dict]:
    """Three questions with different answer patterns."""
    answers = [["Paris", "Paris", "Lyon"], ["Rome", "Milan", "Naples"], ["Bern", "Bern", "Bern"]]
    return [
        {
            "id": i,
            "model": "m",
            "question": f"Q{i}?",
            "greedy_answer": a[0],
            "samples": a,
            "samples_ll_mean": [-0.1, -0.5, -1.0],
        }
        for i, a in enumerate(answers)
    ]


def new_process(monkeypatch: pytest.MonkeyPatch, clusterer: object, settings: NliSettings | None = None) -> None:
    """Forget everything kept in memory, as if the runner started again."""
    settings = settings or NliSettings()
    monkeypatch.setattr(nli_matrix, "_settings", settings)
    monkeypatch.setattr(_se_common, "_clusterers", {settings.hf_name: clusterer})
    monkeypatch.setattr(_se_common, "_matrices", {})
    monkeypatch.setattr(_se_common, "_disk", {})
    monkeypatch.setattr(_se_common, "_nli_seconds", {})


@pytest.fixture
def cache_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A cache file in a temporary folder, set for this test only."""
    path = tmp_path / "triviaqa_3.nli.deberta-large.npz"
    monkeypatch.setattr(nli_matrix, "_cache_file", path)
    return path


def test_second_run_does_no_nli_work(monkeypatch, cache_file):
    rows = rows_for_test()
    new_process(monkeypatch, FakeClusterer())
    first = semantic_entropy.score(rows)
    assert cache_file.exists()
    assert first["nli_pairs_per_question"] > 0

    new_process(monkeypatch, NoNli())
    second = semantic_entropy.score(rows)
    dse = discrete_semantic_entropy.score(rows)

    assert second["nli_pairs_per_question"] == dse["nli_pairs_per_question"] == 0
    assert second["uncertainty"] == first["uncertainty"]
    assert second["failed"] == [False] * 3


def test_nli_cost_is_kept_when_the_cache_is_used(monkeypatch, cache_file):
    """seconds_per_question is this run's time; nli_seconds_per_question keeps what the NLI step cost."""

    class SlowClusterer(FakeClusterer):
        def pairwise_nli_many(self, groups, contexts=None, deduplicate=True):
            time.sleep(0.03 * len(groups))  # 0.03 s per question
            return super().pairwise_nli_many(groups, contexts, deduplicate)

    rows = rows_for_test()
    new_process(monkeypatch, SlowClusterer())
    first = semantic_entropy.score(rows)
    assert first["nli_seconds_per_question"] == pytest.approx(0.03, abs=0.02)
    assert load_nli_cache(cache_file)["seconds_per_question"] == pytest.approx(first["nli_seconds_per_question"])

    new_process(monkeypatch, NoNli())
    second = semantic_entropy.score(rows)
    assert second["seconds_per_question"] < 0.03
    assert second["nli_seconds_per_question"] == pytest.approx(first["nli_seconds_per_question"])


def test_cached_matrices_equal_computed_ones(monkeypatch, cache_file):
    rows = rows_for_test()
    new_process(monkeypatch, FakeClusterer())
    _se_common.prepare_matrices(rows)
    computed = [_se_common.get_matrix(r) for r in rows]

    cache = load_nli_cache(cache_file)
    settings = nli_matrix.nli_settings()
    assert cache["ids"] == [0, 1, 2]
    assert cache["fingerprints"] == [matrix_fingerprint(r, settings) for r in rows]
    for got, want in zip(cache["matrices"], computed, strict=True):
        torch.testing.assert_close(got, want, equal_nan=True)


def test_only_changed_answers_are_recomputed(monkeypatch, cache_file):
    rows = rows_for_test()
    new_process(monkeypatch, FakeClusterer())
    semantic_entropy.score(rows)

    rows[1]["samples"] = ["Rome", "Rome", "Rome"]  # regenerated answers for question 1
    fake = FakeClusterer()
    new_process(monkeypatch, fake)
    record = semantic_entropy.score(rows)

    assert fake.groups == [["Rome", "Rome", "Rome", "Rome"]]  # questions 0 and 2 came from the file
    assert record["uncertainty"][1] == pytest.approx(0.0)
    assert count_cached_questions(cache_file, nli_matrix.nli_settings(), rows) == 3  # file was updated


def test_question_conditioning_changes_the_fingerprint():
    row = rows_for_test()[0]
    plain = matrix_fingerprint(row, NliSettings("deberta-large"))
    with_question = matrix_fingerprint(row, NliSettings("deberta-large", use_question=True))
    other_model = matrix_fingerprint(row, NliSettings("deberta-base"))
    assert len({plain, with_question, other_model}) == 3
    # Without question conditioning the question text does not matter.
    assert matrix_fingerprint({**row, "question": "other"}, NliSettings("deberta-large")) == plain


def test_file_from_another_nli_setup_is_not_used(tmp_path):
    rows = rows_for_test()
    path = tmp_path / "x.npz"
    base = NliSettings("deberta-base")
    save_nli_cache(
        path,
        [torch.zeros((4, 4, 3)) for _ in rows],
        [r["id"] for r in rows],
        nli_model=base.hf_name,
        use_question=False,
        seconds_per_question=0.1,
        fingerprints=[matrix_fingerprint(r, base) for r in rows],
    )
    assert len(read_cached_matrices(path, base)) == 3
    assert read_cached_matrices(path, NliSettings("deberta-large")) == {}
    assert read_cached_matrices(path, NliSettings("deberta-base", use_question=True)) == {}


def test_old_file_without_fingerprints_is_recomputed(monkeypatch, cache_file):
    rows = rows_for_test()
    save_nli_cache(  # a week-2 / A4 file: same setup, but nothing to check the answers against
        cache_file,
        [torch.zeros((4, 4, 3)) for _ in rows],
        [r["id"] for r in rows],
        nli_model=NliSettings().hf_name,
        use_question=False,
        seconds_per_question=0.1,
    )
    fake = FakeClusterer()
    new_process(monkeypatch, fake)
    semantic_entropy.score(rows)

    assert len(fake.groups) == 3
    assert load_nli_cache(cache_file)["fingerprints"] is not None  # rewritten in the new format


def test_nli_cache_path_names():
    data = "data/qwen0.5b/triviaqa_200.jsonl"
    assert str(nli_cache_path(data, NliSettings("deberta-base"))) == "data/qwen0.5b/triviaqa_200.nli.npz"
    assert nli_cache_path(data, NliSettings()).name == "triviaqa_200.nli.deberta-large.npz"
    assert nli_cache_path(data, NliSettings(use_question=True)).name == "triviaqa_200.nli.deberta-large+question.npz"


def test_without_cache_file_nothing_is_written(monkeypatch, tmp_path):
    monkeypatch.setattr(nli_matrix, "_cache_file", None)
    monkeypatch.chdir(tmp_path)
    new_process(monkeypatch, FakeClusterer())
    semantic_entropy.score(rows_for_test())
    assert list(tmp_path.iterdir()) == []


def test_runner_reuses_the_cache_on_a_second_run(monkeypatch, tmp_path, capsys):
    """Two runs with --force on the 5-question data: the second one sends nothing to the NLI model."""
    monkeypatch.setattr(io, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(run_benchmark, "RESULTS_ROOT", tmp_path)
    nli_file = tmp_path / "triviaqa_5.nli.deberta-large.npz"
    monkeypatch.setattr(run_benchmark, "nli_cache_path", lambda *_: nli_file)
    config = tmp_path / "config.yaml"
    config.write_text("model: qwen0.5b\ndataset: triviaqa\nn: 5\nprecision: float32\n")
    argv = ["run", "--config", str(config), "--methods", "semantic_entropy,discrete_semantic_entropy", "--force"]

    new_process(monkeypatch, FakeClusterer())
    cli.main(argv)
    assert "(0/5 questions cached)" in capsys.readouterr().out

    new_process(monkeypatch, NoNli())
    cli.main(argv)
    assert "(5/5 questions cached)" in capsys.readouterr().out
    record = io.load_scores(tmp_path / "qwen0.5b" / "triviaqa_5" / "semantic_entropy.json")
    assert record["nli_pairs_per_question"] == 0
    assert nli_matrix.nli_cache_file() is None  # the runner resets the cache file after the run
    assert len(load_jsonl(cache_path("qwen0.5b", 5))) == 5
