"""Tests for the one-call semantic helper and its docs page (B5)."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, cast

from common.nli_matrix import NliSettings, load_nli_cache, nli_cache_path, nli_settings, set_nli_settings
from common.semantic_scores import (
    SCORES,
    SemanticInputError,
    check_inputs,
    scores_from_matrix,
    semantic_scores,
    semantic_scores_many,
)
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CONTRADICTION, NEUTRAL, ENTAILMENT = 0, 1, 2  # probly's label order


class FakeClusterer:
    """Stands in for the NLI model: texts with the same first word entail each other, others contradict."""

    def __init__(self) -> None:
        self.calls = 0
        self.contexts = []

    def pairwise_nli_many(self, groups, contexts=None, deduplicate=True):
        self.calls += 1
        self.contexts.append(contexts)
        matrices = []
        for texts in groups:
            keys = [text.lower().replace("it is", "").split()[0].strip(".") for text in texts]
            n = len(texts)
            matrix = torch.zeros((n, n, 3))
            for a in range(n):
                for b in range(n):
                    matrix[a, b, ENTAILMENT if keys[a] == keys[b] else CONTRADICTION] = 1.0
            matrix[torch.arange(n), torch.arange(n)] = float("nan")
            matrices.append(matrix)
        return matrices


def test_all_scores_and_clusters_are_returned():
    scores = semantic_scores(["Paris", "Paris", "It is Paris.", "Lyon"], clusterer=cast("Any", FakeClusterer()))

    assert set(SCORES) <= scores.keys()
    assert scores["clusters"] == [0, 0, 0, 1]
    assert scores["num_semantic_sets"] == 2.0
    assert scores["discrete_semantic_entropy"] == pytest.approx(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25)))
    assert math.isnan(scores["semantic_entropy"])  # no log-likelihoods given


def test_agreeing_samples_give_the_lowest_scores():
    scores = semantic_scores(
        ["Paris", "Paris", "Paris"], sample_log_likelihoods=[-0.1, -0.2, -0.3], clusterer=cast("Any", FakeClusterer())
    )

    assert scores["semantic_entropy"] == pytest.approx(0.0)
    assert scores["discrete_semantic_entropy"] == pytest.approx(0.0)
    assert scores["lexical_similarity"] == pytest.approx(-1.0)
    assert scores["num_semantic_sets"] == 1.0
    assert scores["eig_val_laplacian"] == pytest.approx(1.0)
    assert scores["deg_mat"] == pytest.approx(0.0)
    assert scores["eccentricity"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("model", ["qwen0.5b", "qwen1.5b"])
def test_helper_gives_the_benchmark_scores(model):
    """Equivalence: on cached questions the helper's scores equal the stored plugin results."""
    rows = [json.loads(line) for line in (ROOT / f"data/{model}/triviaqa_200.jsonl").open()][:25]
    results = ROOT / f"results/{model}/triviaqa_200"

    for name in SCORES:
        record = json.loads((results / f"{name}.json").read_text())
        tag = record.get("nli", "deberta-base")  # records from before A4 have no tag
        if name == "lexical_similarity":  # no NLI; any cache gives the same score
            tag = "deberta-base"
        settings = NliSettings(model=tag.removesuffix("+question"), use_question=tag.endswith("+question"))
        cache = nli_cache_path(ROOT / f"data/{model}/triviaqa_200.jsonl", settings)
        matrices = load_nli_cache(cache)["matrices"][:25]
        helper = [
            scores_from_matrix(m, r["samples"], r["samples_ll_mean"]) for m, r in zip(matrices, rows, strict=True)
        ]
        assert [h[name] for h in helper] == pytest.approx(record["uncertainty"][:25], abs=1e-5), (name, tag)


def test_many_questions_share_one_nli_call_per_batch_and_log_progress(caplog):
    fake = FakeClusterer()
    items = [{"samples": ["Paris", "Lyon"]}, {"samples": ["1969", "1969"], "sample_log_likelihoods": [-0.1, -0.1]}]

    with caplog.at_level(logging.INFO, logger="common.semantic_scores"):
        results = semantic_scores_many(items, clusterer=cast("Any", fake))

    assert fake.calls == 1
    assert [r["num_semantic_sets"] for r in results] == [2.0, 1.0]
    assert "scored 2/2 questions" in caplog.text
    assert "semantic_entropy is NaN" in caplog.text


@pytest.mark.parametrize(
    ("samples", "lls", "match"),
    [
        ("Paris", None, "got one string"),
        (["Paris"], None, "at least 2 sampled answers"),
        (["Paris", "  "], None, "sample 1 is empty"),
        (["Paris", 3], None, "sample 1 is int, not text"),
        (["Paris", "Lyon"], [-0.1], "2 samples but 1 log-likelihoods"),
        (["Paris", "Lyon"], [-0.1, float("nan")], "log-likelihood 1 is nan"),
    ],
)
def test_clear_errors(samples, lls, match):
    with pytest.raises(SemanticInputError, match=match):
        check_inputs(samples, lls)


def test_errors_come_before_the_nli_model_is_loaded():
    fake = FakeClusterer()

    with pytest.raises(SemanticInputError, match=r"^need at least 2"):
        semantic_scores(["only one"], clusterer=cast("Any", fake))
    with pytest.raises(SemanticInputError, match="question 1: sample 0 is empty"):
        semantic_scores_many([{"samples": ["a", "b"]}, {"samples": ["", "b"]}], clusterer=cast("Any", fake))
    with pytest.raises(SemanticInputError, match="question 0: missing 'samples'"):
        semantic_scores_many([{"answers": ["a", "b"]}], clusterer=cast("Any", fake))
    assert fake.calls == 0


def snippet(text: str) -> str:
    """The code between the snippet markers of scripts/semantic_demo.py."""
    return text.split("# --- snippet start ---\n", 1)[1].split("# --- snippet end ---", 1)[0].strip()


def test_docs_page_shows_exactly_the_demo_script():
    script = snippet((ROOT / "scripts/semantic_demo.py").read_text())
    page = (ROOT / "docs/semantic_methods.md").read_text()

    assert f"```python\n{script}\n```" in page


def test_question_goes_in_front_of_the_answers_only_when_asked():
    fake = FakeClusterer()
    samples = ["Paris", "Lyon"]

    semantic_scores(samples, question="Capital of France?", clusterer=cast("Any", fake))
    semantic_scores(samples, question="Capital of France?", use_question=True, clusterer=cast("Any", fake))

    assert fake.contexts == [[None], ["Capital of France?"]]


def test_many_questions_pass_their_own_question():
    fake = FakeClusterer()
    items = [{"samples": ["Paris", "Lyon"], "question": "q1"}, {"samples": ["1969", "1969"], "question": "q2"}]

    semantic_scores_many(items, use_question=True, clusterer=cast("Any", fake))

    assert fake.contexts == [["q1", "q2"]]


def test_use_question_without_a_question_says_how_to_fix_it():
    with pytest.raises(SemanticInputError, match=r"use_question=True needs the question text\. Pass question="):
        semantic_scores(["Paris", "Lyon"], use_question=True, clusterer=cast("Any", FakeClusterer()))


def test_use_question_follows_the_benchmark_setting_by_default():
    fake = FakeClusterer()
    old = nli_settings()
    set_nli_settings(use_question=True)
    try:
        semantic_scores(["Paris", "Lyon"], question="Capital of France?", clusterer=cast("Any", fake))
    finally:
        set_nli_settings(use_question=old.use_question)

    assert fake.contexts == [["Capital of France?"]]
