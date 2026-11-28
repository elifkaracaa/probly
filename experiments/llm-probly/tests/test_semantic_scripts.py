"""Smoke tests for the semantic example scripts: sample-count curve, decomposed SE example, semantic demo.

No model is loaded: model loading, generation, NLI and plugin scoring are replaced by small fakes,
and every output goes to tmp_path.
"""

from __future__ import annotations

import json
import logging
import runpy

import numpy as np
import torch
import transformers

from scripts import decomposed_example, sample_count_curve

CONTRADICTION, ENTAILMENT = 0, 2  # probly's label order


def matrix_from_groups(groups: list[int]) -> torch.Tensor:
    """Question matrix: same group entails, different groups contradict; index 0 = greedy answer."""
    n = len(groups)
    matrix = torch.zeros((n, n, 3))
    for a in range(n):
        for b in range(n):
            matrix[a, b, ENTAILMENT if groups[a] == groups[b] else CONTRADICTION] = 1.0
    matrix[torch.arange(n), torch.arange(n)] = float("nan")
    return matrix


def test_sample_count_curve_writes_both_tables_and_the_plot(tmp_path, monkeypatch):
    # greedy answer + 10 samples per question: all agree, two clusters, ten clusters, greedy alone
    groups = [[0] * 11, [0] * 6 + [1] * 5, list(range(11)), [1] + [0] * 10]
    rows = [
        {"id": i, "samples": [f"a{g}" for g in grp[1:]], "samples_ll_mean": [-0.5] * 10, "correct": i % 2 == 0}
        for i, grp in enumerate(groups)
    ]
    correct = np.array([r["correct"] for r in rows])
    matrices = [matrix_from_groups(grp) for grp in groups]
    monkeypatch.setattr(sample_count_curve, "load_model", lambda *_args: (rows, matrices, correct))
    monkeypatch.setattr(sample_count_curve, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["sample_count_curve", "--models", "m", "--n", "4", "--draws", "2"])

    sample_count_curve.main()

    analysis = tmp_path / "m" / "triviaqa_4" / "analysis"
    record = json.loads((analysis / "sample_count.json").read_text())
    assert {r["k"] for r in record["curve"]} == set(range(2, 11))
    assert {r["check_at"] for r in record["early_stop"]} == {2, 3, 4, 5}
    assert (analysis / "sample_count.md").read_text().startswith("# Sample-count curve: `m/triviaqa_4`")
    assert (tmp_path / "plots" / "triviaqa_4_m_k_curve.png").exists()


class FakeModel:
    def to(self, _device):
        return self

    def eval(self):
        return self


def fake_decomposition(rows, _model, _tok):
    """What the decomposed SE plugin returns, without generating: 3 clarifications x 5 answers per question."""
    details = {
        "clarifications": ["c1", "c2", "c3"],
        "answers": [f"a{i}" for i in range(15)],
        "clusters": list(range(15)),
    }
    n = len(rows)
    return {
        "uncertainty": [1.0 + i for i in range(n)],
        "aleatoric": [0.5 * i for i in range(n)],
        "epistemic": [1.0 + 0.5 * i for i in range(n)],
        "details": [details] * n,
    }


def test_decomposed_example_writes_record_and_report(tmp_path, monkeypatch):
    rows = [{"question": f"Question {i}?", "greedy_answer": "x", "correct": i % 2 == 0} for i in range(4)]
    monkeypatch.setattr(decomposed_example, "load_jsonl", lambda _path: rows)
    monkeypatch.setattr(decomposed_example, "load_model", lambda _name: (FakeModel(), None))
    monkeypatch.setattr(decomposed_example, "pick_device", lambda: "cpu")
    monkeypatch.setattr(decomposed_example.plugin, "score", fake_decomposition)
    monkeypatch.setattr(decomposed_example.plugin, "SEED", 0)
    monkeypatch.setattr(decomposed_example, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["decomposed_example", "--model", "qwen0.5b", "--questions", "4"])

    decomposed_example.main()

    out = tmp_path / "qwen0.5b" / "triviaqa_200" / "analysis"
    assert json.loads((out / "decomposed_se.json").read_text()) == fake_decomposition(rows, None, None)
    report = (out / "decomposed_se.md").read_text()
    assert "first 4 questions" in report
    assert "| total |" in report
    assert "**Question 3?**" in report  # highest total, aleatoric and epistemic


def test_semantic_demo_scores_only_the_non_empty_samples(monkeypatch, capsys):
    texts = ["Paris", "", "Paris", "Lyon"]  # the empty answer must be dropped before scoring
    seen = {}

    def fake_scores(samples, sample_log_likelihoods):
        seen["samples"], seen["lls"] = samples, sample_log_likelihoods
        return {
            "clusters": [0, 0, 1],
            "semantic_entropy": 0.6,
            "discrete_semantic_entropy": 0.6,
            "num_semantic_sets": 2.0,
            "eig_val_laplacian": 2.0,
        }

    monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "from_pretrained", lambda _name: "tokenizer")
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda _name: "model")
    monkeypatch.setattr("common.generation.stop_at_newline_config", lambda _model, _tok: None)
    monkeypatch.setattr(
        "common.generation.generate_answers",
        lambda *_args, **_kwargs: [{"text": text, "ll_mean": -0.5} for text in texts],
    )
    monkeypatch.setattr("common.semantic_scores.semantic_scores", fake_scores)
    monkeypatch.setattr(logging, "basicConfig", lambda **_kwargs: None)
    logger = logging.getLogger("common.semantic_scores")
    level = logger.level
    try:
        runpy.run_module("scripts.semantic_demo", run_name="__main__")
    finally:
        logger.setLevel(level)  # the demo turns on the helper's progress messages

    assert seen == {"samples": ["Paris", "Paris", "Lyon"], "lls": [-0.5, -0.5, -0.5]}
    out = capsys.readouterr().out
    assert "clusters: [0, 0, 1]" in out
    assert "semantic_entropy: 0.600" in out
