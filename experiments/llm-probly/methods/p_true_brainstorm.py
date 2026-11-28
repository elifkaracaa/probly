"""Brainstorm p(True): zero-shot p(True) with the model's own samples in the prompt.

The prompt adds "Here are some ideas that were brainstormed: ..." with the cached samples.

Thin wrapper around probly's ``HFSelfEvaluator`` in plain_ab mode (week 1, R2; moved into the library in week 3, C1).
The samples come from the cache (row["samples"]), so no new generation.
Uncertainty = 1 - p_true (score contract rule 1).
Also stores coverage = P(A) + P(B) per question.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

from probly.representer.self_evaluation.huggingface import HFSelfEvaluator

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NAME = "p_true_brainstorm"
MODE = "plain_ab"
BATCH_SIZE = 8


def score(
    rows: list[dict[str, Any]],
    model: PreTrainedModel | None = None,
    tok: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Score the greedy answer of every row with brainstorm p(True)."""
    if model is None or tok is None:
        msg = f"{NAME} needs the model (needs_model=True in common/registry.py)"
        raise ValueError(msg)

    # Without samples the prompt silently becomes zero-shot, so those rows are marked as failed.
    with_samples = [i for i, row in enumerate(rows) if row.get("samples")]
    uncertainty = [1.0] * len(rows)
    failed = [True] * len(rows)
    coverage = [float("nan")] * len(rows)

    evaluator = HFSelfEvaluator(model, tok, mode=MODE, batch_size=BATCH_SIZE)
    start = time.perf_counter()
    result = evaluator.evaluate(
        [rows[i]["question"] for i in with_samples],
        [rows[i]["greedy_answer"] for i in with_samples],
        [rows[i]["samples"] for i in with_samples],
    )
    seconds = (time.perf_counter() - start) / len(rows)

    for i, p, cov in zip(with_samples, result.p_true, result.coverage, strict=True):
        coverage[i] = float(cov)
        if math.isfinite(p):
            uncertainty[i] = 1.0 - float(p)  # contract rule 3: failed -> 1.0 (the default above)
            failed[i] = False

    return {
        "method": NAME,
        "model": rows[0]["model"],
        "answer": "greedy",
        "probability": True,  # uncertainty = 1 - probability, so summary.py computes ECE
        "mode": MODE,
        "coverage": coverage,
        "ids": [row["id"] for row in rows],
        "uncertainty": uncertainty,
        "failed": failed,
        "seconds_per_question": seconds,
    }
