"""Zero-shot p(True): ask the model whether its own greedy answer is true.

Thin wrapper around probly's ``HFSelfEvaluator`` in plain_ab mode (week 1, R1; moved into the
library in week 3, C1). Uncertainty = 1 - p_true (score contract rule 1).
Also stores coverage = P(A) + P(B) per question, to check that the model
really answers with A/B.
"""

from __future__ import annotations

import math
import time

from probly.representer.self_evaluation.huggingface import HFSelfEvaluator

NAME = "p_true_zeroshot"
FAMILY = "self_eval"
NEEDS_MODEL = True
MODE = "plain_ab"
BATCH_SIZE = 8


def score(rows, model=None, tok=None) -> dict:
    """Score the greedy answer of every row with zero-shot p(True)."""
    if model is None or tok is None:
        raise ValueError(f"{NAME} needs the model (NEEDS_MODEL = True)")

    evaluator = HFSelfEvaluator(model, tok, mode=MODE, batch_size=BATCH_SIZE)
    start = time.perf_counter()
    result = evaluator.evaluate([row["question"] for row in rows], [row["greedy_answer"] for row in rows])
    seconds = (time.perf_counter() - start) / len(rows)

    ok = [math.isfinite(p) for p in result.p_true]
    uncertainty = [1.0 - p if good else 1.0 for p, good in zip(result.p_true, ok, strict=True)]  # rule 3: failed -> 1.0

    return {
        "method": NAME,
        "model": rows[0]["model"],
        "answer": "greedy",
        "probability": True,  # uncertainty = 1 - probability, so summary.py computes ECE
        "mode": MODE,
        "coverage": result.coverage.tolist(),
        "ids": [row["id"] for row in rows],
        "uncertainty": uncertainty,
        "failed": [not good for good in ok],
        "seconds_per_question": seconds,
    }
