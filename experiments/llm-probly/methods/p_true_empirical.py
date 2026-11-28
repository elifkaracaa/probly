"""Empirical p(True): share of 10 sampled "(A) True / (B) False" judgments that say True.

The model judges its own answer 10 times at temperature 1.0.

Uses probly's ``HFSelfEvaluator.sample_judgments`` (week 1, R3; moved into the library in week 3, C2). The 10
replies of a question come from one ``generate`` call (``num_return_sequences``). Putting several questions into
one call is possible (BATCH_SIZE), but on CPU it was 3 to 5 times slower, so the default is 1 question per call.
No logits needed, but 10 sampled replies per question (the most expensive method).
Uncertainty = 1 - p_true (score contract rule 1).
Also stores parsed_fraction per question: how many of the 10 replies could be
read as True or False (the empirical version of coverage).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch

from probly.representer.self_evaluation.huggingface import HFSelfEvaluator, parse_judgment

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NAME = "p_true_empirical"
IMPLEMENTATION = "HFSelfEvaluator.sample_judgments"
"""Where the judgments come from (C2, library code), stored in every record."""
N_SAMPLES = 10
BATCH_SIZE = 1  # questions per generate call; more was slower on CPU (check on GPU before raising)
SEED = 42  # same seed as the week 1 run, but sampling in batches draws different random numbers


def score(
    rows: list[dict[str, Any]],
    model: PreTrainedModel | None = None,
    tok: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Score the greedy answer of every row with empirical p(True)."""
    if model is None or tok is None:
        msg = f"{NAME} needs the model (needs_model=True in common/registry.py)"
        raise ValueError(msg)

    torch.manual_seed(SEED)  # once, before sampling
    evaluator = HFSelfEvaluator(model, tok, batch_size=BATCH_SIZE)
    start = time.perf_counter()
    replies = evaluator.sample_judgments(
        [row["question"] for row in rows],
        [row["greedy_answer"] for row in rows],
        num_samples=N_SAMPLES,
    )
    seconds = (time.perf_counter() - start) / len(rows)

    uncertainty, failed, parsed_fraction = [], [], []
    for texts in replies:
        judgments = [parse_judgment(text) for text in texts]
        unparsed = sum(judgment is None for judgment in judgments)
        p = sum(judgment is True for judgment in judgments) / len(judgments)
        parsed_fraction.append(1 - unparsed / len(judgments))
        no_reply_parsed = unparsed == len(judgments)
        uncertainty.append(1.0 if no_reply_parsed else 1.0 - p)  # contract rule 3
        failed.append(no_reply_parsed)

    return {
        "method": NAME,
        "model": rows[0]["model"],
        "answer": "greedy",
        "probability": True,  # uncertainty = 1 - probability, so summary.py computes ECE
        "parsed_fraction": parsed_fraction,
        "ids": [row["id"] for row in rows],
        "uncertainty": uncertainty,
        "failed": failed,
        "seconds_per_question": seconds,
        "model_calls_per_question": N_SAMPLES,
        # Provenance: scores of the week-1 code (scripts/r3_methods.py) are not reproducible with this one.
        "implementation": IMPLEMENTATION,
        "device": str(model.device),
        "dtype": str(model.dtype).removeprefix("torch."),
    }
