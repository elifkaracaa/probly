"""Empirical p(True) for a model behind an OpenAI-compatible API (F2).

Same prompt, sampling (10 judgments, temperature 1.0, 3 tokens) and parsing as
``p_true_empirical``, but the judgments come from the API model that generated
the cache (``scripts/generate_api.py``), so no local weights are needed.
Uncertainty = 1 - p_true; if no reply can be parsed: uncertainty 1.0, failed
(CONTRACT.md rule 3). ``parsed_fraction`` is stored as in the HF plugin.
"""

from __future__ import annotations

import time

from common.text_only import client_from_rows, empirical_p_true

NAME = "p_true_empirical_api"
FAMILY = "self_eval"
NEEDS_MODEL = False
N_SAMPLES = 10
SEED = 42


def score(rows: list[dict], model: object = None, tok: object = None) -> dict:
    """Score the greedy answer of every row with empirical p(True) through the API.

    Args:
        rows: Cache rows from ``scripts/generate_api.py``.
        model: Unused (API method).
        tok: Unused (API method).

    Returns:
        A score record in the CONTRACT.md format plus API cost fields.
    """
    client = client_from_rows(rows)
    uncertainty, failed, parsed_fraction = [], [], []
    start = time.perf_counter()
    for i, row in enumerate(rows):
        p, texts, unparsed = empirical_p_true(
            client,
            row["question"],
            row["greedy_answer"],
            n=N_SAMPLES,
            seed=SEED + i * N_SAMPLES,
        )
        parsed_fraction.append(1 - unparsed / len(texts))
        no_reply_parsed = unparsed == len(texts)
        uncertainty.append(1.0 if no_reply_parsed else 1.0 - p)
        failed.append(no_reply_parsed)
    seconds = (time.perf_counter() - start) / len(rows)
    usage = client.usage.snapshot()
    return {
        "method": NAME,
        "model": rows[0]["model"],
        "answer": "greedy",
        "probability": True,
        "parsed_fraction": parsed_fraction,
        "ids": [row["id"] for row in rows],
        "uncertainty": uncertainty,
        "failed": failed,
        "seconds_per_question": seconds,
        "model_calls_per_question": N_SAMPLES,
        "api_requests_per_question": usage["requests"] / len(rows),
        "api_tokens_per_question": (usage["prompt_tokens"] + usage["completion_tokens"])
        / len(rows),
    }
