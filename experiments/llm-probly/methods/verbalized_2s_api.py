"""Verbalized confidence (2S) for a model behind an OpenAI-compatible API (F2).

Same chat (question, the model's answer, "How confident are you ... 0 and 100")
and parsing as ``verbalized_2s``; greedy, 8 tokens, 1 call per question.
Uncertainty = 1 - confidence; if no number can be read: uncertainty 1.0,
failed (CONTRACT.md rule 3). Raw replies are stored to check parsing.
"""

from __future__ import annotations

import time

from common.text_only import client_from_rows, verbalized_2s

NAME = "verbalized_2s_api"
FAMILY = "self_eval"
NEEDS_MODEL = False
SCALE = 100


def score(rows: list[dict], model: object = None, tok: object = None) -> dict:
    """Score the greedy answer of every row with verbalized confidence through the API.

    Args:
        rows: Cache rows from ``scripts/generate_api.py``.
        model: Unused (API method).
        tok: Unused (API method).

    Returns:
        A score record in the CONTRACT.md format plus API cost fields.
    """
    client = client_from_rows(rows)
    uncertainty, failed, replies = [], [], []
    start = time.perf_counter()
    for row in rows:
        confidence, no_number, reply = verbalized_2s(
            client, row["question"], row["greedy_answer"], scale=SCALE
        )
        uncertainty.append(1.0 if no_number else 1.0 - confidence)
        failed.append(no_number)
        replies.append(reply)
    seconds = (time.perf_counter() - start) / len(rows)
    usage = client.usage.snapshot()
    return {
        "method": NAME,
        "model": rows[0]["model"],
        "answer": "greedy",
        "probability": True,
        "scale": SCALE,
        "replies": replies,
        "ids": [row["id"] for row in rows],
        "uncertainty": uncertainty,
        "failed": failed,
        "seconds_per_question": seconds,
        "model_calls_per_question": 1,
        "api_requests_per_question": usage["requests"] / len(rows),
        "api_tokens_per_question": (usage["prompt_tokens"] + usage["completion_tokens"])
        / len(rows),
    }
