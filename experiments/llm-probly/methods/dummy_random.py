"""Dummy method: random uncertainty scores.

Only used to test the pipeline. Its AUROC should be around 0.5 (no signal).
"""

from __future__ import annotations

import random

NAME = "dummy_random"
FAMILY = "baseline"
NEEDS_MODEL = False


def score(rows, model=None, tok=None) -> dict:
    """Give every question a random uncertainty between 0 and 1."""
    rng = random.Random(0)  # fixed seed, so the dummy result is the same every run
    return {
        "method": NAME,
        "model": rows[0]["model"],
        "answer": "greedy",
        "ids": [row["id"] for row in rows],
        "uncertainty": [rng.random() for _ in rows],
        "failed": [False] * len(rows),
        "seconds_per_question": 0.0,
    }
