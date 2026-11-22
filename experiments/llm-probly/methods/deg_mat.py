"""Degree matrix (B1): how weakly the samples agree with each other on average.

uncertainty = 1 - mean degree / K, with the degree of a sample the sum of its
NLI similarities to all samples. Follows LM-Polygraph's DegMat (NLI score,
entail affinity; Lin et al., 2024). Higher = more uncertain.
"""

from __future__ import annotations

from methods._graph_common import score_rows

NAME = "deg_mat"
FAMILY = "agreement"
NEEDS_MODEL = False


def score(rows: list[dict], model: object = None, tok: object = None) -> dict:  # noqa: ARG001
    """Score every row from the shared NLI matrix (see methods/_graph_common.py)."""
    return score_rows(rows, NAME)
