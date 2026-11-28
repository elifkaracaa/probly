"""Plugin: 1 - share of the samples with the same meaning as the greedy answer, from the shared NLI matrix."""

from __future__ import annotations

from common.nli_matrix import cluster_shares

from methods._se_common import get_matrix, score_rows_with

NAME = "greedy_cluster_share"


def score(rows: list[dict], model: object = None, tok: object = None) -> dict:
    """Score every row; see CONTRACT.md for the record format."""
    return score_rows_with(rows, NAME, lambda row: 1.0 - cluster_shares(get_matrix(row))[1])
