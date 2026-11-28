"""Plugin: 1 - share of the samples in the largest meaning cluster, from the shared NLI matrix."""

from __future__ import annotations

from common.nli_matrix import cluster_shares

from methods._se_common import get_matrix, score_rows_with

NAME = "largest_cluster_share"


def score(rows: list[dict], model: object = None, tok: object = None) -> dict:
    """Score every row; see CONTRACT.md for the record format."""
    return score_rows_with(rows, NAME, lambda row: 1.0 - cluster_shares(get_matrix(row))[0])
