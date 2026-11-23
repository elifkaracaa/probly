"""Shared code for the semantic-entropy plugins (SE and discrete SE).

Both plugins read the same per-question NLI matrix (A1). The matrix is computed
once per question and kept in memory, so when the runner calls
``semantic_entropy`` and then ``discrete_semantic_entropy`` in the same process,
the NLI model runs only for the first one.

Which NLI model is used, and whether the question goes in front of each answer, comes from
``common.nli_matrix.nli_settings()`` (default deberta-large, answers only; the runner sets it
from the config keys ``nli_model`` and ``nli_use_question``). Each record says which setup it used
in its ``nli`` field.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from common.nli_matrix import (
    QUESTIONS_PER_BATCH,
    compute_matrices,
    nli_pair_count,
    nli_settings,
    pick_device,
    question_texts,
    se_and_dse,
)

from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer

if TYPE_CHECKING:
    import torch

_clusterers: dict[str, HFGreedySemanticClusterer] = {}
_matrices: dict[tuple, torch.Tensor] = {}
nli_pairs_scored = 0
"""Number of NLI pairs sent to the model in this process (for before / after reports)."""


def get_clusterer() -> HFGreedySemanticClusterer:
    """Load the NLI clusterer of the current settings once per process."""
    name = nli_settings().hf_name
    if name not in _clusterers:
        clusterer = HFGreedySemanticClusterer.from_model_name(name)
        clusterer.model.to(pick_device())
        _clusterers[name] = clusterer
    return _clusterers[name]


def matrix_key(row: dict) -> tuple:
    """Memo key of a question matrix: NLI setup, question (if used) and the answers."""
    settings = nli_settings()
    question = row["question"] if settings.use_question else None
    return (settings.tag, question, *question_texts(row))


def prepare_matrices(rows: list[dict]) -> None:
    """Compute the missing NLI matrices of ``rows``, ``QUESTIONS_PER_BATCH`` questions per model batch."""
    global nli_pairs_scored  # noqa: PLW0603
    missing = {matrix_key(row): row for row in rows if matrix_key(row) not in _matrices}
    todo = list(missing.items())
    for start in range(0, len(todo), QUESTIONS_PER_BATCH):
        chunk = todo[start : start + QUESTIONS_PER_BATCH]
        matrices = compute_matrices(
            get_clusterer(), [row for _, row in chunk], use_question=nli_settings().use_question
        )
        for (key, row), matrix in zip(chunk, matrices, strict=True):
            _matrices[key] = matrix
            nli_pairs_scored += nli_pair_count(row)


def get_matrix(row: dict) -> torch.Tensor:
    """Return the NLI matrix of one question, computing it only the first time."""
    prepare_matrices([row])
    return _matrices[matrix_key(row)]


def compute_se_and_dse(row: dict) -> tuple[float, float]:
    """Semantic entropy and discrete semantic entropy of one question."""
    return se_and_dse(get_matrix(row), row["samples_ll_mean"])


def score_rows(rows: list[dict], name: str, which: int) -> dict:
    """Score record for SE (``which=0``) or discrete SE (``which=1``), see CONTRACT.md."""
    ids, unc, failed = [], [], []
    pairs_before = nli_pairs_scored
    t0 = time.time()
    prepare_matrices(rows)
    for i, r in enumerate(rows):
        ids.append(r.get("id", i))
        try:
            unc.append(compute_se_and_dse(r)[which])
            failed.append(False)
        except Exception:  # noqa: BLE001
            unc.append(1.0)
            failed.append(True)
    return {
        "method": name,
        "model": rows[0].get("model"),
        "answer": "greedy",
        "ids": ids,
        "uncertainty": unc,
        "failed": failed,
        "seconds_per_question": (time.time() - t0) / len(rows),
        "nli_pairs_per_question": (nli_pairs_scored - pairs_before) / len(rows),
        "nli": nli_settings().tag,
    }
