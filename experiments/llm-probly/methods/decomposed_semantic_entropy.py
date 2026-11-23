"""Decomposed semantic entropy (B4): total = aleatoric (ambiguous question) + epistemic (model does not know).

The model rephrases each question 3 times (probly's ``HFQuestionClarifier``) and
answers each rephrasing 5 times; all 15 answers are clustered by meaning with
one NLI matrix (A1), and the entropy of the clusters is split as in Hou et al.
(2024). See common/decomposed_se.py.

uncertainty = total; the record also has ``aleatoric``, ``epistemic`` and
``details`` (clarifications, answers and clusters per question).

Opt-in (registry): about 20 generation passes per question, so a normal run
skips it; run it with ``--methods decomposed_semantic_entropy``.

The 15 answers are new on every run, so their matrices are computed directly with the
benchmark's clusterer (``nli_settings()``) and never go into the persistent NLI cache (A5),
which holds the matrices of the cached samples, one per question id.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from common.decomposed_se import N_ANSWERS, N_CLARIFICATIONS, clarify_and_answer, decompose
from common.nli_matrix import QUESTIONS_PER_BATCH, compute_matrices, nli_pair_count, nli_settings, sample_block
import torch

from methods import _se_common
from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NAME = "decomposed_semantic_entropy"
SEED = 0
"""Random seed set before the clarifications and answers are sampled, so a rerun gives the same numbers."""


def score(
    rows: list[dict[str, Any]],
    model: PreTrainedModel | None = None,
    tok: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Total uncertainty per question, with its aleatoric / epistemic split in the record."""
    if model is None or tok is None:
        msg = f"{NAME} generates clarifications and answers, so it needs the model and tokenizer."
        raise ValueError(msg)
    start = time.perf_counter()
    torch.manual_seed(SEED)
    generated = clarify_and_answer(model, tok, [r["question"] for r in rows])
    pseudo_rows = [
        {"question": r["question"], "greedy_answer": r["greedy_answer"], "samples": g["answers"]}
        for r, g in zip(rows, generated, strict=True)
    ]
    settings = nli_settings()
    clusterer = _se_common.get_clusterer()
    matrices = []
    for first in range(0, len(pseudo_rows), QUESTIONS_PER_BATCH):
        chunk = pseudo_rows[first : first + QUESTIONS_PER_BATCH]
        matrices += compute_matrices(clusterer, chunk, use_question=settings.use_question)

    total, aleatoric, epistemic, failed, details = [], [], [], [], []
    for matrix, gen in zip(matrices, generated, strict=True):
        try:
            clusters = HFGreedySemanticClusterer.cluster_nli_matrix(sample_block(matrix))
            parts = decompose(clusters.tolist(), N_CLARIFICATIONS, N_ANSWERS)
            failed.append(False)
        except Exception:  # noqa: BLE001
            clusters, parts = None, {"total": 1.0, "aleatoric": 0.0, "epistemic": 0.0}
            failed.append(True)
        total.append(parts["total"])
        aleatoric.append(parts["aleatoric"])
        epistemic.append(parts["epistemic"])
        details.append({**gen, "clusters": None if clusters is None else clusters.tolist()})
    return {
        "method": NAME,
        "model": rows[0].get("model"),
        "answer": "greedy",
        "ids": [r["id"] for r in rows],
        "uncertainty": total,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "failed": failed,
        "seconds_per_question": (time.perf_counter() - start) / len(rows),
        "nli_pairs_per_question": sum(nli_pair_count(p) for p in pseudo_rows) / len(rows),
        "nli": settings.tag,
        "details": details,
    }
