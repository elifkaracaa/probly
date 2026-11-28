"""Plugin: CoCoA with zero-shot p(True), (1 - p(True)) times the inconsistency of the samples.

Runs zero-shot p(True) (one scoring pass) and reads the shared NLI matrix, so it adds no NLI
pass. For runs where ``p_true_zeroshot.json`` already exists, ``scripts/hybrid_from_results.py``
builds the same record without the model. See ``common/hybrid.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from common.hybrid import cocoa_record
from common.nli_matrix import nli_settings

from methods import _se_common, p_true_zeroshot

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NAME = "cocoa_p_true"
CONFIDENCE = p_true_zeroshot.NAME


def score(
    rows: list[dict[str, Any]],
    model: PreTrainedModel | None = None,
    tok: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Score every row; see CONTRACT.md for the record format."""
    confidence = p_true_zeroshot.score(rows, model, tok)
    _se_common.prepare_matrices(rows)
    record = cocoa_record(NAME, rows, confidence, _se_common.get_matrix)
    record["nli"] = nli_settings().tag
    return record
