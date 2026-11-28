"""Plugin: CoCoA with MSP, -log P(greedy answer) times the inconsistency of the samples.

Reads the cache (greedy token log-probs) and the shared NLI matrix, so it needs no model
call and no extra NLI pass. See ``common/hybrid.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from common.hybrid import cocoa_record
from common.nli_matrix import nli_settings

from methods import _se_common, msp

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

NAME = "cocoa_msp"


def score(
    rows: list[dict[str, Any]],
    model: PreTrainedModel | None = None,
    tok: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """Score every row; see CONTRACT.md for the record format."""
    _se_common.prepare_matrices(rows)
    record = cocoa_record(NAME, rows, msp.score(rows), _se_common.get_matrix)
    record["nli"] = nli_settings().tag
    return record
