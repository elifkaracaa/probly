"""Week-1 reference code of the text-only reflexive methods (R3), kept for comparisons.

The plugins use the library now: empirical p(True) runs on
``probly.representer.self_evaluation.huggingface.HFSelfEvaluator`` (C2, batched, more robust reply
parsing) and verbalized confidence lives in ``common/verbalized.py``. This module keeps the week-1
empirical p(True) (one question at a time, the original reply rule) because
``scripts/bench_self_eval.py`` measures the library against it and ``notebooks/text_only_self_evaluation.ipynb``
documents how the method was built. The verbalized names are re-exported from ``common.verbalized``,
so there is only one verbalized implementation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

from common.verbalized import ASK, parse_confidence, verbalized_2s
import numpy as np

from probly.representer.sampler.huggingface import HFTextGenerationSampler

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from transformers._typing import GenerativePreTrainedModel

__all__ = ["ASK", "PT_PROMPT", "empirical_p_true", "is_false", "is_true", "parse_confidence", "verbalized_2s"]

PT_PROMPT = (
    "Question: {q}\nPossible answer: {a}\nIs the possible answer:\n (A) True\n (B) False\nThe possible answer is:"
)
"""Week-1 prompt of empirical p(True); the model continues it with its judgment."""


def is_true(text: str) -> bool:
    """Whether a sampled reply says True (week-1 rule: starts with "true" or the letter A)."""
    text = text.strip().lower()
    return text.startswith("true") or re.match(r"^\(?a\)?(\W|$)", text) is not None


def is_false(text: str) -> bool:
    """Whether a sampled reply says False (week-1 rule: starts with "false" or the letter B)."""
    text = text.strip().lower()
    return text.startswith("false") or re.match(r"^\(?b\)?(\W|$)", text) is not None


def empirical_p_true(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    q: str,
    a: str,
    n: int = 10,
) -> tuple[float, list[str], int]:
    """Week-1 empirical p(True) of one answer: the share of ``n`` sampled judgments that say True.

    Args:
        model: Causal language model.
        tok: Its tokenizer.
        q: Question.
        a: Answer to judge.
        n: Number of sampled judgments.

    Returns:
        ``(p_true, replies, unparsed)``: the share of True replies, the raw replies, and how many
        replies were neither True nor False.
    """
    sampler = HFTextGenerationSampler(
        cast("GenerativePreTrainedModel", model),
        tok,
        num_samples=n,
        temperature=1.0,
        max_new_tokens=3,
        with_log_likelihood=False,
    )
    texts = [str(t) for t in sampler([PT_PROMPT.format(q=q, a=a)]).tensor.text[0]]
    p = float(np.mean([is_true(t) for t in texts]))
    unparsed = sum(not is_true(t) and not is_false(t) for t in texts)
    return p, texts, unparsed
