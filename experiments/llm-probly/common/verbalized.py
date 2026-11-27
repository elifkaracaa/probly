"""Verbalized confidence, two-stage (2S): ask the model how confident it is in its own answer.

Moved from scripts/r3_methods.py (week 1, R3) so that the plugin in methods/ does not import from scripts/.
The behavior is unchanged.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from probly.representer.sampler.huggingface import HFTextGenerationSampler

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

ASK = {
    100: "How confident are you that your answer is correct? Reply with only a number between 0 and 100.",
    10: "How confident are you that your answer is correct? Reply with only a number between 0 and 10.",
}

PERCENT_SCALE = 100
"""The 0-100 scale, where a reply like "0.99" is read as a fraction."""

# C4: the plain prompts above make the small models answer almost always "95" or "100". This variant
# tells the model what the ends of the scale mean, so that low numbers become a possible answer.
ASK_ANCHORED = (
    "How confident are you that your answer is correct? Reply with only a number from 0 to 10, "
    "where 0 means your answer is certainly wrong and 10 means it is certainly correct."
)


def parse_confidence(reply: str, scale: int) -> float | None:
    """Turn the first number in the reply into a confidence in 0..1, or None if there is no number.

    On the 0-100 scale, decimals that are at most 1 (like "0.99") are already a fraction and are
    taken as they are. On the 0-10 scale they are not: "1.0" there means 1 out of 10.
    """
    match = re.search(r"(\d+(?:\.\d+)?)", reply)
    if match is None:
        return None
    value = float(match.group(1))
    if scale == PERCENT_SCALE and "." in match.group(1) and value <= 1.0:
        return value
    return min(max(value, 0.0), scale) / scale


def verbalized_2s(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    question: str,
    answer: str,
    scale: int = 100,
    ask: str | None = None,
) -> tuple[float, bool, str]:
    """Return (confidence 0..1, failed, raw reply). Unreadable replies give confidence 0 (uncertainty 1.0).

    ``ask`` replaces the default question for ``scale`` (used by the C4 prompt experiment).
    """
    chat = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": ask or ASK[scale]},
    ]
    sampler = HFTextGenerationSampler(
        model, tok, num_samples=1, do_sample=False, max_new_tokens=8, with_log_likelihood=False
    )
    reply = str(sampler([chat]).tensor.text[0, 0])
    confidence = parse_confidence(reply, scale)
    return (0.0, True, reply) if confidence is None else (confidence, False, reply)
