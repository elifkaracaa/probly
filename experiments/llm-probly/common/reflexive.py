"""Week-1 reference p(True) for the reflexive methods (zero-shot and brainstorm), one prompt at a time.

The plugins use the library now: ``probly.representer.self_evaluation.huggingface.HFSelfEvaluator``
(C1) scores the same prompts in batches with left padding. This module is kept as the reference that
the library is checked against (``tests/test_self_eval_equivalence.py``, ``scripts/bench_self_eval.py``)
and because the week-1 notebooks (zero-shot, brainstorm, mode comparison) document the method with it.

Same prompt for both methods; the only difference is the brainstorm line. Two modes:

- ``"chat_groups"`` (zero-shot): chat template, next token compared over the token groups
  True/A/(A vs False/B/(B (only strings that are single tokens).
- ``"plain_ab"`` (brainstorm): no chat template, the prompt ends with ``" ("``, next token A vs B.

Raw baseline: :func:`true_logit` (logit of the single token "True", as LM-Polygraph does).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

DEFAULT_MODE = "plain_ab"
"""Chosen by the sanity comparison (notebooks/reflexive_mode_comparison.ipynb)."""

GROUPS = {
    "chat_groups": (["True", " True", "A", " A", "(A"], ["False", " False", "B", " B", "(B"]),
    "plain_ab": (["A"], ["B"]),
}
"""Strings whose next-token probability counts as True / False, per mode."""


def build_text(question: str, answer: str, brainstorm: Sequence[str] | None = None) -> str:
    """Self-evaluation text of one answer, with the brainstorm line when ``brainstorm`` is given."""
    ideas = f"Here are some ideas that were brainstormed: {', '.join(brainstorm)}\n" if brainstorm else ""
    return (
        f"Question: {question}\n"
        f"{ideas}"
        f"Possible answer: {answer}\n"
        "Is the possible answer:\n"
        " (A) True\n"
        " (B) False\n"
        "The possible answer is:"
    )


def build_prompt(
    tok: PreTrainedTokenizerBase,
    question: str,
    answer: str,
    brainstorm: Sequence[str] | None = None,
    mode: str = DEFAULT_MODE,
) -> str:
    """Prompt text of one answer in the given mode.

    Raises:
        ValueError: For an unknown mode.
    """
    text = build_text(question, answer, brainstorm)
    if mode == "chat_groups":
        return str(
            tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        )
    if mode == "plain_ab":
        return text + " ("
    msg = f"unknown mode: {mode}"
    raise ValueError(msg)


def single_token_ids(tok: PreTrainedTokenizerBase, strings: Sequence[str]) -> list[int]:
    """Token ids of the strings that are exactly one token for this tokenizer."""
    ids = set()
    for s in strings:
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return sorted(ids)


@torch.no_grad()
def next_token_logits(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    question: str,
    answer: str,
    brainstorm: Sequence[str] | None = None,
    mode: str = DEFAULT_MODE,
) -> torch.Tensor:
    """Logits of the next token after the prompt (float32, on the model's device)."""
    text = build_prompt(tok, question, answer, brainstorm, mode)
    # The chat template already contains its special tokens, so none are added again.
    enc = tok(text, return_tensors="pt", add_special_tokens=(mode != "chat_groups")).to(model.device)
    return model(**enc).logits[0, -1].float()


def p_true_details(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    question: str,
    answer: str,
    brainstorm: Sequence[str] | None = None,
    mode: str = DEFAULT_MODE,
) -> dict[str, Any]:
    """Normalized p(True) of one answer plus coverage.

    Args:
        model: Causal language model.
        tok: Its tokenizer.
        question: Question.
        answer: Answer to judge.
        brainstorm: Sampled answers for the brainstorm line, or None for zero-shot.
        mode: ``"plain_ab"`` or ``"chat_groups"``.

    Returns:
        ``{"p_true": P(True) / (P(True) + P(False)), "coverage": P(True) + P(False), "mode": mode}``.

    Raises:
        ValueError: If one of the token groups is empty for this tokenizer.
    """
    logp = torch.log_softmax(next_token_logits(model, tok, question, answer, brainstorm, mode), -1)
    true_ids, false_ids = (single_token_ids(tok, group) for group in GROUPS[mode])
    if not true_ids or not false_ids:
        msg = f"empty token group for mode {mode!r} (true={true_ids}, false={false_ids}); check the tokenizer"
        raise ValueError(msg)
    lt = torch.logsumexp(logp[true_ids], dim=0)
    lf = torch.logsumexp(logp[false_ids], dim=0)
    return {
        "p_true": torch.sigmoid(lt - lf).item(),  # = P(T) / (P(T) + P(F))
        "coverage": (lt.exp() + lf.exp()).item(),
        "mode": mode,
    }


def p_true(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    question: str,
    answer: str,
    brainstorm: Sequence[str] | None = None,
    mode: str = DEFAULT_MODE,
) -> float:
    """Confidence in [0, 1]; for the score contract the uncertainty is ``1 - p_true``."""
    return float(p_true_details(model, tok, question, answer, brainstorm, mode)["p_true"])


def true_logit(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    question: str,
    answer: str,
    brainstorm: Sequence[str] | None = None,
) -> float:
    """Raw baseline: logit of the single token "True" (LM-Polygraph), chat-template prompt.

    High = more confident; for the score contract use ``uncertainty = -true_logit``.
    """
    logits = next_token_logits(model, tok, question, answer, brainstorm, mode="chat_groups")
    true_id = tok.encode("True", add_special_tokens=False)[0]
    return float(logits[true_id].item())
