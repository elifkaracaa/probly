"""Generate an answer like the benchmark and compute the raw uncertainty.

Generation reuses ``common.generation`` (chat template, stop at the first
newline, only tokens before the stop are scored), with the same prompt and
settings as ``scripts/generate.py``, so the API's numbers match the cache the
profile was fitted on.

``SCORERS`` maps ``profile.method.name`` to a function that turns one
generation into a raw uncertainty (higher = more uncertain).
``selector.profile.confidence`` then calibrates it.

Only methods that need nothing beyond the greedy generation are supported.
Methods that need extra model calls (p(True), semantic entropy, ...) are added
when their library versions are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from common.generation import generate_answers, stop_at_newline_config
import torch

from probly.representer.sampler.huggingface import load_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

# Keep in sync with scripts/generate.py (TriviaQA-style prompt).
INSTRUCTION = "Answer the following question with as few words as possible."
MAX_NEW_TOKENS = 32


@dataclass
class Generation:
    """A greedy generation with per-token statistics.

    Attributes:
        answer: Decoded answer text.
        token_logprobs: Log-probability of each scored token.
        token_entropies: Entropy (nats) of the next-token distribution at each scored step.
    """

    answer: str
    token_logprobs: list[float]
    token_entropies: list[float]


def likelihood_sum(gen: Generation) -> float:
    """Uncertainty = minus the summed log likelihood (``likelihood_sum`` in the benchmark).

    Args:
        gen: The generation.

    Returns:
        ``-sum(token_logprobs)``.
    """
    return -sum(gen.token_logprobs)


def likelihood_mean(gen: Generation) -> float:
    """Uncertainty = minus the mean log likelihood (``methods/likelihood_mean.py``).

    Args:
        gen: The generation.

    Returns:
        ``-mean(token_logprobs)``; 0.0 for an empty generation, as in the cache.
    """
    n = len(gen.token_logprobs)
    return -sum(gen.token_logprobs) / n if n else 0.0


def msp(gen: Generation) -> float | None:
    """Maximum sequence probability baseline (``methods/msp.py``).

    Args:
        gen: The generation.

    Returns:
        ``-log P(greedy answer)``, which is ``-sum(token_logprobs)``; None for an empty generation.
    """
    return -sum(gen.token_logprobs) if gen.token_logprobs else None


def mean_token_entropy(gen: Generation) -> float | None:
    """Mean token entropy baseline (``methods/mean_token_entropy.py``).

    Args:
        gen: The generation.

    Returns:
        Mean entropy of the next-token distributions; None for an empty generation.
    """
    ents = gen.token_entropies
    return sum(ents) / len(ents) if ents else None


# Every definition below is checked against its plugin in methods/ by
# test_scorers_match_the_plugins. There is no perplexity scorer: the plugin was removed
# (it ranks exactly like likelihood_mean), so profiles use likelihood_mean instead.
SCORERS: dict[str, Callable[[Generation], float | None]] = {
    "likelihood_sum": likelihood_sum,
    "likelihood_mean": likelihood_mean,
    "msp": msp,
    "mean_token_entropy": mean_token_entropy,
}


@lru_cache(maxsize=2)
def _load(model_id: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig]:
    """Load and cache the model, tokenizer and newline-stop generation config.

    Args:
        model_id: Hugging Face model id.

    Returns:
        A ``(model, tokenizer, generation_config)`` tuple.
    """
    model, tok = load_model(model_id, model_kwargs={"dtype": torch.float32})
    model.eval()
    return model, tok, stop_at_newline_config(model, tok)


def generate(model_id: str, question: str) -> Generation:
    """Generate the greedy answer exactly as the benchmark does.

    Args:
        model_id: Hugging Face id of an instruct model with a chat template.
        question: The question text.

    Returns:
        The answer with per-token log-probabilities and entropies.
    """
    model, tok, config = _load(model_id)
    prompt = f"{INSTRUCTION}\nQuestion: {question}"
    result = generate_answers(
        model,
        tok,
        prompt,
        num_return_sequences=1,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        generation_config=config,
    )[0]
    return Generation(
        answer=result["text"],
        token_logprobs=result["token_logprobs"],
        token_entropies=result["token_entropies"],
    )
