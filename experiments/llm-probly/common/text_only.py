"""Text-only reflexive methods for models behind an OpenAI-compatible API (F2).

The prompts and parsers are the week-1 ones from ``scripts/r3_methods.py``, so
API results can be compared with the local HF results. They are copied, not
imported, because plugins must not import from ``scripts/`` and that module
imports torch / transformers. When C2 moves empirical p(True) into the
library, these copies should be replaced by imports from there.

Cache rows written by ``scripts/generate_api.py`` carry an ``"api"`` field
(``{"model": ..., "base_url": ...}``); :func:`client_from_rows` reads it.
"""

from __future__ import annotations

import os
import re

from common.openai_client import OpenAICompatibleClient

# ---------------- Empirical p(True) (same as scripts/r3_methods.py) ----------------
PT_PROMPT = (
    "Question: {q}\nPossible answer: {a}\nIs the possible answer:\n (A) True\n (B) False\nThe possible answer is:"
)


def is_true(text: str) -> bool:
    """Whether a judgment reply says True (``true...``, ``A``, ``(A)``, ``A)``).

    Args:
        text: Model reply.

    Returns:
        True if the reply starts with a True answer.
    """
    t = text.strip().lower()
    return t.startswith("true") or re.match(r"^\(?a\)?(\W|$)", t) is not None


def is_false(text: str) -> bool:
    """Whether a judgment reply says False (``false...``, ``B``, ``(B)``, ``B)``).

    Args:
        text: Model reply.

    Returns:
        True if the reply starts with a False answer.
    """
    t = text.strip().lower()
    return t.startswith("false") or re.match(r"^\(?b\)?(\W|$)", t) is not None


def empirical_p_true(
    client: OpenAICompatibleClient,
    question: str,
    answer: str,
    n: int = 10,
    seed: int | None = None,
) -> tuple[float, list[str], int]:
    """Sample ``n`` True/False judgments (temperature 1.0, 3 tokens) and count True.

    Same definition as the week-1 HF version: ``p`` is the share of *all* replies
    that say True, so unparsable replies count as not True.

    Args:
        client: API client.
        question: Question text.
        answer: Answer to judge.
        n: Number of judgments.
        seed: Base seed for the sampled judgments.

    Returns:
        ``(p_true, replies, number of replies that are neither True nor False)``.
    """
    texts = client.chat(
        [{"role": "user", "content": PT_PROMPT.format(q=question, a=answer)}],
        n=n,
        temperature=1.0,
        max_tokens=3,
        seed=seed,
    )
    p = sum(is_true(t) for t in texts) / len(texts)
    unparsed = sum(not is_true(t) and not is_false(t) for t in texts)
    return p, texts, unparsed


# ---------------- Verbalized 2S (same as scripts/r3_methods.py) ----------------
ASK = {
    100: "How confident are you that your answer is correct? Reply with only a number between 0 and 100.",
    10: "How confident are you that your answer is correct? Reply with only a number between 0 and 10.",
}


def parse_confidence(reply: str, scale: int) -> float | None:
    """Turn the first number in a reply into a confidence in [0, 1].

    Decimals ``<= 1`` (e.g. ``"0.99"``) are taken as a fraction already;
    anything else is clipped to ``[0, scale]`` and divided by ``scale``.

    Args:
        reply: Model reply.
        scale: The scale the prompt asked for (10 or 100).

    Returns:
        Confidence in [0, 1], or ``None`` if the reply has no number.
    """
    m = re.search(r"(\d+(?:\.\d+)?)", reply)
    if m is None:
        return None
    x = float(m.group(1))
    if "." in m.group(1) and x <= 1.0:
        return x
    return min(max(x, 0.0), scale) / scale


def verbalized_2s(
    client: OpenAICompatibleClient, question: str, answer: str, scale: int = 100
) -> tuple[float, bool, str]:
    """Ask for a confidence after the model's own answer (greedy, 8 tokens).

    Args:
        client: API client.
        question: Question text.
        answer: The model's answer, inserted as its own assistant turn.
        scale: 10 or 100.

    Returns:
        ``(confidence, failed, raw reply)``; confidence is 0 when no number is found.
    """
    chat = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": ASK[scale]},
    ]
    reply = client.chat(chat, temperature=0.0, max_tokens=8)[0]
    c = parse_confidence(reply, scale)
    return (0.0, True, reply) if c is None else (c, False, reply)


# ---------------- Client from cache rows ----------------


def client_from_rows(rows: list[dict]) -> OpenAICompatibleClient:
    """Build the API client for a cache written by ``scripts/generate_api.py``.

    ``$LLM_PROBLY_API_BASE_URL`` overrides the stored base URL (e.g. when the
    server moved); the key is read from ``$OPENAI_API_KEY``.

    Args:
        rows: Cache rows.

    Returns:
        A client for the model that generated the rows.

    Raises:
        ValueError: If the rows were not generated through an API.
    """
    api = rows[0].get("api") if rows else None
    if not api:
        msg = (
            "these rows have no 'api' field: API methods only run on caches made by "
            "scripts/generate_api.py (use verbalized_2s / p_true_empirical for local models)"
        )
        raise ValueError(msg)
    base_url = os.environ.get("LLM_PROBLY_API_BASE_URL") or api["base_url"]
    return OpenAICompatibleClient(model=api["model"], base_url=base_url)
