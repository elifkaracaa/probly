"""Hand-labelled answer pairs for choosing the NLI setup (A4).

A labelled pair says whether two answers to the same question mean the same
thing. The pairs are picked from the TriviaQA-200 caches so that they are
informative for comparing NLI setups: a third where deberta-base and
deberta-large disagree, a third where both say "same meaning" and a third where
both say "different". The NLI decisions are stored with the pair but are not
shown while labelling, so they cannot bias the label.

Label file (``data/nli_hand_labels.json``): a JSON list, one object per pair::

    {"model": "qwen1.5b", "id": 61, "question": "...", "a": "Arsenal", "b": "Arsenal.",
     "index_a": 0, "index_b": 1, "stratum": "both_same", "base_same": true, "large_same": true,
     "same_meaning": true, "labeller": "elif"}

``index_a`` / ``index_b`` are positions in ``question_texts(row)`` (0 = greedy answer).
``same_meaning`` is ``null`` for a pair skipped as unclear.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np

from common.nli_matrix import question_texts, same_meaning

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

LABELS_PATH = Path(__file__).resolve().parent.parent / "data" / "nli_hand_labels.json"
STRATA = ("disagree", "both_same", "both_different")
MAX_CHARS = 160
"""Longer answers are left out; they are slow to label and rare in TriviaQA."""


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and extra spaces (pairs equal after this are trivially the same)."""
    return re.sub(r"[^\w\s]", "", " ".join(text.split()).casefold()).strip()


def candidate_pairs(
    model: str,
    rows: Sequence[dict],
    base: Sequence[torch.Tensor],
    large: Sequence[torch.Tensor],
) -> list[dict]:
    """All non-trivial answer pairs of a model with the base and large NLI decisions.

    Args:
        model: Short model name, stored in each pair.
        rows: Rows of the data cache.
        base: deberta-base matrices (probly order), one per row.
        large: deberta-large matrices (probly order), one per row.

    Returns:
        One dict per pair of different answers to the same question (see the module docstring).
    """
    pairs = []
    for row, m_base, m_large in zip(rows, base, large, strict=True):
        texts = question_texts(row)
        first = {}
        for index, text in enumerate(texts):
            if len(text) <= MAX_CHARS:
                first.setdefault(normalize(text), index)
        indices = sorted(first.values())
        for pos, a in enumerate(indices):
            for b in indices[pos + 1 :]:
                base_same = same_meaning(m_base, a, b)
                large_same = same_meaning(m_large, a, b)
                stratum = "disagree" if base_same != large_same else ("both_same" if base_same else "both_different")
                pairs.append(
                    {
                        "model": model,
                        "id": row["id"],
                        "question": row["question"],
                        "a": texts[a],
                        "b": texts[b],
                        "index_a": a,
                        "index_b": b,
                        "stratum": stratum,
                        "base_same": base_same,
                        "large_same": large_same,
                    }
                )
    return pairs


def select_pairs(candidates: Sequence[dict], per_stratum: int, seed: int = 0) -> list[dict]:
    """Pick ``per_stratum`` pairs of each stratum, at most one pair per question.

    Args:
        candidates: Pairs of one model from :func:`candidate_pairs`.
        per_stratum: Pairs to pick per stratum (fewer if a stratum is too small).
        seed: Random seed; the same seed gives the same pairs.

    Returns:
        The picked pairs, strata mixed in random order.
    """
    rng = np.random.default_rng(seed)
    shuffled = [candidates[i] for i in rng.permutation(len(candidates))]
    used_questions: set[int] = set()
    picked: list[dict] = []
    for stratum in STRATA:
        count = 0
        for pair in shuffled:
            if count == per_stratum:
                break
            if pair["stratum"] == stratum and pair["id"] not in used_questions:
                picked.append(pair)
                used_questions.add(pair["id"])
                count += 1
    return [picked[i] for i in rng.permutation(len(picked))]


def pair_key(pair: dict) -> tuple[str, int, int, int]:
    """Identity of a pair (model, question id, positions of the two answers)."""
    return pair["model"], pair["id"], pair["index_a"], pair["index_b"]


def load_labels(path: str | Path = LABELS_PATH) -> list[dict]:
    """Read the label file (an empty list if it does not exist yet)."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_labels(labels: Sequence[dict], path: str | Path = LABELS_PATH) -> None:
    """Write the label file (indented JSON, final newline)."""
    Path(path).write_text(json.dumps(list(labels), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
