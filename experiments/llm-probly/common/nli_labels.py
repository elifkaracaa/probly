"""Hand-labelled answer pairs for choosing the NLI setup (A4).

A labelled pair says whether two answers to the same question mean the same
thing. The pairs are picked from the TriviaQA-200 caches so that they are
informative for comparing NLI setups, in four strata: pairs where deberta-base and
deberta-large disagree, pairs where both say "same meaning", pairs where both say
"different", and near duplicates (texts equal up to case, punctuation and spaces,
e.g. "1984" vs "1984."). The NLI decisions are stored with the pair but are not
shown while labelling, so they cannot bias the label.

Label file (``data/nli_hand_labels.jsonl``): one JSON object per line and pair::

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

LABELS_PATH = Path(__file__).resolve().parent.parent / "data" / "nli_hand_labels.jsonl"
STRATA = ("disagree", "both_same", "both_different", "near_duplicate")
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
        One dict per pair of different answers to the same question (see the module docstring):
        every pair of answers that differ after ``normalize``, plus one near-duplicate pair per
        answer that differs from an earlier one only in case, punctuation or spaces.
    """
    pairs = []
    for row, m_base, m_large in zip(rows, base, large, strict=True):
        texts = question_texts(row)
        first: dict[str, int] = {}  # normalized text -> first index
        near: dict[str, int] = {}  # normalized text -> first index of a different spelling
        for index, text in enumerate(texts):
            if len(text) > MAX_CHARS:
                continue
            key = normalize(text)
            if key not in first:
                first[key] = index
            elif key not in near and text != texts[first[key]]:
                near[key] = index
        indices = sorted(first.values())
        combos = [(a, b, None) for pos, a in enumerate(indices) for b in indices[pos + 1 :]]
        combos += [(first[key], index, "near_duplicate") for key, index in near.items()]
        for a, b, fixed in combos:
            base_same = same_meaning(m_base, a, b)
            large_same = same_meaning(m_large, a, b)
            stratum = fixed or (
                "disagree" if base_same != large_same else ("both_same" if base_same else "both_different")
            )
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


def select_pairs(
    candidates: Sequence[dict],
    per_stratum: int,
    seed: int = 0,
    strata: Sequence[str] = STRATA,
) -> list[dict]:
    """Pick ``per_stratum`` pairs of each stratum, at most one pair per question.

    Args:
        candidates: Pairs of one model from :func:`candidate_pairs`.
        per_stratum: Pairs to pick per stratum (fewer if a stratum is too small).
        seed: Random seed; the same seed gives the same pairs.
        strata: Strata to pick from (all by default).

    Returns:
        The picked pairs, strata mixed in random order.
    """
    rng = np.random.default_rng(seed)
    shuffled = [candidates[i] for i in rng.permutation(len(candidates))]
    used_questions: set[int] = set()
    picked: list[dict] = []
    for stratum in strata:
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
    """Read a label file, one JSON object per line (an empty list if it does not exist yet)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_labels(labels: Sequence[dict], path: str | Path = LABELS_PATH) -> None:
    """Write a label file, one JSON object per line."""
    lines = [json.dumps(label, ensure_ascii=False) + "\n" for label in labels]
    Path(path).write_text("".join(lines), encoding="utf-8")


def cohen_kappa(first: Sequence[bool], second: Sequence[bool]) -> float:
    """Cohen's kappa of two labellers on the same pairs (1 = perfect, 0 = chance level).

    Args:
        first: Labels of the first person.
        second: Labels of the second person, same pairs in the same order.

    Returns:
        ``(observed - expected) / (1 - expected)``; 1.0 if both always give the same single label.
    """
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / n
    p_first, p_second = sum(first) / n, sum(second) / n
    expected = p_first * p_second + (1 - p_first) * (1 - p_second)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def stratum_shares(candidates: Sequence[dict]) -> dict[str, float]:
    """Share of each stratum among all candidate pairs of one model (used as weights).

    Args:
        candidates: Pairs from :func:`candidate_pairs` (all of them, not only the labelled ones).

    Returns:
        ``{stratum: share}``; the shares add up to 1.
    """
    counts = dict.fromkeys(STRATA, 0)
    for pair in candidates:
        counts[pair["stratum"]] += 1
    return {stratum: count / len(candidates) for stratum, count in counts.items()}


def weighted_agreement(
    labels: Sequence[dict],
    decisions: Sequence[bool],
    shares: dict[str, dict[str, float]],
) -> float:
    """Agreement of NLI decisions with hand labels, weighted to the real mix of pairs.

    The labelled pairs over-sample the pairs where deberta-base and deberta-large disagree (a third
    of the labels, about 5 % of all pairs), so a plain accuracy would exaggerate differences. Here the
    accuracy is computed per model and stratum and weighted by that stratum's share of all pairs;
    models get equal weight. Skipped pairs (``same_meaning`` is None) are left out. Strata or models
    without any labelled pair (for example with ``--limit``) are left out and the weights of the rest
    are scaled up, so the result stays a share between 0 and 1.

    Args:
        labels: Labelled pairs (see the module docstring).
        decisions: NLI decision for each labelled pair (True = same meaning), same order.
        shares: ``{model: stratum_shares(...)}``.

    Returns:
        Weighted share of pairs on which the decision equals the hand label.

    Raises:
        ValueError: If no labelled pair is left.
    """
    per_model = []
    for model, model_shares in shares.items():
        weighted, weight = 0.0, 0.0
        for stratum, share in model_shares.items():
            hits = [
                decision == label["same_meaning"]
                for label, decision in zip(labels, decisions, strict=True)
                if label["model"] == model and label["stratum"] == stratum and label["same_meaning"] is not None
            ]
            if hits:
                weighted += share * sum(hits) / len(hits)
                weight += share
        if weight > 0:
            per_model.append(weighted / weight)
    if not per_model:
        msg = "no labelled pair to compare with"
        raise ValueError(msg)
    return sum(per_model) / len(per_model)
