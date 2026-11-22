"""Label answer pairs by hand (A4): do two answers to a question mean the same thing?

The pairs come from the qwen0.5b and qwen1.5b TriviaQA-200 caches; see
``common/nli_labels.py`` for how they are picked and for the file format. The
NLI decisions are not shown, so they cannot bias the label. Labels are saved
after every answer, so you can stop with ``q`` and continue later.

Run from ``experiments/llm-probly``::

    uv run python -m scripts.label_nli_pairs --labeller elif

Keys: ``y`` same meaning, ``n`` different meaning, ``s`` skip (unclear), ``q`` quit.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from common.data import cache_path, load_jsonl
from common.nli_labels import LABELS_PATH, candidate_pairs, load_labels, pair_key, save_labels, select_pairs
from common.nli_matrix import load_nli_cache

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

MODELS = ("qwen0.5b", "qwen1.5b")
ANSWERS = {"y": True, "n": False, "s": None}


def pairs_to_label(per_stratum: int, seed: int) -> list[dict]:
    """Pick the pairs of all models (``per_stratum`` per stratum and model)."""
    picked = []
    for model in MODELS:
        rows = load_jsonl(cache_path(model, 200))
        base = load_nli_cache(cache_path(model, 200, suffix=".nli.npz"))["matrices"]
        large = load_nli_cache(cache_path(model, 200, suffix=".nli.deberta-large.npz"))["matrices"]
        picked += select_pairs(candidate_pairs(model, rows, base, large), per_stratum, seed=seed)
    return picked


def label_pairs(
    pairs: list[dict],
    labeller: str,
    path: Path = LABELS_PATH,
    ask: Callable[[str], str] = input,
) -> int:
    """Ask for a label for every pair that is not labelled yet; save after each answer.

    Args:
        pairs: Pairs to label.
        labeller: Name stored with each label.
        path: Label file.
        ask: Function that shows a prompt and returns the typed answer (``input`` by default).

    Returns:
        Number of pairs labelled in this session.
    """
    labels = load_labels(path)
    done = {pair_key(label) for label in labels}
    todo = [pair for pair in pairs if pair_key(pair) not in done]
    print(f"{len(done)} pairs labelled, {len(todo)} to go. y = same meaning, n = different, s = skip, q = quit")
    new = 0
    for number, pair in enumerate(todo, start=1):
        print(f"\n[{number}/{len(todo)}] {pair['question']}")
        print(f"  A: {pair['a']}")
        print(f"  B: {pair['b']}")
        answer = ""
        while answer not in {*ANSWERS, "q"}:
            answer = ask("  same meaning? [y/n/s/q] ").strip().lower()
        if answer == "q":
            break
        labels.append({**pair, "same_meaning": ANSWERS[answer], "labeller": labeller})
        save_labels(labels, path)
        new += 1
    print(f"\nsaved {len(labels)} labels to {path}")
    return new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labeller", required=True, help="your name, stored with each label")
    parser.add_argument("--per-stratum", type=int, default=10, help="pairs per stratum and model (3 strata, 2 models)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    label_pairs(pairs_to_label(args.per_stratum, args.seed), args.labeller)
