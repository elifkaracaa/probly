"""Label answer pairs by hand (A4): do two answers to a question mean the same thing?

The pairs come from the qwen0.5b and qwen1.5b TriviaQA-200 caches; see
``common/nli_labels.py`` for how they are picked and for the file format. The
NLI decisions (and, for a second labeller, the first person's labels) are not
shown, so they cannot bias the label. Labels are saved after every answer, so
you can stop with ``q`` and continue later.

Run from ``experiments/llm-probly``::

    # first labeller, all strata (10 pairs per stratum and model)
    uv run python -m scripts.label_nli_pairs --labeller elif
    # only the near-duplicate pairs ("1984" vs "1984.")
    uv run python -m scripts.label_nli_pairs --labeller elif --strata near_duplicate
    # second labeller: 20 of the pairs already labelled, saved to a separate file
    uv run python -m scripts.label_nli_pairs --labeller eylul --second-labeller 20
    # agreement between the two labellers (Cohen's kappa)
    uv run python -m scripts.label_nli_pairs --agreement data/nli_hand_labels_eylul.jsonl

Keys: ``y`` same meaning, ``n`` different meaning, ``s`` skip (unclear), ``q`` quit.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from common.data import cache_path, load_jsonl
from common.nli_labels import (
    LABELS_PATH,
    STRATA,
    candidate_pairs,
    cohen_kappa,
    load_labels,
    pair_key,
    save_labels,
    select_pairs,
)
from common.nli_matrix import load_nli_cache
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

MODELS = ("qwen0.5b", "qwen1.5b")
ANSWERS = {"y": True, "n": False, "s": None}
NLI_FIELDS = ("base_same", "large_same", "same_meaning", "labeller")


def pairs_to_label(per_stratum: int, seed: int, strata: Sequence[str] = STRATA) -> list[dict]:
    """Pick the pairs of all models (``per_stratum`` per stratum and model)."""
    picked = []
    for model in MODELS:
        rows = load_jsonl(cache_path(model, 200))
        base = load_nli_cache(cache_path(model, 200, suffix=".nli.npz"))["matrices"]
        large = load_nli_cache(cache_path(model, 200, suffix=".nli.deberta-large.npz"))["matrices"]
        picked += select_pairs(candidate_pairs(model, rows, base, large), per_stratum, seed=seed, strata=strata)
    return picked


def second_labeller_pairs(first_labels: Sequence[dict], n: int, seed: int = 0) -> list[dict]:
    """``n`` of the pairs the first person labelled (not skipped), without their labels or NLI decisions."""
    usable = [label for label in first_labels if label["same_meaning"] is not None]
    rng = np.random.default_rng(seed)
    chosen = [usable[i] for i in rng.permutation(len(usable))[:n]]
    return [{key: value for key, value in pair.items() if key not in NLI_FIELDS} for pair in chosen]


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


def labeller_agreement(first: Sequence[dict], second: Sequence[dict]) -> dict:
    """Raw agreement and Cohen's kappa on the pairs both people labelled (skips left out)."""
    by_key = {pair_key(label): label["same_meaning"] for label in first if label["same_meaning"] is not None}
    both = [
        (by_key[pair_key(label)], label["same_meaning"])
        for label in second
        if label["same_meaning"] is not None and pair_key(label) in by_key
    ]
    if not both:
        msg = "no pair labelled by both people"
        raise ValueError(msg)
    a, b = [x for x, _ in both], [y for _, y in both]
    return {"pairs": len(both), "agreement": sum(x == y for x, y in both) / len(both), "kappa": cohen_kappa(a, b)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labeller", help="your name, stored with each label")
    parser.add_argument("--per-stratum", type=int, default=10, help="pairs per stratum and model")
    parser.add_argument("--strata", nargs="+", choices=STRATA, default=list(STRATA), help="strata to label")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--second-labeller", type=int, metavar="N", help="label N already labelled pairs again")
    parser.add_argument("--agreement", metavar="FILE", help="compare FILE with data/nli_hand_labels.jsonl")
    args = parser.parse_args()
    if not args.agreement and not args.labeller:
        parser.error("--labeller is needed to label pairs")
    if args.agreement:
        result = labeller_agreement(load_labels(), load_labels(args.agreement))
        print(f"{result['pairs']} pairs: agreement {result['agreement']:.2f}, Cohen's kappa {result['kappa']:.2f}")
    elif args.second_labeller:
        out = LABELS_PATH.with_name(f"nli_hand_labels_{args.labeller}.jsonl")
        label_pairs(second_labeller_pairs(load_labels(), args.second_labeller), args.labeller, out)
    else:
        label_pairs(pairs_to_label(args.per_stratum, args.seed, args.strata), args.labeller)
