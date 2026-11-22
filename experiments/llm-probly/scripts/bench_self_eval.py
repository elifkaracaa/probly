"""C1 before/after: seconds per question of the old p(True) code vs the library's batched evaluator.

Runs both on the same questions of one cache and prints a Markdown table plus the largest score difference.
Old = ``common.reflexive.p_true_details`` one prompt at a time (week 1). New = the plugins, which now use
``HFSelfEvaluator`` with batches.

Example:
    uv run python -m scripts.bench_self_eval --run qwen0.5b --n 200 --questions 50
"""

from __future__ import annotations

import argparse
import time

import torch

from common import reflexive
from common.data import load_data
from methods import p_true_brainstorm, p_true_empirical, p_true_zeroshot
from scripts.r3_methods import empirical_p_true
from probly.representer.sampler.huggingface import load_model

MODELS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="qwen0.5b", choices=sorted(MODELS))
    parser.add_argument("--n", type=int, default=200, help="cache size to read")
    parser.add_argument("--questions", type=int, default=50, help="how many questions to time")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size of the new code")
    args = parser.parse_args()

    p_true_zeroshot.BATCH_SIZE = p_true_brainstorm.BATCH_SIZE = args.batch_size
    rows = load_data(args.run, args.n)[: args.questions]
    model, tok = load_model(MODELS[args.run], model_kwargs={"dtype": torch.float32})
    print(f"{args.run}: {len(rows)} questions, device {model.device}, torch threads {torch.get_num_threads()}")

    # Warm-up so that one-time costs (first forward pass, allocations) are not timed.
    reflexive.p_true_details(model, tok, rows[0]["question"], rows[0]["greedy_answer"], mode="plain_ab")
    p_true_zeroshot.score(rows[:2], model, tok)

    results = []
    variants = (("p_true_zeroshot", p_true_zeroshot, False), ("p_true_brainstorm", p_true_brainstorm, True))
    for name, plugin, use_samples in variants:
        start = time.perf_counter()
        old = []
        for r in rows:
            ideas = r["samples"] if use_samples else None
            details = reflexive.p_true_details(model, tok, r["question"], r["greedy_answer"], ideas, mode="plain_ab")
            old.append(1.0 - details["p_true"])
        old_s = (time.perf_counter() - start) / len(rows)

        start = time.perf_counter()
        new = plugin.score(rows, model, tok)
        new_s = (time.perf_counter() - start) / len(rows)

        diff = max(abs(a - b) for a, b in zip(old, new["uncertainty"], strict=True))
        results.append((name, old_s, new_s, diff))

    # Empirical p(True) samples its replies, so the scores differ run to run: compare cost and mean p(True).
    start = time.perf_counter()
    torch.manual_seed(p_true_empirical.SEED)
    old_p = [
        empirical_p_true(model, tok, r["question"], r["greedy_answer"], n=p_true_empirical.N_SAMPLES)[0] for r in rows
    ]
    old_s = (time.perf_counter() - start) / len(rows)
    p_true_empirical.BATCH_SIZE = args.batch_size
    new = p_true_empirical.score(rows, model, tok)
    new_s = new["seconds_per_question"]
    new_p = [1.0 - u for u in new["uncertainty"]]
    print(f"\nempirical p(True): before {old_s:.3f} s/question, after {new_s:.3f} s/question ({old_s / new_s:.2f}x)")
    print(f"  mean p(True) {sum(old_p) / len(old_p):.3f} -> {sum(new_p) / len(new_p):.3f}; "
          f"mean parsed fraction {sum(new['parsed_fraction']) / len(rows):.3f}")

    print("\n| method | before (s/question) | after (s/question) | speed-up | max abs score diff |")
    print("|---|---|---|---|---|")
    for name, old_s, new_s, diff in results:
        print(f"| {name} | {old_s:.3f} | {new_s:.3f} | {old_s / new_s:.2f}x | {diff:.1e} |")


if __name__ == "__main__":
    main()
