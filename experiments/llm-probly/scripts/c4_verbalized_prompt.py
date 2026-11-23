"""C4: does an anchored prompt fix verbalized confidence? Runs it on the cached greedy answers.

The week 1 prompt ("a number between 0 and 100") makes the small models answer almost always "95"
or "100". This script asks the same question with ``common.verbalized.ASK_ANCHORED`` (scale 0-10,
both ends explained) and compares it with the saved ``verbalized_2s`` result of the same model:
AUROC with a bootstrap interval, how many distinct replies the model gives, and the most common ones.

The new scores go to ``results/<run>/triviaqa_200/c4/verbalized_2s_anchored.json``. That folder is not
read by the summary table, so nothing in the main results changes.

Example:
    uv run python -m scripts.c4_verbalized_prompt --run qwen1.5b
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import time

from common.data import load_data
from common.evaluate import bootstrap_auroc, evaluate
from common.io import RESULTS_ROOT
from common.verbalized import ASK_ANCHORED, verbalized_2s
import torch

from probly.representer.sampler.huggingface import load_model

MODELS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}
SCALE = 10


def summarize(label: str, replies: list[str], uncertainty: list[float], correct: list[bool]) -> None:
    """Print AUROC (with interval) and the reply distribution of one prompt."""
    scores = evaluate(uncertainty, correct)
    low, high = bootstrap_auroc(uncertainty, correct)
    counts = Counter(reply.strip() for reply in replies)
    print(f"{label}: AUROC {scores['auroc']:.3f} [{low:.3f}, {high:.3f}], {len(counts)} distinct replies")
    print(f"  most common: {counts.most_common(5)}")


def main() -> None:
    """Run the anchored prompt on one model and compare it with the saved week 1 result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="qwen1.5b", choices=sorted(MODELS))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    rows = load_data(args.run, args.n)
    correct = [row["correct"] for row in rows]
    folder = RESULTS_ROOT / args.run / f"triviaqa_{args.n}"

    old = json.loads((folder / "verbalized_2s.json").read_text())
    summarize("before (0-100)", old["replies"], old["uncertainty"], correct)

    model, tok = load_model(MODELS[args.run], model_kwargs={"dtype": getattr(torch, args.dtype)})
    uncertainty, failed, replies = [], [], []
    start = time.perf_counter()
    for row in rows:
        confidence, no_number, reply = verbalized_2s(
            model, tok, row["question"], row["greedy_answer"], scale=SCALE, ask=ASK_ANCHORED
        )
        uncertainty.append(1.0 if no_number else 1.0 - confidence)
        failed.append(no_number)
        replies.append(reply)
    seconds = (time.perf_counter() - start) / len(rows)
    summarize("after (anchored 0-10)", replies, uncertainty, correct)
    print(f"unreadable replies: {sum(failed)}, {seconds:.2f} s per question")

    out = folder / "c4" / "verbalized_2s_anchored.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "method": "verbalized_2s_anchored",
        "model": MODELS[args.run],
        "answer": "greedy",
        "probability": True,
        "scale": SCALE,
        "prompt": ASK_ANCHORED,
        "replies": replies,
        "ids": [row["id"] for row in rows],
        "uncertainty": uncertainty,
        "failed": failed,
        "seconds_per_question": seconds,
        "model_calls_per_question": 1,
    }
    out.write_text(json.dumps(record))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
