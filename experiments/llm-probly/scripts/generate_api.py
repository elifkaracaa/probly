r"""Generate a text-only cache from a model behind an OpenAI-compatible API (F2).

Takes the questions and aliases from an existing cache (so the API run uses
exactly the same questions as the local runs), asks the API model for a greedy
answer and K sampled answers, labels them with ``common.correctness`` and
writes rows in the usual cache format to ``data/<model-key>/<source name>``.

The API gives no token log-probabilities, so every log-likelihood field is
``null`` and rows carry ``"access": "text"``. Cache-only text methods
(``lexical_similarity``, ``discrete_semantic_entropy``) and the ``*_api``
plugins run on these rows; logit methods (MSP, perplexity, ...) do not apply.

Example (Ollama on a laptop, run from experiments/llm-probly)::

    python -m scripts.generate_api --model-key ollama-qwen0.5b \
        --api-model qwen2.5:0.5b --base-url http://localhost:11434/v1 \
        --source data/qwen0.5b/triviaqa_200.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python scripts/generate_api.py` as well as `-m`
    sys.path.insert(0, str(ROOT))

from common.correctness import is_correct, is_correct_letter  # noqa: E402
from common.openai_client import OpenAICompatibleClient  # noqa: E402

NULL_LOGPROB_FIELDS = (
    "greedy_ll_sum",
    "greedy_ll_mean",
    "greedy_token_logprobs",
    "greedy_token_entropies",
    "samples_ll_mean",
    "samples_ll_sum",
)


def label(answer: str, row: dict) -> bool:
    """Label one answer with the same correctness rule as the local caches.

    Args:
        answer: Model answer.
        row: Source row with ``aliases`` and ``dataset``.

    Returns:
        Whether the answer is correct.
    """
    check = is_correct_letter if row.get("dataset") == "mmlu" else is_correct
    return check(answer, row["aliases"])


def generate_row(
    client: OpenAICompatibleClient,
    source: dict,
    *,
    num_samples: int,
    temperature: float,
    max_tokens: int,
    seed: int,
    system: str | None,
) -> dict:
    """Generate the greedy answer and samples for one source row.

    Args:
        client: API client.
        source: Row with at least ``id``, ``question`` and ``aliases``.
        num_samples: Number of sampled answers.
        temperature: Sampling temperature for the samples.
        max_tokens: Maximum new tokens per answer.
        seed: Base seed for this question.
        system: Optional system prompt.

    Returns:
        A cache row in the usual format, with ``null`` log-likelihood fields.
    """
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": source["question"]})
    greedy = client.chat(messages, temperature=0.0, max_tokens=max_tokens, seed=seed)[0]
    samples = client.chat(
        messages,
        n=num_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    correct = label(greedy, source)
    row = {
        "id": source["id"],
        "dataset": source.get("dataset"),
        "question": source["question"],
        "aliases": source["aliases"],
        "greedy_answer": greedy,
        "samples": samples,
        "model": f"api:{client.model}",
        "access": "text",
        "api": {"model": client.model, "base_url": client.base_url},
        "correct_alias": correct,
        "correct": correct,
        "samples_correct": [label(s, source) for s in samples],
    }
    row |= dict.fromkeys(NULL_LOGPROB_FIELDS)
    return row


def main(argv: list[str] | None = None) -> Path:
    """Command-line entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv``).

    Returns:
        Path of the written cache.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-key",
        required=True,
        help="folder name under data/, e.g. ollama-qwen0.5b",
    )
    parser.add_argument("--api-model", required=True, help="model name on the server, e.g. qwen2.5:0.5b")
    parser.add_argument("--base-url", default=None, help="API root incl. /v1 (default $OPENAI_BASE_URL)")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="existing cache to take questions from",
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "data")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--system",
        default=None,
        help="optional system prompt (default: none, question only)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep the finished rows of an interrupted run and generate only the rest",
    )
    args = parser.parse_args(argv)

    sources = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    if args.limit:
        sources = sources[: args.limit]
    client = OpenAICompatibleClient(model=args.api_model, base_url=args.base_url)
    out = args.out_root / args.model_key / args.source.name
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set = set()
    if args.resume and out.exists():
        # Keep only complete lines; a line cut off by a crash is dropped and redone.
        kept = []
        for line in out.read_text().splitlines():
            try:
                kept.append(json.loads(line))
            except json.JSONDecodeError:
                break
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
        done = {r["id"] for r in kept}
        print(
            f"resuming: {len(done)} of {len(sources)} questions already done",
            flush=True,
        )

    start = time.perf_counter()
    generated = 0
    with out.open("a" if args.resume else "w") as fh:
        for i, source in enumerate(sources):
            if source["id"] in done:
                continue
            row = generate_row(
                client,
                source,
                num_samples=args.num_samples,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=args.seed + i * (args.num_samples + 1),
                system=args.system,
            )
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            generated += 1
            print(
                f"[{i + 1}/{len(sources)}] {row['greedy_answer']!r} correct={row['correct']}",
                flush=True,
            )

    seconds = time.perf_counter() - start
    meta = {
        "model": f"api:{args.api_model}",
        "access": "text",
        "base_url": client.base_url,
        "source": str(args.source),
        "n": len(sources),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "system": args.system,
        "accuracy": sum(json.loads(r)["correct"] for r in out.read_text().splitlines()) / max(len(sources), 1),
        # Cost is measured on the questions generated in this run only (a resumed
        # run did not see the cost of the rows it kept).
        "questions_generated_this_run": generated,
        "seconds_per_question": seconds / generated if generated else None,
        "requests_per_question": client.usage.requests / generated if generated else None,
        "tokens_per_question": (client.usage.prompt_tokens + client.usage.completion_tokens) / generated
        if generated
        else None,
        "usage": client.usage.snapshot(),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return out


if __name__ == "__main__":
    main()
