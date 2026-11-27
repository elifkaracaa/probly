"""Pick and calibrate an uncertainty score for an LLM answer (skeleton: generate and score)."""

from __future__ import annotations

from collections import Counter
import logging
import os
import re

from probly.representer.sampler.huggingface import HFTextGenerationSampler, load_model

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"  # small enough for a laptop CPU
NUM_ANSWERS = 5
INSTRUCTION = "Answer the following question with as few words as possible."

QUESTIONS: list[str] = [
    "What is the capital of France?",
    "What is the chemical symbol for water?",
    "Who was the first person to walk on Mars?",
]


def normalize(answer: str) -> str:
    """Lower-case an answer and drop punctuation and extra spaces, so equal answers compare equal."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", answer.lower())).strip()


def disagreement(answers: list[str]) -> float:
    """Return the share of sampled answers that differ from the most common answer.

    0.0 means all samples agree (the model is consistent), values near 1.0 mean
    the samples are all different.

    Args:
        answers: Sampled answers to one question.

    Returns:
        A score in [0, 1]; higher means more uncertain.
    """
    counts = Counter(normalize(answer) for answer in answers)
    return 1.0 - counts.most_common(1)[0][1] / len(answers)


def main() -> None:
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Step 1: generate. Sample several answers per question.
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model(MODEL_NAME)
    model.eval()
    sampler = HFTextGenerationSampler(
        model,
        tokenizer,
        num_samples=NUM_ANSWERS,
        temperature=0.7,
        max_new_tokens=16,
    )
    prompts = [f"{INSTRUCTION}\nQuestion: {question}" for question in QUESTIONS]
    print(f"Sampling {NUM_ANSWERS} answers for {len(QUESTIONS)} questions...\n")
    samples = sampler(prompts)
    texts = samples.tensor.text  # shape (question, answer)
    log_likelihoods = samples.tensor.log_likelihood  # same shape, one value per answer

    # Step 2: methods. Two simple scores, both read from the same samples (higher = more uncertain).
    #   - likelihood: minus the mean log likelihood the model gave its own answers
    #   - disagreement: share of samples that differ from the most common answer
    print(f"{'Question':<42} {'Answer':<22} {'Likelihood':>10} {'Disagreement':>12}")
    print("-" * 90)
    for question, question_texts, question_log_likelihoods in zip(QUESTIONS, texts, log_likelihoods, strict=True):
        answers = [str(text).strip() for text in question_texts]
        likelihood = -sum(float(value.item()) for value in question_log_likelihoods) / len(answers)
        print(
            f"{question:<42} {answers[0][:20]:<22} {likelihood:>10.3f} {disagreement(answers):>12.3f}"
        )

    # Step 3: selector (placeholder). Filled in once the per-model profiles exist:
    #   profile = load_profile("profiles/<model>.json")
    #   -> which of the methods above works best for this model, and at what cost.

    # Step 4: calibrated confidence (placeholder). Filled in with the same profile:
    #   confidence = confidence(profile, uncertainty)
    #   -> a number in [0, 1] that says how likely the answer is correct.


if __name__ == "__main__":
    main()
