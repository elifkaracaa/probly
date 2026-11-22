"""Self-evaluation (p(True)) scoring with Hugging Face causal language models.

The model is shown a question and a possible answer and is asked whether the answer is true. The
probability it puts on the "True" option, normalized against the "False" option, is its confidence in
the answer. For the uncertainty score of an experiment use ``1 - p_true``.

Two prompt modes exist:

* ``"plain_ab"``: no chat template, the prompt ends with `` (`` and the next token is compared between
  ``A`` and ``B``.
* ``"chat_groups"``: chat template, the next token is compared between the token groups
  ``True`` / ``A`` / ``(A`` and ``False`` / ``B`` / ``(B`` (only strings that are a single token).

Prompts of a batch are padded on the left and the position ids are computed from the attention mask,
so a prompt gets the same scores in a batch as it would alone.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from probly.representer.sampler.huggingface import HFTextGenerationSampler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

type SelfEvaluationMode = Literal["plain_ab", "chat_groups"]

DEFAULT_MODE: SelfEvaluationMode = "plain_ab"

TOKEN_GROUPS: dict[str, tuple[list[str], list[str]]] = {
    "chat_groups": (["True", " True", "A", " A", "(A"], ["False", " False", "B", " B", "(B"]),
    "plain_ab": (["A"], ["B"]),
}


@dataclass(frozen=True)
class SelfEvaluation:
    """Result of evaluating a batch of answers.

    Attributes:
        p_true: Confidence in each answer in ``[0, 1]``, one value per question.
        coverage: Probability mass the model puts on the two answer options, one value per question.
            Values far below one mean that the model does not answer with the expected option.
        mode: Prompt mode that was used.
    """

    p_true: np.ndarray
    coverage: np.ndarray
    mode: str


def build_self_evaluation_text(question: str, answer: str, brainstorm: Sequence[str] | None = None) -> str:
    """Build the multiple-choice prompt text that asks whether an answer is true.

    Args:
        question: The question.
        answer: The possible answer that is evaluated.
        brainstorm: Optional other answers of the model, shown as brainstormed ideas.

    Returns:
        The prompt text, ending with ``The possible answer is:``.
    """
    ideas = f"Here are some ideas that were brainstormed: {', '.join(brainstorm)}\n" if brainstorm else ""
    return (
        f"Question: {question}\n"
        f"{ideas}"
        f"Possible answer: {answer}\n"
        "Is the possible answer:\n"
        " (A) True\n"
        " (B) False\n"
        "The possible answer is:"
    )


def parse_judgment(text: str) -> bool | None:
    """Read a sampled reply to the multiple-choice prompt as a judgment.

    Leading markdown, quotes and brackets are ignored. A reply counts as ``True`` if it starts with
    ``true`` or with the option ``A`` and as ``False`` if it starts with ``false`` or the option ``B``.

    Args:
        text: The decoded reply of the model.

    Returns:
        ``True``, ``False``, or ``None`` if the reply can be read as neither.
    """
    cleaned = re.sub(r"^[^0-9a-z]+", "", text.strip().lower())
    if cleaned.startswith("true"):
        return True
    if cleaned.startswith("false"):
        return False
    option = re.match(r"^([ab])(?![0-9a-z_])", cleaned)
    if option is None:
        return None
    return option.group(1) == "a"


def _single_token_ids(tokenizer: PreTrainedTokenizerBase, strings: Sequence[str]) -> list[int]:
    """Return the token ids of the strings that are exactly one token for this tokenizer."""
    ids: set[int] = set()
    for string in strings:
        encoded = tokenizer.encode(string, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(encoded[0])
    return sorted(ids)


class HFSelfEvaluator:
    """Score how much a Hugging Face causal language model believes its own answers.

    Attributes:
        model: The causal language model.
        tokenizer: A copy of the tokenizer, configured for left padding.
        mode: Prompt mode, ``"plain_ab"`` or ``"chat_groups"``.
        batch_size: Number of prompts that go through the model at once.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        mode: SelfEvaluationMode = DEFAULT_MODE,
        batch_size: int = 8,
    ) -> None:
        """Initialize the evaluator.

        Args:
            model: A transformers causal language model (decoder only).
            tokenizer: The tokenizer of ``model``. It is copied, so the original keeps its padding side.
            mode: Prompt mode, ``"plain_ab"`` or ``"chat_groups"``.
            batch_size: Number of prompts that go through the model at once.

        Raises:
            ValueError: If the mode is unknown, ``batch_size`` is below one, or the tokenizer has no
                single token for one of the two answer options.
        """
        if mode not in TOKEN_GROUPS:
            msg = f"Unknown mode {mode!r}, expected one of {sorted(TOKEN_GROUPS)}."
            raise ValueError(msg)
        if batch_size < 1:
            msg = f"batch_size must be at least 1, got {batch_size}."
            raise ValueError(msg)
        self.model = model
        self.mode = mode
        self.batch_size = batch_size

        self.tokenizer = copy(tokenizer)
        self.tokenizer.padding_side = "left"
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        true_strings, false_strings = TOKEN_GROUPS[mode]
        self._true_ids = _single_token_ids(self.tokenizer, true_strings)
        self._false_ids = _single_token_ids(self.tokenizer, false_strings)
        if not self._true_ids or not self._false_ids:
            msg = (
                f"Empty token group for mode {mode!r} (true={self._true_ids}, false={self._false_ids}); "
                "check the tokenizer."
            )
            raise ValueError(msg)

    def _build_prompt(self, question: str, answer: str, brainstorm: Sequence[str] | None) -> str:
        text = build_self_evaluation_text(question, answer, brainstorm)
        if self.mode == "chat_groups":
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if not isinstance(prompt, str):
                msg = f"Expected the chat template to return a string, got {type(prompt)}."
                raise TypeError(msg)
            return prompt
        return text + " ("

    @torch.no_grad()
    def _next_token_logits(self, prompts: Sequence[str]) -> torch.Tensor:
        """Return the logits of the token after each prompt, shape ``(len(prompts), vocab)``."""
        # The chat template already contains its special tokens, so none are added again.
        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=self.mode != "chat_groups",
        )
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        attention_mask = encoded["attention_mask"]
        # With left padding the real tokens start later in the row; position ids must start at zero there.
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        output = self.model(input_ids=encoded["input_ids"], attention_mask=attention_mask, position_ids=position_ids)
        return output.logits[:, -1].float()

    def evaluate(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        brainstorm: Sequence[Sequence[str] | None] | None = None,
    ) -> SelfEvaluation:
        """Compute p(True) and coverage for every question and answer pair.

        Args:
            questions: The questions.
            answers: One possible answer per question.
            brainstorm: Optional other answers per question (``None`` entries mean no brainstorm line).

        Returns:
            A :class:`SelfEvaluation` with one ``p_true`` and one ``coverage`` value per question.

        Raises:
            ValueError: If the inputs do not have the same length.
        """
        if len(questions) != len(answers):
            msg = f"Got {len(questions)} questions but {len(answers)} answers."
            raise ValueError(msg)
        if brainstorm is not None and len(brainstorm) != len(questions):
            msg = f"Got {len(questions)} questions but {len(brainstorm)} brainstorm lists."
            raise ValueError(msg)
        ideas = brainstorm if brainstorm is not None else [None] * len(questions)

        prompts = [
            self._build_prompt(question, answer, idea)
            for question, answer, idea in zip(questions, answers, ideas, strict=True)
        ]
        # Prompts of similar length share a batch, so that little compute is spent on padding.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        p_true = np.empty(len(prompts), dtype=float)
        coverage = np.empty(len(prompts), dtype=float)
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            log_probs = torch.log_softmax(self._next_token_logits([prompts[i] for i in indices]), dim=-1)
            log_true = torch.logsumexp(log_probs[:, self._true_ids], dim=-1)
            log_false = torch.logsumexp(log_probs[:, self._false_ids], dim=-1)
            # sigmoid(lt - lf) = P(T) / (P(T) + P(F))
            p_true[indices] = torch.sigmoid(log_true - log_false).cpu().numpy()
            coverage[indices] = (log_true.exp() + log_false.exp()).cpu().numpy()

        return SelfEvaluation(p_true=p_true, coverage=coverage, mode=self.mode)

    def sample_judgments(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        brainstorm: Sequence[Sequence[str] | None] | None = None,
        num_samples: int = 10,
        temperature: float = 1.0,
        max_new_tokens: int = 3,
        question_batch_size: int | None = None,
    ) -> list[list[str]]:
        """Sample several short judgments per question and decode them.

        The replies of a batch of questions come from one ``generate`` call (``num_return_sequences``), so the
        cost does not grow with one call per question. Read the replies with :func:`parse_judgment`; the share
        of ``True`` replies is the empirical p(True).

        Args:
            questions: The questions.
            answers: One possible answer per question.
            brainstorm: Optional other answers per question (``None`` entries mean no brainstorm line).
            num_samples: Number of replies per question.
            temperature: Sampling temperature.
            max_new_tokens: Maximum number of generated tokens per reply.
            question_batch_size: Number of questions per ``generate`` call. ``None`` uses ``batch_size``.

        Returns:
            For every question, the list of ``num_samples`` decoded replies.

        Raises:
            ValueError: If the inputs do not have the same length or ``question_batch_size`` is below one.
        """
        if len(questions) != len(answers):
            msg = f"Got {len(questions)} questions but {len(answers)} answers."
            raise ValueError(msg)
        if brainstorm is not None and len(brainstorm) != len(questions):
            msg = f"Got {len(questions)} questions but {len(brainstorm)} brainstorm lists."
            raise ValueError(msg)
        size = self.batch_size if question_batch_size is None else question_batch_size
        if size < 1:
            msg = f"question_batch_size must be at least 1, got {size}."
            raise ValueError(msg)
        ideas = brainstorm if brainstorm is not None else [None] * len(questions)

        prompts = [
            build_self_evaluation_text(question, answer, idea)
            for question, answer, idea in zip(questions, answers, ideas, strict=True)
        ]
        sampler = HFTextGenerationSampler(
            self.model,
            self.tokenizer,
            num_samples=num_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            with_log_likelihood=False,
        )
        # Prompts of similar length share a batch, so that little compute is spent on padding.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        replies: list[list[str]] = [[] for _ in prompts]
        for start in range(0, len(order), size):
            indices = order[start : start + size]
            sample = sampler([prompts[i] for i in indices])
            for index, row in zip(indices, sample.tensor.text, strict=True):
                replies[index] = [str(text) for text in row]
        return replies
