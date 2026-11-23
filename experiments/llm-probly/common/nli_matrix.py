"""Shared pairwise NLI matrix per question (A1).

One matrix per question holds the NLI probabilities for every ordered pair of
``[greedy_answer, sample_1, ..., sample_K]``. Semantic entropy, discrete semantic
entropy and the graph measures all read from it, so the NLI model runs once per
question instead of once per method.

The matrix itself comes from probly
(``HFSemanticClusterer.pairwise_nli``, last axis ordered contradiction, neutral,
entailment). This module only adds what the experiments need on top: which texts
go in, SE / discrete SE from a matrix, and the ``.nli.npz`` cache file format.

The cache is persistent (A5): the runner reads it before computing anything and writes new
matrices back, so a second run of SE / DSE on the same data does no NLI work. One file per
data cache and NLI setup, named by :func:`nli_cache_path`::

    data/<model>/<dataset>_<n>.nli.npz                  deberta-base, answers only (week-2 name)
    data/<model>/<dataset>_<n>.nli.<tag>.npz            any other setup, e.g. .nli.deberta-large.npz

A cached matrix is only reused if the file was made with the same NLI model and conditioning
and the question's fingerprint (:func:`matrix_fingerprint`: NLI model, question if used, and the
exact answer texts) is unchanged; otherwise that question is computed again and the file updated.

Cache file format (version 2; version 1 files load too, but have no fingerprints)::

    nli                   float32 (N, K+1, K+1, 3)  index 0 = greedy answer, NaN = diagonal/padding
    n_texts               int16   (N,)              1 + number of samples per question
    ids                   int64   (N,)              question ids, same order as the data cache
    labels                str     (3,)              order of the last axis of ``nli`` in the file
    nli_model             str     ()                Hugging Face name of the NLI model
    use_question          bool    ()                True if the question was put in front of each text
    deduplicate           bool    ()                True if identical texts were scored once (A2); missing = False
    seconds_per_question  float   ()                NLI time when the file was built
    fingerprints          str     (N,)              per question, see matrix_fingerprint (version 2)

The file stores the last axis as ``labels`` says (entail, neutral, contradict,
as in the week-2 files). :func:`load_nli_cache` always returns probly's order.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from probly.quantification import entropy
from probly.representation.distribution.torch_sparse_log_categorical import TorchSparseLogCategoricalDistribution
from probly.representer.semantic_clustering.huggingface import NLI_LABELS, HFGreedySemanticClusterer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from probly.representer.semantic_clustering.huggingface import HFSemanticClusterer

FORMAT_VERSION = 2
GREEDY_INDEX = 0
FILE_LABELS = ("entail", "neutral", "contradict")
"""Order of the last axis inside ``.nli.npz`` files (kept from week 2 so old readers keep working)."""


NLI_MODELS = {"deberta-base": "microsoft/deberta-base-mnli", "deberta-large": "microsoft/deberta-large-mnli"}
"""Short names of the NLI models compared in A4 (any Hugging Face NLI model name works too)."""
DEFAULT_NLI_MODEL = "deberta-large"
"""Default NLI model of the experiments (A4, see docs/methods.md "NLI setup").

On 80 hand-labelled answer pairs deberta-large agrees best with the human labels (0.971 vs 0.937 for
deberta-base, weighted), while SE / DSE AUROC stays the same within noise; it is about 3x slower, a
cost that is paid once per cache. probly's library default stays deberta-base.
"""


@dataclass(frozen=True)
class NliSettings:
    """Which NLI model the semantic methods use, and whether the question goes in front of each answer.

    Attributes:
        model: Short name from ``NLI_MODELS`` or a Hugging Face model name.
        use_question: Put the question in front of every answer before NLI (A4: no clear gain, 2x slower).
    """

    model: str = DEFAULT_NLI_MODEL
    use_question: bool = False

    @property
    def hf_name(self) -> str:
        """Hugging Face name of the NLI model."""
        return NLI_MODELS.get(self.model, self.model)

    @property
    def tag(self) -> str:
        """Short label for file names and records, e.g. ``deberta-large+question``."""
        name = next((short for short, hf in NLI_MODELS.items() if hf == self.hf_name), self.hf_name.split("/")[-1])
        return f"{name}+question" if self.use_question else name


_settings = NliSettings()


def nli_settings() -> NliSettings:
    """The NLI settings used by the semantic methods in this process."""
    return _settings


def set_nli_settings(model: str | None = None, *, use_question: bool | None = None) -> NliSettings:
    """Change the NLI settings for this process (the runner calls this with the config values).

    Args:
        model: Short name from ``NLI_MODELS`` or a Hugging Face name; None keeps the current one.
        use_question: Whether to put the question in front of each answer; None keeps the current value.

    Returns:
        The new settings.
    """
    global _settings  # noqa: PLW0603
    _settings = NliSettings(
        model=_settings.model if model is None else model,
        use_question=_settings.use_question if use_question is None else use_question,
    )
    return _settings


_cache_file: Path | None = None


def nli_cache_file() -> Path | None:
    """The persistent NLI cache the semantic methods read and update in this process (None = memory only)."""
    return _cache_file


def set_nli_cache_file(path: str | Path | None) -> None:
    """Set the persistent NLI cache file (the runner calls this, see :func:`nli_cache_path`).

    Args:
        path: The ``.nli*.npz`` file, or None to keep matrices in memory only.
    """
    global _cache_file  # noqa: PLW0603
    _cache_file = None if path is None else Path(path)


def nli_cache_path(data_file: str | Path, settings: NliSettings) -> Path:
    """Path of the NLI cache that belongs to a data cache and an NLI setup.

    Args:
        data_file: The data cache, e.g. ``data/qwen0.5b/triviaqa_200.jsonl`` (model and dataset are in the path).
        settings: The NLI setup; deberta-base on answers only keeps the week-2 name ``.nli.npz``.

    Returns:
        E.g. ``data/qwen0.5b/triviaqa_200.nli.deberta-large.npz``.
    """
    data_file = Path(data_file)
    tag = "" if settings.tag == "deberta-base" else f".{settings.tag}"
    return data_file.with_name(f"{data_file.stem}.nli{tag}.npz")


def question_texts(row: dict) -> list[str]:
    """Return the texts that go into the NLI matrix of one question.

    Args:
        row: One row of the data cache with ``greedy_answer`` and ``samples``.

    Returns:
        ``[greedy_answer, sample_1, ..., sample_K]``; index 0 is the greedy answer.
    """
    return [row["greedy_answer"], *row["samples"]]


QUESTIONS_PER_BATCH = 16
"""Questions whose NLI pairs share model batches (A3); a question has at most 110 pairs."""


def pick_device() -> str:
    """Return ``"cuda"``, ``"mps"`` or ``"cpu"``, whichever is available."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str) -> torch.dtype:
    """Map ``"float32"`` / ``"float16"`` to a torch dtype; float16 only on a GPU.

    Args:
        name: ``"float32"`` or ``"float16"``.
        device: ``"cuda"``, ``"mps"`` or ``"cpu"``.

    Returns:
        The torch dtype for the NLI model.

    Raises:
        ValueError: For an unknown name, or float16 on the CPU (slow and not supported by every op).
    """
    dtypes = {"float32": torch.float32, "float16": torch.float16}
    if name not in dtypes:
        msg = f"dtype must be one of {sorted(dtypes)}, got {name!r}."
        raise ValueError(msg)
    if dtypes[name] == torch.float16 and device == "cpu":
        msg = "float16 NLI needs a GPU (cuda or mps); use float32 on the CPU."
        raise ValueError(msg)
    return dtypes[name]


def matrix_fingerprint(row: dict, settings: NliSettings) -> str:
    """Short hash of everything a question's NLI matrix depends on.

    Args:
        row: One row of the data cache.
        settings: The NLI setup.

    Returns:
        16 hex characters of the SHA-256 of the NLI model name, the question (only if it is
        put in front of the answers) and the answer texts. Any change gives another fingerprint.
    """
    content = {
        "nli_model": settings.hf_name,
        "question": row["question"] if settings.use_question else None,
        "texts": question_texts(row),
    }
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()[:16]


def compute_matrices(
    clusterer: HFSemanticClusterer,
    rows: Sequence[dict],
    *,
    use_question: bool = False,
    deduplicate: bool = True,
) -> list[torch.Tensor]:
    """Compute the NLI matrices of several questions in shared model batches (A3).

    Args:
        clusterer: probly semantic clusterer that holds the NLI model.
        rows: Rows of the data cache.
        use_question: If True, the question is put in front of every text.
        deduplicate: If True, identical texts of a question are sent to the NLI model only once (A2).

    Returns:
        One ``(K+1, K+1, 3)`` tensor per row in probly's label order (``NLI_LABELS``).
    """
    contexts = [row["question"] if use_question else None for row in rows]
    groups = [question_texts(row) for row in rows]
    return clusterer.pairwise_nli_many(groups, contexts=contexts, deduplicate=deduplicate)


def compute_matrix(
    clusterer: HFSemanticClusterer,
    row: dict,
    *,
    use_question: bool = False,
    deduplicate: bool = True,
) -> torch.Tensor:
    """Compute the NLI matrix of one question; see :func:`compute_matrices`."""
    return compute_matrices(clusterer, [row], use_question=use_question, deduplicate=deduplicate)[0]


def nli_pair_count(row: dict, *, deduplicate: bool = True) -> int:
    """Number of ordered pairs the NLI model scores for one question.

    Args:
        row: One row of the data cache.
        deduplicate: Count distinct texts only, as :func:`compute_matrix` does with ``deduplicate=True``.

    Returns:
        ``m * (m - 1)`` with ``m`` the number of (distinct) texts of the question.
    """
    texts = question_texts(row)
    m = len(set(texts)) if deduplicate else len(texts)
    return m * (m - 1)


RULES = ("probly", "strict")
"""Equivalence rules: ``probly`` = semantic entropy's rule; ``strict`` = entailment in both directions."""


def same_meaning(matrix: torch.Tensor, a: int, b: int, rule: str = "probly") -> bool:
    """Whether texts ``a`` and ``b`` of a question matrix have the same meaning.

    With ``rule="probly"`` this is exactly the rule semantic entropy uses (via
    :meth:`HFGreedySemanticClusterer.cluster_nli_matrix` on the 2 x 2 block): no contradiction in
    either direction and not neutral in both directions. ``rule="strict"`` asks for entailment in
    both directions (bidirectional entailment), which never merges a vaguer answer with a specific one.

    Args:
        matrix: Question matrix of shape ``(n, n, 3)`` in probly's label order.
        a: Index of the first text.
        b: Index of the second text.
        rule: ``"probly"`` or ``"strict"``.

    Returns:
        True if the two texts are treated as having the same meaning.

    Raises:
        ValueError: For an unknown rule.
    """
    if rule == "probly":
        ids = HFGreedySemanticClusterer.cluster_nli_matrix(matrix[[a, b]][:, [a, b]])
        return bool(ids[0] == ids[1])
    if rule == "strict":
        entailment = NLI_LABELS.index("entailment")
        return bool(matrix[a, b].argmax() == entailment and matrix[b, a].argmax() == entailment)
    msg = f"rule must be one of {RULES}, got {rule!r}."
    raise ValueError(msg)


def sample_block(matrix: torch.Tensor) -> torch.Tensor:
    """Return the samples-vs-samples part of a question matrix (drops the greedy answer)."""
    return matrix[GREEDY_INDEX + 1 :, GREEDY_INDEX + 1 :]


def canonical_group_ids(group_ids: torch.Tensor) -> torch.Tensor:
    """Renumber clusters by size (largest first, ties by first appearance).

    Entropies are sums over clusters, and float sums depend on the order; with this numbering the
    same partition always gives the same order, so equal partitions give exactly equal entropies.

    Args:
        group_ids: Cluster id of every sample (ids ``0, 1, ...`` in order of first appearance).

    Returns:
        The same partition with ids ordered by cluster size.
    """
    counts = torch.bincount(group_ids)
    order = sorted(range(len(counts)), key=lambda cluster: (-int(counts[cluster]), cluster))
    new_id = torch.empty_like(counts)
    new_id[torch.tensor(order)] = torch.arange(len(order))
    return new_id[group_ids]


def se_and_dse(matrix: torch.Tensor, samples_ll_mean: Sequence[float]) -> tuple[float, float]:
    """Semantic entropy and discrete semantic entropy from one question matrix.

    Clusters the K samples with probly's greedy rule on the shared matrix, then
    computes the entropy of the cluster distribution weighted by the sample
    likelihoods (SE) and with equal weights (discrete SE), exactly as the
    clusterer path did before.

    The clusters are numbered by size first (see :func:`canonical_group_ids`), so the same partition
    always gives exactly the same float. Discrete SE has few distinct values (51 of 200 on qwen0.5b);
    with the clusters in order of appearance, equal partitions differed in the last float32 digit
    (2.16395545 vs 2.16395569) and broke those ties, and DSE AUROC moved between 0.741 and 0.749 under
    such noise. (Rounding does not fix this: the two values can round to different sides.)

    Args:
        matrix: Question matrix of shape ``(K+1, K+1, 3)`` in probly's label order.
        samples_ll_mean: Mean token log-likelihood of each of the K samples.

    Returns:
        ``(semantic_entropy, discrete_semantic_entropy)`` in nats.
    """
    samples = sample_block(matrix)
    if samples.shape[0] != len(samples_ll_mean):
        msg = f"matrix has {samples.shape[0]} samples but {len(samples_ll_mean)} likelihoods were given."
        raise ValueError(msg)
    group_ids = canonical_group_ids(HFGreedySemanticClusterer.cluster_nli_matrix(samples))
    semantic = TorchSparseLogCategoricalDistribution(
        group_ids=group_ids,
        entry_logits=torch.tensor(list(samples_ll_mean), dtype=torch.float32),
    )
    se = float(entropy(semantic.to_dense()).item())
    dse = float(entropy(semantic.uniform_logits().to_dense()).item())
    return se, dse


def _file_to_probly_order(labels: Sequence[str]) -> list[int]:
    """Column indices that turn the file's label order into probly's order."""
    names = [str(label).lower() for label in labels]
    columns = []
    for wanted in NLI_LABELS:
        matches = [i for i, name in enumerate(names) if name.startswith(wanted[:6])]
        if len(matches) != 1:
            msg = f"cannot find label {wanted!r} in cache labels {names}."
            raise ValueError(msg)
        columns.append(matches[0])
    return columns


def save_nli_cache(
    path: str | Path,
    matrices: Sequence[torch.Tensor],
    ids: Sequence[int],
    *,
    nli_model: str,
    use_question: bool,
    seconds_per_question: float,
    deduplicate: bool = True,
    fingerprints: Sequence[str] | None = None,
) -> None:
    """Save question matrices (probly order) as a ``.nli.npz`` file.

    Args:
        path: Output file.
        matrices: One ``(n_i, n_i, 3)`` matrix per question in probly's label order.
        ids: Question ids, same order as ``matrices``.
        nli_model: Name of the NLI model that produced the matrices.
        use_question: Whether the question was put in front of each text.
        seconds_per_question: Measured NLI time per question.
        deduplicate: Whether identical texts were scored once.
        fingerprints: :func:`matrix_fingerprint` of each question; without them the runner
            does not trust the file and computes the matrices again.
    """
    n_texts = np.array([m.shape[0] for m in matrices], dtype=np.int16)
    size = int(n_texts.max()) if len(matrices) else 0
    big = np.full((len(matrices), size, size, 3), np.nan, dtype=np.float32)
    to_file = [NLI_LABELS.index(name) for name in ("entailment", "neutral", "contradiction")]
    for i, matrix in enumerate(matrices):
        n = matrix.shape[0]
        big[i, :n, :n] = matrix.numpy()[..., to_file]
    np.savez_compressed(
        path,
        nli=big,
        n_texts=n_texts,
        ids=np.asarray(ids),
        labels=np.array(FILE_LABELS),
        nli_model=np.array(nli_model),
        use_question=np.array(use_question),
        seconds_per_question=np.array(seconds_per_question),
        deduplicate=np.array(deduplicate),
        format_version=np.array(FORMAT_VERSION),
        fingerprints=np.array([] if fingerprints is None else list(fingerprints), dtype=str),  # empty = unknown
    )


def load_nli_cache(path: str | Path) -> dict:
    """Load a ``.nli.npz`` file with the matrices in probly's label order.

    Args:
        path: The cache file (week-2 files without ``format_version`` load too).

    Returns:
        Dict with ``matrices`` (list of ``(n_i, n_i, 3)`` float32 tensors, probly order),
        ``ids``, ``nli_model``, ``use_question``, ``seconds_per_question``, ``deduplicate`` and
        ``fingerprints`` (None for files without them).
    """
    with np.load(path, allow_pickle=False) as z:
        columns = _file_to_probly_order(z["labels"].tolist())
        big = z["nli"][..., columns]
        n_texts = z["n_texts"]
        return {
            "matrices": [torch.from_numpy(big[i, :n, :n].copy()) for i, n in enumerate(n_texts)],
            "ids": z["ids"].tolist(),
            "nli_model": str(z["nli_model"]),
            "use_question": bool(z["use_question"]),
            "seconds_per_question": float(z["seconds_per_question"]),
            "deduplicate": bool(z["deduplicate"]) if "deduplicate" in z.files else False,
            "fingerprints": z["fingerprints"].tolist()
            if "fingerprints" in z.files and z["fingerprints"].size
            else None,
        }


def read_cached_matrices(path: str | Path, settings: NliSettings) -> dict[int, tuple[str, torch.Tensor]]:
    """Read the reusable entries of a persistent NLI cache.

    Args:
        path: The cache file; a missing file gives an empty dict.
        settings: The NLI setup of this run.

    Returns:
        ``{question id: (fingerprint, matrix)}``. Empty if the file was made with another NLI
        model or conditioning, or has no fingerprints (version 1): then nothing in it can be
        checked, so everything is computed again and the file is rewritten.
    """
    path = Path(path)
    if not path.exists():
        return {}
    cache = load_nli_cache(path)
    same_setup = (
        NLI_MODELS.get(cache["nli_model"], cache["nli_model"]) == settings.hf_name
        and cache["use_question"] == settings.use_question
    )
    if not same_setup:
        print(f"   NLI cache {path.name}: made with another NLI setup, computing again")
        return {}
    if cache["fingerprints"] is None:
        print(f"   NLI cache {path.name}: no fingerprints (old format), computing again")
        return {}
    return {
        int(qid): (fp, matrix)
        for qid, fp, matrix in zip(cache["ids"], cache["fingerprints"], cache["matrices"], strict=True)
    }


def cached_nli_seconds(path: str | Path) -> float:
    """NLI time per question stored in a cache file (0.0 if the file does not exist).

    Args:
        path: The cache file.

    Returns:
        The file's ``seconds_per_question``: what building its matrices cost.
    """
    path = Path(path)
    if not path.exists():
        return 0.0
    with np.load(path, allow_pickle=False) as z:
        return float(z["seconds_per_question"])


def count_cached_questions(path: str | Path, settings: NliSettings, rows: Sequence[dict]) -> int:
    """Number of ``rows`` whose NLI matrix the cache file can give without any NLI work.

    Args:
        path: The cache file.
        settings: The NLI setup of this run.
        rows: Rows of the data cache (with ``id``).

    Returns:
        How many rows have a matching fingerprint in the file.
    """
    entries = read_cached_matrices(path, settings)
    return sum(1 for row in rows if row["id"] in entries and entries[row["id"]][0] == matrix_fingerprint(row, settings))
