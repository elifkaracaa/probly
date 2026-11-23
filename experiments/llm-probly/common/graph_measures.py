"""Graph measures of the sampled answers from the shared NLI matrix (B1).

The K sampled answers of a question are the nodes of a weighted graph. The weight
of an edge is how strongly two answers agree. When the model is sure, all
answers form one tightly connected group; when it is unsure, the graph falls
apart into several groups. The four measures of Lin et al. (2024, "Generating
with Confidence", https://arxiv.org/abs/2305.19187) turn this into a number,
following LM-Polygraph's estimators (``NumSemSets``, ``EigValLaplacian``,
``DegMat``, ``Eccentricity``; MIT-licensed like probly). Higher always means more
uncertain.

All four read the shared per-question NLI matrix (A1), so they need no extra NLI
call. Differences to LM-Polygraph:

- Number of semantic sets counts the clusters of probly's greedy clusterer
  (the clusters semantic entropy uses) instead of a second clustering rule.
- The diagonal of the similarity matrix (an answer compared with itself) is set
  to ``self_similarity`` (default 1.0) instead of being scored by the NLI model.
  On the week-2 caches the NLI model gives a median of 0.96 to identical texts
  (minimum 0.23), so 1.0 removes that noise. AUROC of DegMat and EigValLaplacian
  does not change between 1.0, 0.969 and 0.9 on 0.5B and 1.5B (at most 0.001).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from common.nli_matrix import sample_block
from probly.representer.semantic_clustering.huggingface import NLI_LABELS, HFGreedySemanticClusterer

if TYPE_CHECKING:
    import torch

ENTAILMENT = NLI_LABELS.index("entailment")
ECCENTRICITY_THRESHOLD = 0.9
"""Eigenvalues of the Laplacian below this value give the embedding of Eccentricity (LM-Polygraph default)."""


def similarity_matrix(matrix: torch.Tensor, self_similarity: float = 1.0) -> np.ndarray:
    """Symmetric similarity of the sampled answers from a question's NLI matrix.

    ``W[i, j]`` is the mean of the entailment probabilities in both directions
    (``i`` entails ``j`` and ``j`` entails ``i``), as LM-Polygraph's
    ``affinity="entail"``.

    Args:
        matrix: Question matrix ``(K+1, K+1, 3)`` in probly's label order;
            index 0 (the greedy answer) is dropped.
        self_similarity: Value put on the diagonal.

    Returns:
        Array of shape ``(K, K)``, symmetric, values in ``[0, 1]``.
    """
    entail = sample_block(matrix)[..., ENTAILMENT].numpy().astype(np.float64)
    similarity = (entail + entail.T) / 2
    np.fill_diagonal(similarity, self_similarity)
    return similarity


def normalized_laplacian(similarity: np.ndarray) -> np.ndarray:
    """Symmetric normalized graph Laplacian ``L = I - D^(-1/2) W D^(-1/2)``.

    ``D`` is the diagonal matrix of node degrees (row sums of ``W``). Its
    eigenvalues lie in ``[0, 2]``; the number of eigenvalues equal to 0 is the
    number of disconnected groups in the graph.

    Args:
        similarity: Symmetric similarity matrix ``(K, K)`` with positive row sums.

    Returns:
        Array of shape ``(K, K)``.
    """
    inv_sqrt_degree = 1.0 / np.sqrt(similarity.sum(axis=1))
    scaled = inv_sqrt_degree[:, None] * similarity * inv_sqrt_degree[None, :]
    return np.eye(len(similarity)) - scaled


def num_semantic_sets(group_ids: np.ndarray | torch.Tensor) -> float:
    """Number of semantic clusters among the sampled answers.

    Args:
        group_ids: Cluster id per sample, e.g. from
            ``HFGreedySemanticClusterer.cluster_nli_matrix``.

    Returns:
        Number of distinct clusters (1 = all answers agree).
    """
    return float(len(np.unique(np.asarray(group_ids))))


def eig_val_laplacian(similarity: np.ndarray) -> float:
    """Sum of ``max(0, 1 - eigenvalue)`` over the eigenvalues of the normalized Laplacian.

    A continuous version of the number of semantic sets: a graph of ``c``
    disconnected, fully agreeing groups gives exactly ``c``. The more the groups
    agree with each other, the lower the value (two equal groups with cross
    similarity ``s`` give ``1 + (1 - s) / (1 + s)``). It is not a count: a single
    weak link between two groups can even raise it slightly above 2.

    Args:
        similarity: Symmetric similarity matrix ``(K, K)``.

    Returns:
        Uncertainty in ``[1, K]``.
    """
    eigenvalues = np.linalg.eigvalsh(normalized_laplacian(similarity))
    return float(np.clip(1.0 - eigenvalues, 0.0, None).sum())


def deg_mat(similarity: np.ndarray) -> float:
    """One minus the mean node degree divided by ``K``.

    The degree of an answer is the sum of its similarities to all answers
    (itself included). If every answer agrees with every other one, each degree
    is ``K`` and the uncertainty is 0.

    Args:
        similarity: Symmetric similarity matrix ``(K, K)``.

    Returns:
        Uncertainty in ``[0, 1)``; equals LM-Polygraph's ``trace(K I - D) / K^2``.
    """
    k = len(similarity)
    return float(1.0 - similarity.sum() / k**2)


def eccentricity(similarity: np.ndarray, threshold: float = ECCENTRICITY_THRESHOLD) -> float:
    """Spread of the answers in the spectral embedding of the graph.

    The eigenvectors of the normalized Laplacian with eigenvalue below
    ``threshold`` give every answer a coordinate; similar answers get close
    coordinates. The uncertainty is the Frobenius norm of the coordinates after
    subtracting their mean, i.e. how far the answers lie from their centre.

    Args:
        similarity: Symmetric similarity matrix ``(K, K)``.
        threshold: Keep the eigenvectors whose eigenvalue is below this value.
            The smallest eigenvalue is always 0, so at least one is kept.

    Returns:
        Uncertainty, 0 when all answers get the same coordinate.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_laplacian(similarity))
    embedding = eigenvectors[:, eigenvalues < threshold]  # one row per answer
    centred = embedding - embedding.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(centred))


def graph_scores(matrix: torch.Tensor, self_similarity: float = 1.0) -> dict[str, float]:
    """All four graph measures of one question.

    Args:
        matrix: Question matrix ``(K+1, K+1, 3)`` in probly's label order.
        self_similarity: Diagonal of the similarity matrix, see :func:`similarity_matrix`.

    Returns:
        ``{"num_semantic_sets", "eig_val_laplacian", "deg_mat", "eccentricity"}``.
    """
    similarity = similarity_matrix(matrix, self_similarity)
    return {
        "num_semantic_sets": num_semantic_sets(HFGreedySemanticClusterer.cluster_nli_matrix(sample_block(matrix))),
        "eig_val_laplacian": eig_val_laplacian(similarity),
        "deg_mat": deg_mat(similarity),
        "eccentricity": eccentricity(similarity),
    }
