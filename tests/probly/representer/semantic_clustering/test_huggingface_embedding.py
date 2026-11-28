"""Tests for the sentence-embedding semantic clusterer."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")
import torch

from probly.representation.text_generation.torch import TorchTextGeneration
from probly.representer.embedder.huggingface import HFTextEmbedder
from probly.representer.semantic_clustering.huggingface import (
    NLI_LABELS,
    HFEmbeddingSemanticClusterer,
    HFGreedySemanticClusterer,
    cosine_to_nli_probabilities,
)

CONTRADICTION = NLI_LABELS.index("contradiction")
ENTAILMENT = NLI_LABELS.index("entailment")

VECTORS = {  # "Paris" and "It is Paris." have cosine 0.95; "Lyon" is orthogonal to both
    "Paris": [1.0, 0.0],
    "It is Paris.": [0.95, 0.312],
    "Lyon": [0.0, 1.0],
}


class FakeEmbeddingModel:
    """Sentence-transformer stand-in that looks the vectors up and records every call."""

    def __init__(self) -> None:
        """Initialize call recording."""
        self.calls: list[list[str]] = []

    def encode(self, sentences: list[str], **kwargs: object) -> torch.Tensor:  # noqa: ARG002
        """Return the vector of every sentence."""
        self.calls.append(list(sentences))
        return torch.tensor([VECTORS[s] for s in sentences], dtype=torch.float32)


def make_clusterer(threshold: float = 0.9) -> tuple[HFEmbeddingSemanticClusterer, FakeEmbeddingModel]:
    model = FakeEmbeddingModel()
    return HFEmbeddingSemanticClusterer(HFTextEmbedder(model), threshold=threshold), model


def test_cosine_to_nli_probabilities_switches_label_above_the_threshold():
    probabilities = cosine_to_nli_probabilities(torch.tensor([0.95, 0.9, 0.85, -0.3]), threshold=0.9)

    labels = probabilities.argmax(-1).tolist()
    assert labels == [ENTAILMENT, CONTRADICTION, CONTRADICTION, CONTRADICTION]  # equivalent only above t
    assert probabilities[1, ENTAILMENT] == pytest.approx(0.5)  # a tie at the threshold
    assert probabilities[3, ENTAILMENT] == 0.0
    assert torch.allclose(probabilities.sum(-1), torch.ones(4))


@pytest.mark.parametrize("threshold", [0.0, -0.5, 1.2])
def test_threshold_must_be_in_unit_interval(threshold):
    with pytest.raises(ValueError, match="threshold must be in"):
        HFEmbeddingSemanticClusterer(HFTextEmbedder(FakeEmbeddingModel()), threshold=threshold)


def test_pairwise_nli_has_the_nli_shape_and_embeds_each_distinct_text_once():
    clusterer, model = make_clusterer()

    matrix = clusterer.pairwise_nli(["Paris", "It is Paris.", "Lyon", "Paris"])

    assert matrix.shape == (4, 4, 3)
    assert torch.isnan(matrix.diagonal(dim1=0, dim2=1)).all()
    assert model.calls == [["Paris", "It is Paris.", "Lyon"]]
    assert torch.equal(HFGreedySemanticClusterer.cluster_nli_matrix(matrix), torch.tensor([0, 0, 1, 0]))


def test_pairwise_nli_many_shares_one_embedding_call():
    clusterer, model = make_clusterer()

    matrices = clusterer.pairwise_nli_many([["Paris", "Lyon"], ["It is Paris.", "Paris"]])

    assert [m.shape for m in matrices] == [(2, 2, 3), (2, 2, 3)]
    assert len(model.calls) == 1


def test_represent_clusters_like_the_nli_clusterer():
    clusterer, _ = make_clusterer()
    generation = TorchTextGeneration(
        text=np.asarray([["Paris", "It is Paris.", "Lyon"], ["Lyon", "Lyon", "Paris"]], dtype=object),
        log_likelihood=torch.zeros((2, 3)),
    )

    clustered = clusterer.represent(generation, axis=1)

    assert torch.equal(clustered.group_ids, torch.tensor([[0, 0, 1], [0, 0, 1]]))


def test_a_higher_threshold_splits_paraphrases():
    strict, _ = make_clusterer(threshold=0.99)

    matrix = strict.pairwise_nli(["Paris", "It is Paris."])

    assert torch.equal(HFGreedySemanticClusterer.cluster_nli_matrix(matrix), torch.tensor([0, 1]))
