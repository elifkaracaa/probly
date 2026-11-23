.. _methods-semantic:

===============================================
Semantic Uncertainty for Language Models
===============================================

.. currentmodule:: probly.representer.semantic_clustering.huggingface

A language model that is asked the same question several times does not
return the same string twice, even when it is sure of the answer: "Paris",
"It is Paris." and "Paris, France." differ as text but not in meaning.
Uncertainty over *strings* therefore overstates how unsure the model is. The
methods on this page measure uncertainty over *meanings* instead. They sample
several answers, decide which answers mean the same thing, and measure how
much the meanings disagree. A model that knows the answer produces one
meaning in many wordings; a model that guesses spreads its samples over
several meanings.

All of them need only the sampled texts (and, for semantic entropy, their
likelihoods), so they also work for models behind an API. They share one
expensive step: comparing the answers pairwise. In ``probly`` this step is a
*semantic clusterer*, a representer that turns a
:class:`~probly.representation.text_generation.torch.TorchTextGeneration` into
a sparse categorical distribution over semantic clusters. Every score below
reads that one representation, so computing several of them costs no more
than computing one.

.. _m-semantic-clustering:

Semantic clustering
-------------------

:class:`HFGreedySemanticClusterer` runs a natural language inference (NLI)
model on every ordered pair of answers and puts two answers into the same
cluster when neither contradicts the other and they are not both neutral.
Identical answers are compared once, and the pairs of several questions share
model batches. :meth:`~HFSemanticClusterer.pairwise_nli` returns the full
matrix of NLI probabilities, which the graph measures below read directly.

:class:`HFEmbeddingSemanticClusterer` is a cheaper drop-in with the same
interface. It embeds every answer once and compares answers by the cosine
similarity of their sentence embeddings; two answers are equivalent when the
similarity is above a threshold (0.9 by default). It replaces a forward pass
per *pair* by one per *answer*, which made it 26-33x faster than
``microsoft/deberta-large-mnli`` in our TriviaQA experiments, at the cost of
a coarser notion of meaning (see :ref:`m-semantic-results`).

:Representation: :class:`~probly.representation.distribution.torch_sparse_log_categorical.TorchSparseLogCategoricalDistribution`
    whose ``group_ids`` give the cluster of every answer and whose
    ``entry_logits`` are the answers' log-likelihoods.
:Advantages: Text only; one clustering serves every score on this page; the
    embedding backend makes it cheap enough for large runs.
:Disadvantages: The NLI step scores every ordered pair, about ``K²`` model
    calls for ``K`` answers (the embedding backend needs ``K``); whether two answers "mean the same" is only as good as the NLI
    or embedding model, and short answers ("1969") can be hard to judge
    without the question.

.. _m-semantic-entropy:

Semantic entropy
----------------

The entropy of the distribution over semantic clusters, where each cluster's
probability is the summed likelihood of its answers. *Discrete* semantic
entropy weights every answer equally instead, so it needs no likelihoods at
all and also works for black-box models.

:Idea: Entropy over meanings rather than over strings.
:Representation: A scalar per question, from
    :func:`probly.quantification.entropy` of the clustered distribution (or of
    its ``uniform_logits()`` for the discrete variant).
:Advantages: Simple, well studied, and one of the strongest black-box scores
    for detecting wrong answers.
:Disadvantages: Needs several samples per question (10 in the original
    work); the likelihood-weighted variant needs token log-probabilities.
:Reference: :cite:`kuhnSemanticUncertainty2023`, :cite:`farquharDetectingHallucinations2024`

.. _m-semantic-graph:

Graph measures
--------------

Instead of committing to hard clusters, the answers can be read as a weighted
graph whose edge weights are the mean entailment probability of the two
directions. Four measures summarise how connected that graph is: the
**number of semantic sets** (the cluster count), **EigValLaplacian** (the
sum of ``max(0, 1 - λ)`` over the eigenvalues of the normalized graph
Laplacian, a soft cluster count), **DegMat** (one minus the mean degree,
i.e. how weakly the answers support each other on average) and
**Eccentricity** (how far the answers lie from their centre in the
Laplacian's spectral embedding).

:Idea: Uncertainty as the lack of connectivity in the graph of pairwise
    semantic similarities.
:Representation: A scalar per question, computed from the pairwise NLI
    matrix of :meth:`~HFSemanticClusterer.pairwise_nli`.
:Advantages: No likelihoods needed; soft measures see *how strongly* answers
    agree, not only whether, so paraphrases with lower entailment
    probabilities still count.
:Disadvantages: Same sampling and NLI cost as semantic entropy; the scores
    have no probabilistic interpretation.
:Reference: :cite:`linGeneratingConfidence2024`

.. _m-semantic-decomposition:

Decomposition by input clarification
------------------------------------

Semantic entropy cannot tell whether the answers disagree because the
*question* is ambiguous ("Who won the World Cup?") or because the *model*
does not know. Clarification ensembling separates the two: a
:class:`~probly.representer.clarifier.huggingface.HFQuestionClarifier`
rephrases the question several times, the model answers each rephrasing
several times, all answers are clustered together, and the entropy is split
with :class:`~probly.quantification.SecondOrderEntropyDecomposition` over the
rephrasings. Here the members of the ensemble are readings of the question,
not models, so the *mutual information* measures how much the answer depends
on the reading (ambiguity of the input, called aleatoric by the authors) and
the *mean conditional entropy* measures disagreement within one reading (the
model does not know, called epistemic). Note that these names are the other
way round from the usual model-ensemble reading of the same two quantities.

:Idea: An ensemble over inputs instead of models: total = ambiguity of the
    question + uncertainty of the model.
:Representation: A ``Sample`` of sparse categorical distributions over the
    clarification axis, decomposed into two scalars per question.
:Advantages: Explains *why* an answer is uncertain, which a single score
    cannot.
:Disadvantages: ``1 + C + C · K`` generations per question (19 for
    ``C = 3`` clarifications with ``K = 5`` answers each) plus the NLI pass; the
    clarifier must not change the meaning of the question, which small
    models sometimes do.
:Reference: :cite:`houDecomposingUncertainty2024`

.. _m-semantic-example:

Example
-------

The snippet clusters five answers to one question with an NLI model and with
sentence embeddings, and computes semantic entropy and its discrete variant.
It runs on a laptop CPU; the first run downloads the two models (about
1.6 GB and 0.4 GB).

.. code-block:: python

    import numpy as np
    import torch

    from probly.quantification import entropy
    from probly.representation.text_generation.torch import TorchTextGeneration
    from probly.representer.semantic_clustering.huggingface import (
        HFEmbeddingSemanticClusterer,
        HFGreedySemanticClusterer,
    )

    # Five sampled answers to "What is the capital of France?" (one question, axis 1 = samples)
    # and the mean token log-likelihood of each answer.
    answers = np.array([["Paris", "It is Paris.", "Paris, France.", "Lyon", "Paris"]], dtype=object)
    generation = TorchTextGeneration(
        text=answers,
        log_likelihood=torch.tensor([[-0.1, -0.4, -0.5, -1.2, -0.1]]),
    )

    nli = HFGreedySemanticClusterer.from_model_name("microsoft/deberta-large-mnli")
    semantic = nli(generation, axis=1)
    print(semantic.group_ids)  # cluster of every answer, e.g. tensor([[0, 0, 0, 1, 0]])
    print(entropy(semantic.to_dense()))  # semantic entropy (nats)
    print(entropy(semantic.uniform_logits().to_dense()))  # discrete semantic entropy

    cheap = HFEmbeddingSemanticClusterer.from_model_name()  # all-mpnet-base-v2, threshold 0.9
    print(cheap(generation, axis=1).group_ids)

    matrix = nli.pairwise_nli(["Paris", "It is Paris.", "Lyon"])  # (3, 3, 3): contradiction, neutral, entailment

The experiments in ``experiments/llm-probly`` wrap these calls in a
one-call helper, ``common.semantic_scores.semantic_scores``, that returns all
seven scores of this page for a list of answers and checks the inputs first;
see ``experiments/llm-probly/docs/semantic_methods.md``.

.. _m-semantic-results:

Results on TriviaQA
-------------------

AUROC for detecting a wrong greedy answer on 200 TriviaQA questions, 10
samples per question, NLI model ``microsoft/deberta-large-mnli``. The
embedding threshold was chosen on Qwen2.5-0.5B; Qwen2.5-1.5B is held out.
With 200 questions the 95% interval is about ±0.05, so the differences
within a row are not significant.

.. list-table::
    :header-rows: 1

    * - Model, backend
      - SE
      - Discrete SE
      - EigValLaplacian
      - DegMat
      - NumSemSets
    * - Qwen2.5-0.5B, NLI
      - 0.744
      - 0.751
      - 0.758
      - 0.751
      - 0.759
    * - Qwen2.5-0.5B, embedding
      - 0.745
      - 0.755
      - 0.774
      - 0.762
      - 0.763
    * - Qwen2.5-1.5B, NLI
      - 0.776
      - 0.779
      - 0.784
      - 0.779
      - 0.775
    * - Qwen2.5-1.5B, embedding (held out)
      - 0.744
      - 0.749
      - 0.752
      - 0.751
      - 0.748

Five samples instead of ten cost about 0.01 AUROC; two or three samples cost
clearly more. The clarification decomposition did not detect wrong answers
better than semantic entropy (on 50 questions of Qwen2.5-0.5B: 0.71 for the
total against 0.86 for semantic entropy), but it shows which answers are
uncertain because of the question.
