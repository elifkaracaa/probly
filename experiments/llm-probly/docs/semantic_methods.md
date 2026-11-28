# Semantic uncertainty methods

The semantic methods ask a model the same question several times and measure
how much the sampled answers **agree in meaning**. If the model knows the answer,
the samples say the same thing ("Paris", "It is Paris", "Paris, France"). If it
is guessing, they spread out ("Paris", "Lyon", "Marseille"). Every score below is
an **uncertainty**: higher means the samples agree less.

Two answers agree in meaning when an NLI model (the benchmark's setup,
`microsoft/deberta-large-mnli` by default, see `common.nli_matrix.nli_settings`)
finds that neither contradicts the other. The NLI model is run **once per
question** on all pairs of answers (one shared matrix), and every score reads
that matrix, so asking for all seven scores costs the same as asking for one.

## Quick start (laptop CPU)

Run from `experiments/llm-probly`:

```bash
uv run python -m scripts.semantic_demo
```

This is the whole script. It samples 10 answers from SmolLM2-135M and scores them:

```python
import logging

from common.generation import generate_answers, stop_at_newline_config
from common.semantic_scores import semantic_scores
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

logging.basicConfig(format="%(message)s")
logging.getLogger("common.semantic_scores").setLevel(logging.INFO)  # progress messages of the helper only
torch.manual_seed(0)

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = PreTrainedTokenizerFast.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

question = "What is the capital of France?"
prompt = f"Answer the following question with as few words as possible.\nQuestion: {question}"
samples = generate_answers(
    model,
    tok,
    prompt,
    num_return_sequences=10,
    do_sample=True,
    temperature=0.7,
    max_new_tokens=32,
    generation_config=stop_at_newline_config(model, tok),
)
samples = [s for s in samples if s["text"]]  # tiny models sometimes return an empty answer

scores = semantic_scores(
    [s["text"] for s in samples],
    sample_log_likelihoods=[s["ll_mean"] for s in samples],
)
print("samples: ", [s["text"] for s in samples])
print("clusters:", scores["clusters"])
for name in ("semantic_entropy", "discrete_semantic_entropy", "num_semantic_sets", "eig_val_laplacian"):
    print(f"{name:>26}: {scores[name]:.3f}")
```

The first run downloads SmolLM2-135M-Instruct (~270 MB) and the NLI model (~1.6 GB).
Output of a run on a MacBook (Apple silicon) with both models already downloaded
(SmolLM on the CPU, the NLI model on the GPU if there is one):

```text
loading the NLI model microsoft/deberta-large-mnli (first call only)
NLI model ready in 2.6 s
scored 1/1 questions (0.51 s per question, about 0 s left)
samples:  ['The capital of France is Paris.', 'The capital of France is Paris.', ...]
clusters: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
          semantic_entropy: 0.000
 discrete_semantic_entropy: 0.000
         num_semantic_sets: 1.000
         eig_val_laplacian: 1.290
```

All 10 answers fall into one cluster, so the cluster-based scores are 0 (or 1 for the
cluster count). `eig_val_laplacian` is a bit above 1 because the NLI model gives
paraphrases ("Paris." vs "Paris, a city known for ...") an entailment probability
below 1: the graph scores see *how strongly* the answers agree, not only whether.

If you load the NLI model yourself, transformers may print a long red traceback ending in
`KeyError: 'event_id'` while it loads. It is a background attempt to convert the model file to
safetensors and does not affect the scores; the helper switches it off
(`DISABLE_SAFETENSORS_CONVERSION=1`), so you can also set that variable yourself.

Already have answers from another model or an API? Skip the generation:

```python
from common.semantic_scores import semantic_scores

scores = semantic_scores(["Paris", "Paris", "It is Paris.", "Lyon"])
scores["num_semantic_sets"]  # e.g. 2.0 if the NLI model groups {Paris, Paris, It is Paris.} and {Lyon}
scores["semantic_entropy"]  # nan: needs sample_log_likelihoods
```

Short answers like "1969" or "Lyon" can be hard to compare without the question. With
`use_question=True` the question is put in front of every answer before NLI (the benchmark's
`nli_use_question` setting, A4; by default the helper follows that setting, which is off):

```python
scores = semantic_scores(["1969", "In 1969.", "1972"], question="When did Apollo 11 land?", use_question=True)
```

In A4 this did not change the AUROC clearly and doubled the NLI time, so it is off by default.
`semantic_scores_many` takes `question` in each item and the same `use_question` argument.

## The scores

| Score | Idea | All samples agree | Needs |
|---|---|---|---|
| `semantic_entropy` | Entropy of the meaning clusters, each weighted by how likely the model found its samples (Kuhn et al., 2023) | 0 | sample log-likelihoods (`logprobs` access) |
| `discrete_semantic_entropy` | Same, every sample counts equally | 0 | text only |
| `lexical_similarity` | Minus the mean word overlap (ROUGE-L) between samples; no NLI | -1 | text only |
| `num_semantic_sets` | Number of meaning clusters (Lin et al., 2024) | 1 | text only |
| `eig_val_laplacian` | Continuous cluster count from the eigenvalues of the samples' similarity graph | 1 | text only |
| `deg_mat` | 1 minus how strongly, on average, each sample agrees with the others | 0 | text only |
| `eccentricity` | How far the samples lie from their centre in the graph's spectral embedding | 0 | text only |

`scores["clusters"]` gives the cluster of every sample, e.g. `[0, 0, 0, 1]`.

Which one to use? On TriviaQA (200 questions, Qwen2.5 0.5B and 1.5B) semantic entropy,
discrete SE, `eig_val_laplacian` and `deg_mat` are statistically tied (AUROC 0.74-0.79);
`eig_val_laplacian` had the best PRR on both models. `num_semantic_sets` is a whole
number, so many questions tie. Without log-likelihoods (an API without `logprobs`), use
`discrete_semantic_entropy` or `eig_val_laplacian`. Full numbers: `results/*/triviaqa_200/summary.md`.

## Many questions

`semantic_scores_many` scores a list of questions; the NLI pairs of 16 questions
share model batches, and progress is logged:

```python
from common.semantic_scores import semantic_scores_many

items = [
    {"samples": ["Paris", "Paris", "Lyon"], "sample_log_likelihoods": [-0.1, -0.2, -1.5]},
    {"samples": ["1969", "1969", "1969"]},  # no log-likelihoods: semantic_entropy is nan
]
results = semantic_scores_many(items)
```

Example log (times differ by machine):

```text
loading the NLI model microsoft/deberta-large-mnli (first call only)
NLI model ready in 3.2 s
scored 2/2 questions (0.41 s per question, about 0 s left)
```

## Cheaper: the embedding backend

Instead of the NLI model, the answers can be compared with sentence embeddings
(`sentence-transformers/all-mpnet-base-v2`, cosine similarity above 0.9 counts as the
same meaning). It is about 26-33x faster than the deberta-large NLI model (measured on
20 questions). On TriviaQA its AUROC was on par with NLI on Qwen 0.5B, where the threshold
was chosen (SE 0.745 vs 0.744), and about 0.03 lower on the held-out Qwen 1.5B (SE 0.744 vs
0.776, EigVal 0.752 vs 0.784) (B3, `results/*/triviaqa_200/analysis/embedding_backend.md`). The clusterer is
`probly.representer.semantic_clustering.huggingface.HFEmbeddingSemanticClusterer`,
a drop-in for the NLI clusterer:

```python
from common.semantic_scores import semantic_scores

from probly.representer.semantic_clustering.huggingface import HFEmbeddingSemanticClusterer

scores = semantic_scores(samples, clusterer=HFEmbeddingSemanticClusterer.from_model_name())
```

In the benchmark, one argument switches SE, discrete SE and the graph measures
(results go to `results/<model>/<dataset>_<n>_embedding/`, next to the NLI ones):

```bash
uv run python -m cli run --config configs/qwen0.5b_triviaqa_200.yaml --semantic-backend embedding
```

The backend is part of the NLI settings (`NliSettings(backend="embedding", threshold=0.9)`,
also settable with `set_nli_settings(backend=...)`). It goes into the setup's tag and into every
matrix fingerprint, so embedding matrices get their own cache file
(`data/<model>/<dataset>_<n>.nli.embedding-all-mpnet-base-v2@0.9.npz`) and can never be read
as NLI matrices, or the other way round.

## Why uncertain? Decomposed semantic entropy

`decomposed_semantic_entropy` (B4, Hou et al., 2024) asks *why* the answers disagree.
The model rephrases the question 3 times and answers each rephrasing 5 times; all 15
answers are clustered together and the entropy splits into

- **aleatoric**: the answers change with the reading of the question, so the question
  is ambiguous ("Who won the World Cup?" without a year);
- **epistemic**: the answers disagree even for one reading, so the model does not know.

total = aleatoric + epistemic. It costs about 20 generation passes per question, so a
normal benchmark run skips it (opt-in in the registry). Run it on 50 questions:

```bash
uv run python -m scripts.decomposed_example --model qwen0.5b --questions 50
```

or on a whole run with `--methods decomposed_semantic_entropy`. The record has
`aleatoric`, `epistemic` and the clarifications, answers and clusters per question.
Results: `results/qwen0.5b/triviaqa_200/analysis/decomposed_se.md`.
On the first 50 TriviaQA questions of Qwen 0.5B (8 answered correctly) the AUROC was 0.71 for
the total, 0.72 for the epistemic and 0.57 for the aleatoric part, against 0.86 for plain
semantic entropy on the same questions. With only 8 correct answers these numbers move by
several points between runs (an earlier run with deberta-base gave 0.76 for the total), so the
split is for explaining *why* an answer is uncertain, not a better error detector.

## How many samples?

10 samples are the benchmark default. With 5 the AUROC of semantic entropy drops by
about 0.01-0.02, with 2-3 clearly more (see `results/*/triviaqa_200/analysis/sample_count.md`, B2).

## Errors

Inputs are checked before the NLI model is loaded, and the message says how to fix them:

| Message starts with | Fix |
|---|---|
| `samples must be a list of answers, got one string` | pass `["answer 1", "answer 2", ...]` |
| `need at least 2 sampled answers` | sample the model several times (`do_sample=True`) |
| `sample 3 is empty` | drop empty answers or allow more new tokens |
| `sample 1 is int, not text` | decode the generated tokens first |
| `got 10 samples but 9 log-likelihoods` | one log-likelihood per sample, same order |
| `log-likelihood 2 is nan` | use finite numbers, e.g. the mean token log-prob |
| `question 4: ...` (`semantic_scores_many`) | the same checks, with the question's position |

All of them are `SemanticInputError`, a `ValueError`.

## Limits

- The NLI model compares the answers without the question. Very short answers
  ("1969" vs "1970") are judged as text only; A4 studies adding the question.
- The clusters come from probly's greedy clusterer: a sample joins the first
  cluster whose first member it does not contradict (and is not neutral to in
  both directions).
- The graph scores follow LM-Polygraph's NLI versions with two documented
  differences (see `common/graph_measures.py`).
