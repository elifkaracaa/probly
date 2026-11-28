# Text-only methods for API-hosted models (F2)

The four text-only methods run on any model served behind an
**OpenAI-compatible API** (Ollama, vLLM, LM Studio, llama.cpp server, OpenAI,
OpenRouter, ...). No weights, no logits, no new dependency: the client uses
only the Python standard library.

| Method | Plugin | API calls per question |
|---|---|---|
| Lexical similarity | `lexical_similarity` (unchanged, reads the cache) | 0 |
| Discrete semantic entropy | `discrete_semantic_entropy` (unchanged, reads the cache) | 0 |
| Verbalized confidence (2S) | `verbalized_2s_api` | 1 |
| Empirical p(True) | `p_true_empirical_api` | 10 (1 if the server honours `n`) |

Generating the cache costs 1 greedy + 10 sample calls per question (2 if the
server honours `n`). The prompts and parsers are the week-1 ones
(`scripts/r3_methods.py`); `tests/test_api_backend.py` checks they stay identical.

## Run it on a laptop (Ollama, CPU)

```bash
ollama pull qwen2.5:0.5b
ollama serve          # on old GPUs: OLLAMA_VULKAN=0 GGML_VK_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES=-1 ollama serve

cd experiments/llm-probly
# 1. generate: same questions and aliases as the local cache
python -m scripts.generate_api --model-key ollama-qwen0.5b \
  --api-model qwen2.5:0.5b --base-url http://localhost:11434/v1 \
  --source data/qwen0.5b/triviaqa_200.jsonl          # add --resume after an interruption

# 2. score: writes results/ollama-qwen0.5b/triviaqa_200/<method>.json
python -m scripts.score_api_cache --cache data/ollama-qwen0.5b/triviaqa_200.jsonl \
  --compare data/qwen0.5b/triviaqa_200.jsonl
```

For a hosted API set `OPENAI_API_KEY` and pass e.g.
`--base-url https://api.openai.com/v1 --api-model gpt-4o-mini`.

## Cache format

Rows have the usual fields (`id`, `question`, `aliases`, `greedy_answer`,
`samples`, `correct`, `samples_correct`, `model`) plus `"access": "text"` and
`"api": {"model", "base_url"}`. All log-likelihood fields are `null`:
logit methods (MSP, perplexity, likelihood, SE with likelihoods) do not apply,
and `semantic_entropy` returns `nan` on these rows (`se_and_dse` in
`common/nli_matrix.py`; discrete SE and the graph measures still work). The `*_api`
plugins raise a clear error on caches of local models; in `common/registry.py` they are
`text_only` and `opt_in`, so a benchmark run over `all` methods skips them.

## Result: qwen2.5:0.5b via Ollama, TriviaQA-200

| Method | AUROC (API) | AUROC (local HF Qwen2.5-0.5B-Instruct) |
|---|---|---|
| Lexical similarity | 0.699 | 0.757 |
| Discrete SE | 0.603 | 0.744 |
| Verbalized 2S | 0.438 | 0.584 |
| Empirical p(True) | 0.565 | 0.518 |

Accuracy 23.5 % (API) vs 19.0 % (local); 0 failures; 11 requests and
~8.8 s per question for generation on a 2014 laptop CPU.

The API run is **not** an equivalence test of the local run: Ollama serves a
4-bit quantized build with its own sampling defaults (top-k 40, top-p 0.9,
repeat penalty 1.1), which changes sample diversity and therefore the
consistency methods. The claim is that the text-only methods work for
API-hosted models, not that they reproduce the local numbers.
