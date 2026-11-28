# Score contract

Every method saves one file: `results/qwen0.5b/<method>.json`

1. Direction: higher = more uncertain. For reflexive methods, uncertainty = 1 - confidence.
2. Raw values: store unnormalized scores. No rescaling.
3. No missing values: if parsing fails, uncertainty = 1.0 and failed = true.
4. Same answer: all methods score the main (greedy) answer.
5. Same question ids: `ids` in the same order as `id` in `data/qwen0.5b/triviaqa_50.jsonl`.

Example:

    {"method": "p_true_zeroshot_norm", "model": "Qwen/Qwen2.5-0.5B-Instruct",
     "answer": "greedy", "ids": [0, 1, 2], "uncertainty": [0.12, 0.87, 0.40],
     "failed": [false, false, true], "seconds_per_question": 0.8}


## Pipeline

### Folders
    data/<model>/<dataset>_<n>.jsonl         cache: one row per question
    data/<model>/<dataset>_<n>.meta.json     generation settings
    results/<model>/<dataset>_<n>/<method>.json  one score file per method
    configs/<model>_<dataset>.yaml           one config per run
    methods/<method>.py                      one plugin per method

`<model>` is the short name from `scripts/generate.py` (`qwen0.5b`, `qwen1.5b`, `qwen3b`, `qwen7b`).

### Method plugin
Every file in `methods/` has only a name and a score function:

```python
NAME = "likelihood_mean"  # same as the file name and the result file name


def score(rows, model=None, tok=None) -> dict:
    """Return one score record (rules 1-5 above) for these rows."""
```

and one entry in `common/registry.py` with what the selector needs to know:

```python
MethodSpec("likelihood_mean", "baseline", "logprobs", GREEDY, note="mean greedy token log-prob")
#           name              family      access      cost (generation passes per question)
```

- `family`: `agreement` | `self_eval` | `baseline` | `hybrid`
- `access`: `text_only` (generated text only, works behind any API) | `logprobs` (token
  log-probabilities, as APIs with a logprobs option give them) | `white_box` (full next-token
  distribution or weights, local model only)
- `cost`: generation passes per question, greedy answer included: 1 = greedy only,
  +10 for the cached samples, +1 per extra model call, +1 for an NLI pass (semantic entropy = 12)
- `needs_model=True`: the runner loads the LLM and passes `model, tok`
- `uses_nli=True`: the method also runs the NLI model
- `probability=True`: `1 - uncertainty` is a probability of being correct, so ECE is computed

`tests/test_registry.py` fails if a plugin has no entry or an entry has no plugin.
`uv run python -m cli methods` prints the registry.

Rules:
- `score()` does **not** save and does **not** load data. The runner gives it the rows and saves what it returns.
- `ids` in the record are `[row["id"] for row in rows]`, in the same order.
- No hardcoded model or dataset names inside a plugin.
- Put `seconds_per_question` in the record. Use 0.0 if the method only reads the cache.
- If your score is a probability of being correct (uncertainty = 1 - probability), set `probability=True` in its registry entry, so ECE is computed for it.

### Runner
    uv run python -m cli run --config configs/qwen0.5b_triviaqa_200.yaml
    # steps: generate -> label -> methods -> evaluate -> table
    # a step is skipped if its output file already exists
    # --methods a,b            run only these methods
    # --skip-model-methods     skip methods with needs_model=True in the registry
    # --force                  recompute even if the file exists

## Profile

The selector writes one file per model: `profiles/<model>.json`. The API and the demo
only read this file, with `selector.profile.load_profile()`; they never run the selector.
Example: `docs/profile_example_qwen1.5b.json` (real 1.5B numbers; the chosen method is only an example).

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | Format version, currently 1. The loader refuses other versions. |
| `model`, `model_id` | str | Short name (`qwen1.5b`) and Hugging Face id (`Qwen/Qwen2.5-1.5B-Instruct`). |
| `dataset`, `n_questions` | str, int | Run the selector was fitted on (`triviaqa`, 200). |
| `split_seed`, `calibration_size` | int, int | Calibration / held-out split (`selector.split`), so the result can be rebuilt. |
| `method.name` | str | Chosen method, same as its plugin and result file; `"stack"` if the stacked combination (`selector/combine.py`) wins. |
| `method.family` | str | `agreement` \| `self_eval` \| `baseline` \| `hybrid` (as in the plugins). |
| `method.access` | str | `text_only` \| `logprobs` \| `white_box`, from `common/registry.py`. |
| `method.cost` | float | Generation passes per question from `common/registry.py`: 1 = greedy answer only, +10 samples, +1 per extra call, +1 for an NLI pass. |
| `method.parameters` | object | Settings the API needs to run the method, e.g. `{"n_samples": 10}`. For `stack`: `methods`, `weights`, `intercept`, `means`, `stds`. |
| `method.probability` | bool | True if `1 - uncertainty` is already a probability of being correct. |
| `reason` | str | One sentence on why this method was chosen (shown in the demo). |
| `calibration_metrics` | object | `auroc`, `auroc_low`, `auroc_high`, `prr`, `prr_low`, `prr_high` (95 % bootstrap) on the calibration part. |
| `held_out_metrics` | object \| null | Same fields for the chosen method on the held-out part; filled by `selector.select --save` (`null` only in hand-made profiles). |
| `calibration.type` | str | `none` (not calibrated yet) \| `platt` \| `isotonic` (`selector/calibrate.py`). |
| `calibration.parameters` | object | `platt`: `a`, `b`. `isotonic`: `x` (increasing uncertainties), `y` (confidences). |
| `calibration.ece_before`, `ece_after` | float \| null | ECE on the held-out part before and after calibration. |
| `candidates` | object | Calibration-part metrics of every compared method, keyed by name (for the demo's comparison). |
| `created_at` | str | UTC time the profile was written (ISO 8601). |

To turn a raw score into a confidence, the API calls `selector.profile.confidence(profile, uncertainty)`:

- `platt`: confidence = 1 / (1 + exp(a * u + b))
- `isotonic`: linear interpolation of `(x, y)`, constant outside the range
- `none`: `1 - u` for probability methods; an error for raw scores (msp, SE, ...), which need calibration first

Rules: fields are only added in a new `schema_version`; missing or unknown fields make `load_profile()` fail.
