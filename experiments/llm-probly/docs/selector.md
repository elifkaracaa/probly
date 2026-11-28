# Method selector

Which uncertainty method should I use for my model, at my budget? The selector answers this per model
and turns the chosen method's score into a calibrated confidence in [0, 1]. The result is one profile
per model (`profiles/<model>.json`, format in [`CONTRACT.md`](../CONTRACT.md#profile)) that the API and the demo read.

How it decides, per model (`selector/`):
1. Split the questions into a calibration half (100) and a held-out half (100), stratified by correctness.
2. On the calibration half, rank every method by AUROC and PRR with 95 % bootstrap intervals.
3. Keep only the methods the user can run (access level and cost from `common/registry.py`).
4. Compare every method with the best one in a paired bootstrap; among the methods that are not
   significantly worse, choose the cheapest.
5. Calibrate the chosen score with probly's Platt or isotonic calibration (chosen by 5-fold CV).

Run it (from `experiments/llm-probly`):

    uv run python -m selector.select --save        # choose, calibrate, write profiles/<model>.json
    uv run python -m selector.evaluate --splits 10 --out results/selector_held_out.md   # selector vs fixed methods, held-out, 10 splits
    uv run python -m selector.combine              # stacking all methods vs the best single method

**Choice per model** (TriviaQA-200):

| model | chosen | why | calibration | held-out ECE |
|---|---|---|---|---|
| qwen0.5b | likelihood_mean | tied with the best (eccentricity), 12x cheaper | Platt | 0.091 |
| qwen1.5b | likelihood_mean | tied with the best (p_true_zeroshot), 2x cheaper | Platt | 0.095 |
| qwen3b | p_true_zeroshot | best; every cheaper method is significantly worse | isotonic | 0.219 → 0.103 |
| qwen7b | mean_token_entropy | tied with the best (p_true_zeroshot), 2x cheaper | Platt | 0.126 |

**Held-out check** (100 questions per model that the selector never saw). The per-model table above is one
random split (seed 0); which questions land in the calibration half changes the choice, so the evaluation
was repeated over 10 random splits ([`results/selector_held_out.md`](../results/selector_held_out.md)):

| model | chosen method (number of splits out of 10) |
|---|---|
| qwen0.5b | likelihood_mean (9), mean_token_entropy (1) |
| qwen1.5b | mean_token_entropy (9), likelihood_mean (1) |
| qwen3b | mean_token_entropy (8), p_true_zeroshot (2) |
| qwen7b | mean_token_entropy (8), msp (2) |

| strategy | mean AUROC over models (sd over splits) | selector minus strategy, mean [min, max] | cost |
|---|---|---|---|
| **selector (per model)** | **0.813** (0.013) | | mostly 1 |
| always p_true_zeroshot | 0.821 (0.020) | −0.008 [−0.035, +0.025] | 2 |
| always mean_token_entropy | 0.811 (0.015) | +0.002 [−0.005, +0.011] | 1 |
| always likelihood_mean | 0.803 (0.011) | +0.010 [+0.001, +0.021] | 1 |
| always semantic_entropy | 0.759 (0.017) | +0.054 [+0.032, +0.082] | 12 |
| always greedy_cluster_share | 0.757 (0.016) | +0.056 [+0.044, +0.076] | 12 |
| always largest_cluster_share | 0.755 (0.017) | +0.058 [+0.037, +0.086] | 12 |

**Takeaways**
1. Over 10 splits the selector is on par with the best fixed methods: within ±0.01 AUROC of always using
   p_true_zeroshot or mean_token_entropy, with no consistent winner (the min/max ranges include 0).
2. It is clearly better than every expensive method: semantic entropy and the two cluster shares cost
   12x more and are about 0.05 worse. All methods: [`results/selector_held_out.md`](../results/selector_held_out.md).
3. With 100 calibration questions most methods are statistically tied, so the selector mostly picks the
   cheapest good one (mean_token_entropy). The per-model choice depends on the split; the single split in the
   first table (seed 0, mean 0.839) was a favourable one. More calibration questions (TriviaQA-1000) would let
   the selector separate the methods.
4. Failed questions (placeholder scores) are not trusted: methods that failed on more than 5 % of all
   questions in the run are excluded (only p_true_empirical on 7B, 7 %).
5. Calibration makes the score usable as a confidence: zero-shot p(True) on 3B is overconfident
   (ECE 0.219) and calibration halves that (0.103).
6. Combining all methods (`selector.combine`) helps on small models: +0.069 AUROC on 0.5B and +0.067 on 1.5B
   held-out (both significant); on 3B nothing, on 7B +0.041 but not significant.
7. Results use deberta-large for the semantic methods (NLI setup and cache, see [semantic_results.md](semantic_results.md)); the graph measures are candidates where they
   exist (0.5B, 1.5B). On 0.5B eccentricity is the best single method, but likelihood_mean is tied and 12x cheaper.
