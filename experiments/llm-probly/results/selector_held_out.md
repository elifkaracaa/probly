# Selector vs fixed methods (held-out, triviaqa 200, 100 held-out questions per model)

The selector chooses on the calibration part only; every number below is on the held-out part.

| Model | Selector's choice | Cost | Held-out AUROC | Best fixed on this model (oracle) | Selector - oracle [95 % CI] |
|---|---|---|---|---|---|
| qwen0.5b | likelihood_mean | 1 | 0.838 | likelihood_mean 0.838 | +0.000 [+0.000, +0.000] |
| qwen1.5b | likelihood_mean | 1 | 0.813 | mean_token_entropy 0.847 | -0.034 [-0.070, -0.001] |
| qwen3b | p_true_zeroshot | 2 | 0.851 | p_true_zeroshot 0.851 | +0.000 [+0.000, +0.000] |
| qwen7b | mean_token_entropy | 1 | 0.855 | p_true_zeroshot 0.880 | -0.025 [-0.114, +0.063] |

Mean over the models (selector vs always using one method):

| Strategy | Mean held-out AUROC | Mean cost |
|---|---|---|
| **Selector (per model)** | **0.839** | **1.2** |
| always mean_token_entropy | 0.841 | 1 |
| always likelihood_mean | 0.818 | 1 |
| always p_true_zeroshot | 0.814 | 2 |
| always p_true_brainstorm | 0.804 | 12 |
| always lexical_similarity | 0.773 | 11 |
| always discrete_semantic_entropy | 0.772 | 12 |
| always semantic_entropy | 0.765 | 12 |
| always msp | 0.761 | 1 |
| always p_true_empirical | 0.719 | 11 |
| always verbalized_2s | 0.648 | 2 |

## Over 10 random splits (seeds 0-9)

The table above is one split. Which questions land in the calibration half changes the choice per model, so the selector was rerun on every split:

| Model | Chosen method (number of splits) |
|---|---|
| qwen0.5b | likelihood_mean (9), mean_token_entropy (1) |
| qwen1.5b | mean_token_entropy (9), likelihood_mean (1) |
| qwen3b | mean_token_entropy (8), p_true_zeroshot (2) |
| qwen7b | mean_token_entropy (8), msp (2) |

Mean held-out AUROC over the models, averaged over the splits:

| Strategy | Mean AUROC (sd over splits) | Selector minus strategy: mean [min, max] over splits |
|---|---|---|
| **Selector (per model)** | **0.813** (0.013) | |
| always p_true_zeroshot | 0.821 (0.020) | -0.008 [-0.035, +0.025] |
| always mean_token_entropy | 0.811 (0.015) | +0.002 [-0.005, +0.011] |
| always likelihood_mean | 0.803 (0.011) | +0.010 [+0.001, +0.021] |
| always p_true_brainstorm | 0.798 (0.010) | +0.015 [-0.029, +0.035] |
| always msp | 0.764 (0.012) | +0.049 [+0.010, +0.078] |
| always discrete_semantic_entropy | 0.763 (0.017) | +0.050 [+0.029, +0.079] |
| always semantic_entropy | 0.759 (0.017) | +0.054 [+0.032, +0.082] |
| always lexical_similarity | 0.756 (0.015) | +0.057 [+0.044, +0.091] |
| always p_true_empirical | 0.740 (0.018) | +0.073 [+0.050, +0.120] |
| always verbalized_2s | 0.645 (0.008) | +0.168 [+0.148, +0.191] |
