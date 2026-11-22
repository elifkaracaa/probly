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
| always discrete_semantic_entropy | 0.773 | 12 |
| always semantic_entropy | 0.768 | 12 |
| always msp | 0.761 | 1 |
| always p_true_empirical | 0.719 | 11 |
| always verbalized_2s | 0.648 | 2 |
