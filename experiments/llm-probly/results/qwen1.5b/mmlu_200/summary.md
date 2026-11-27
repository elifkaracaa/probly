Run `qwen1.5b/mmlu_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| semantic_entropy | agreement | logprobs | 12 | ok | 0.520 | 0.447 | 0.597 | tied | -0.033 | -0.138 | 0.066 | 0.631 | 0.047 |  | 0 | 0.021 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.520 | 0.446 | 0.597 | tied | -0.033 | -0.139 | 0.067 | 0.630 | 0.047 |  | 0 | 0.000 |
| num_semantic_sets | agreement | text_only | 12 | ok | 0.526 | 0.457 | 0.598 | tied | -0.027 | -0.129 | 0.074 | 0.628 | 0.052 |  | 0 | 0.000 |
| eig_val_laplacian | agreement | text_only | 12 | ok | 0.524 | 0.451 | 0.597 | tied | -0.029 | -0.133 | 0.068 | 0.624 | 0.054 |  | 0 | 0.000 |
| deg_mat | agreement | text_only | 12 | ok | 0.524 | 0.449 | 0.598 | tied | -0.029 | -0.135 | 0.070 | 0.626 | 0.054 |  | 0 | 0.000 |
| eccentricity | agreement | text_only | 12 | ok | 0.550 | 0.481 | 0.625 | tied | -0.003 | -0.110 | 0.099 | 0.617 | 0.076 |  | 0 | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.553 | 0.469 | 0.638 | best | 0.000 | 0.000 | 0.000 | 0.610 | 0.153 |  | 0 | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.526 | 0.451 | 0.602 | tied | -0.027 | -0.133 | 0.072 | 0.625 | 0.055 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.505 | 0.419 | 0.587 | tied | -0.048 | -0.167 | 0.071 | 0.646 | 0.006 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.508 | 0.421 | 0.594 | tied | -0.045 | -0.175 | 0.075 | 0.638 | 0.007 |  | 0 | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.505 | 0.419 | 0.587 | tied | -0.048 | -0.167 | 0.071 | 0.646 | 0.006 |  | 0 | 0.000 |
