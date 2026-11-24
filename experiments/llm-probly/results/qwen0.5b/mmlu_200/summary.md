Run `qwen0.5b/mmlu_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| semantic_entropy | agreement | logprobs | 12 | ok | 0.504 | 0.398 | 0.609 | tied | -0.075 | -0.162 | 0.011 | 0.796 | 0.062 |  | 0 | 0.041 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.517 | 0.418 | 0.617 | tied | -0.063 | -0.152 | 0.023 | 0.792 | 0.073 |  | 0 | 0.000 |
| num_semantic_sets | agreement | text_only | 12 | ok | 0.533 | 0.450 | 0.620 | tied | -0.047 | -0.126 | 0.040 | 0.782 | 0.084 |  | 0 | 0.000 |
| eig_val_laplacian | agreement | text_only | 12 | ok | 0.522 | 0.426 | 0.620 | tied | -0.057 | -0.151 | 0.028 | 0.792 | 0.070 |  | 0 | 0.000 |
| deg_mat | agreement | text_only | 12 | ok | 0.503 | 0.400 | 0.608 | tied | -0.077 | -0.173 | 0.010 | 0.795 | 0.053 |  | 0 | 0.000 |
| eccentricity | agreement | text_only | 12 | ok | 0.518 | 0.419 | 0.620 | tied | -0.062 | -0.158 | 0.034 | 0.791 | 0.065 |  | 0 | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.516 | 0.420 | 0.611 | tied | -0.064 | -0.224 | 0.092 | 0.808 | -0.062 |  | 0 | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.507 | 0.403 | 0.611 | tied | -0.072 | -0.168 | 0.016 | 0.794 | 0.068 |  | 0 | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.543 | 0.436 | 0.648 | tied | -0.037 | -0.088 | 0.013 | 0.776 | 0.168 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.580 | 0.471 | 0.685 | best | 0.000 | 0.000 | 0.000 | 0.759 | 0.203 |  | 0 | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.501 | 0.394 | 0.605 | worse | -0.079 | -0.163 | -0.003 | 0.798 | 0.094 |  | 0 | 0.000 |
