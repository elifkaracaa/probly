Run `qwen1.5b/mmlu_200_embedding`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| semantic_entropy | agreement | logprobs | 12 | ok | 0.527 | 0.452 | 0.603 | tied | -0.007 | -0.028 | 0.011 | 0.625 | 0.056 |  | 0 | 0.022 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.526 | 0.451 | 0.601 | tied | -0.008 | -0.025 | 0.009 | 0.625 | 0.056 |  | 0 | 0.000 |
| num_semantic_sets | agreement | text_only | 12 | ok | 0.534 | 0.463 | 0.607 | best | 0.000 | 0.000 | 0.000 | 0.622 | 0.062 |  | 0 | 0.000 |
| eig_val_laplacian | agreement | text_only | 12 | ok | 0.522 | 0.448 | 0.598 | tied | -0.011 | -0.031 | 0.007 | 0.627 | 0.053 |  | 0 | 0.000 |
| deg_mat | agreement | text_only | 12 | ok | 0.523 | 0.449 | 0.599 | tied | -0.011 | -0.031 | 0.008 | 0.627 | 0.054 |  | 0 | 0.000 |
| eccentricity | agreement | text_only | 12 | ok | 0.521 | 0.445 | 0.594 | tied | -0.013 | -0.031 | 0.001 | 0.627 | 0.051 |  | 0 | 0.000 |
