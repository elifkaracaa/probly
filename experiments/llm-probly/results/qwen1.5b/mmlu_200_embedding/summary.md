Run `qwen1.5b/mmlu_200_embedding`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | coverage | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deg_mat | agreement | text_only | 12 | ok | 0.523 | 0.449 | 0.599 | tied | -0.012 | -0.033 | 0.006 | 0.627 | 0.054 |  | 0 |  | 0.000 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.526 | 0.451 | 0.601 | tied | -0.009 | -0.027 | 0.008 | 0.625 | 0.056 |  | 0 |  | 0.000 |
| eccentricity | agreement | text_only | 12 | ok | 0.535 | 0.465 | 0.609 | best | 0.000 | 0.000 | 0.000 | 0.622 | 0.063 |  | 0 |  | 0.000 |
| eig_val_laplacian | agreement | text_only | 12 | ok | 0.522 | 0.448 | 0.598 | tied | -0.013 | -0.031 | 0.005 | 0.627 | 0.053 |  | 0 |  | 0.000 |
| num_semantic_sets | agreement | text_only | 12 | ok | 0.534 | 0.463 | 0.607 | tied | -0.001 | -0.004 | 0.000 | 0.622 | 0.062 |  | 0 |  | 0.000 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.527 | 0.452 | 0.603 | tied | -0.009 | -0.029 | 0.010 | 0.625 | 0.056 |  | 0 |  | 0.022 |
