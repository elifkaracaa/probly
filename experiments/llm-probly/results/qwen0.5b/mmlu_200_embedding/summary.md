Run `qwen0.5b/mmlu_200_embedding`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | coverage | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deg_mat | agreement | text_only | 12 | ok | 0.490 | 0.381 | 0.600 | tied | -0.032 | -0.094 | 0.028 | 0.799 | 0.041 |  | 0 |  | 0.000 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.514 | 0.415 | 0.616 | tied | -0.008 | -0.055 | 0.040 | 0.793 | 0.061 |  | 0 |  | 0.000 |
| eccentricity | agreement | text_only | 12 | ok | 0.522 | 0.429 | 0.616 | best | 0.000 | 0.000 | 0.000 | 0.788 | 0.061 |  | 0 |  | 0.000 |
| eig_val_laplacian | agreement | text_only | 12 | ok | 0.494 | 0.385 | 0.604 | tied | -0.028 | -0.086 | 0.029 | 0.798 | 0.046 |  | 0 |  | 0.000 |
| num_semantic_sets | agreement | text_only | 12 | ok | 0.522 | 0.429 | 0.616 | tied | -0.000 | -0.001 | 0.000 | 0.788 | 0.061 |  | 0 |  | 0.000 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.495 | 0.387 | 0.603 | tied | -0.026 | -0.090 | 0.036 | 0.801 | 0.045 |  | 0 |  | 0.033 |
