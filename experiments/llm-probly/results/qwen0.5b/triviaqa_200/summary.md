Run `qwen0.5b/triviaqa_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| largest_cluster_share | agreement | text_only | 12 | ok | 0.738 | 0.636 | 0.829 | tied | -0.053 | -0.146 | 0.043 | 0.688 | 0.500 |  | 0 | 0.000 |
| greedy_cluster_share | agreement | text_only | 12 | ok | 0.758 | 0.655 | 0.845 | tied | -0.033 | -0.128 | 0.060 | 0.679 | 0.517 |  | 0 | 0.000 |
| deg_mat | agreement | text_only | 12 | ok | 0.751 | 0.650 | 0.839 | tied | -0.040 | -0.131 | 0.053 | 0.686 | 0.514 |  | 0 | 0.000 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.751 | 0.646 | 0.839 | tied | -0.040 | -0.131 | 0.056 | 0.680 | 0.520 |  | 0 | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.567 | 0.467 | 0.669 | worse | -0.224 | -0.368 | -0.086 | 0.771 | 0.160 |  | 0 | 0.000 |
| eccentricity | agreement | text_only | 12 | ok | 0.750 | 0.650 | 0.839 | tied | -0.040 | -0.130 | 0.053 | 0.681 | 0.513 |  | 0 | 0.000 |
| eig_val_laplacian | agreement | text_only | 12 | ok | 0.758 | 0.649 | 0.844 | tied | -0.033 | -0.127 | 0.060 | 0.674 | 0.530 |  | 0 | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.757 | 0.663 | 0.839 | tied | -0.034 | -0.097 | 0.033 | 0.692 | 0.467 |  | 0 | 0.001 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.791 | 0.701 | 0.877 | best | 0.000 | 0.000 | 0.000 | 0.671 | 0.576 |  | 0 | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.773 | 0.676 | 0.858 | tied | -0.018 | -0.046 | 0.008 | 0.680 | 0.548 |  | 0 | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.680 | 0.576 | 0.777 | worse | -0.110 | -0.188 | -0.028 | 0.719 | 0.400 |  | 0 | 0.000 |
| num_semantic_sets | agreement | text_only | 12 | ok | 0.759 | 0.657 | 0.847 | tied | -0.032 | -0.126 | 0.063 | 0.676 | 0.530 |  | 0 | 0.001 |
| p_true_brainstorm | self_eval | logprobs | 12 | ok | 0.685 | 0.588 | 0.769 | worse | -0.106 | -0.208 | -0.001 | 0.730 | 0.291 | 0.662 | 0 | 0.293 |
| p_true_empirical | self_eval | text_only | 11 | ok | 0.518 | 0.421 | 0.611 | worse | -0.273 | -0.396 | -0.154 | 0.805 | -0.000 | 0.651 | 0 | 0.847 |
| p_true_zeroshot | self_eval | logprobs | 2 | ok | 0.669 | 0.581 | 0.751 | worse | -0.122 | -0.236 | -0.019 | 0.744 | 0.275 | 0.690 | 0 | 0.113 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.744 | 0.638 | 0.835 | tied | -0.047 | -0.137 | 0.051 | 0.685 | 0.514 |  | 0 | 0.000 |
| verbalized_2s | self_eval | text_only | 2 | ok | 0.584 | 0.505 | 0.658 | worse | -0.207 | -0.322 | -0.102 | 0.777 | 0.090 | 0.722 | 0 | 0.194 |
