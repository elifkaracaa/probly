Run `qwen1.5b/triviaqa_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | coverage | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p_true_empirical | self_eval | text_only | 11 | ok | 0.765 | 0.703 | 0.822 | worse | -0.060 | -0.098 | -0.025 | 0.471 | 0.460 | 0.308 | 0 | 1.000 | 1.592 |
| deg_mat | agreement | text_only | 12 | ok | 0.779 | 0.711 | 0.835 | tied | -0.046 | -0.133 | 0.035 | 0.449 | 0.499 |  | 0 |  | 0.000 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.779 | 0.716 | 0.830 | tied | -0.046 | -0.128 | 0.035 | 0.452 | 0.508 |  | 0 |  | 0.000 |
| dummy_random | baseline | text_only | 1 | ok | 0.521 | 0.439 | 0.610 | worse | -0.304 | -0.399 | -0.204 | 0.597 | 0.131 |  | 0 |  | 0.000 |
| eccentricity | agreement | text_only | 12 | ok | 0.748 | 0.674 | 0.813 | tied | -0.077 | -0.169 | 0.009 | 0.469 | 0.460 |  | 0 |  | 0.000 |
| eig_val_laplacian | agreement | text_only | 12 | ok | 0.784 | 0.717 | 0.841 | tied | -0.041 | -0.129 | 0.044 | 0.441 | 0.506 |  | 0 |  | 0.000 |
| greedy_cluster_share | agreement | text_only | 12 | ok | 0.770 | 0.705 | 0.826 | tied | -0.055 | -0.142 | 0.025 | 0.455 | 0.501 |  | 0 |  | 0.000 |
| largest_cluster_share | agreement | text_only | 12 | ok | 0.777 | 0.714 | 0.830 | tied | -0.049 | -0.133 | 0.030 | 0.452 | 0.507 |  | 0 |  | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.752 | 0.680 | 0.817 | tied | -0.073 | -0.167 | 0.006 | 0.460 | 0.479 |  | 0 |  | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.790 | 0.723 | 0.846 | tied | -0.035 | -0.118 | 0.038 | 0.433 | 0.618 |  | 0 |  | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.805 | 0.747 | 0.860 | tied | -0.020 | -0.097 | 0.051 | 0.421 | 0.653 |  | 0 |  | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.748 | 0.673 | 0.817 | tied | -0.077 | -0.175 | 0.019 | 0.462 | 0.552 |  | 0 |  | 0.000 |
| num_semantic_sets | agreement | text_only | 12 | ok | 0.775 | 0.716 | 0.830 | tied | -0.050 | -0.133 | 0.032 | 0.453 | 0.504 |  | 0 |  | 0.004 |
| p_true_brainstorm | self_eval | logprobs | 12 | ok | 0.806 | 0.738 | 0.864 | tied | -0.019 | -0.083 | 0.044 | 0.427 | 0.577 | 0.389 | 0 | 0.999 | 0.505 |
| p_true_zeroshot | self_eval | logprobs | 2 | ok | 0.825 | 0.764 | 0.883 | best | 0.000 | 0.000 | 0.000 | 0.411 | 0.650 | 0.335 | 0 | 1.000 | 0.313 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.776 | 0.713 | 0.827 | tied | -0.049 | -0.131 | 0.033 | 0.455 | 0.505 |  | 0 |  | 0.255 |
| verbalized_2s | self_eval | text_only | 2 | ok | 0.538 | 0.488 | 0.587 | worse | -0.287 | -0.366 | -0.216 | 0.606 | 0.081 | 0.563 | 0 |  | 0.560 |
