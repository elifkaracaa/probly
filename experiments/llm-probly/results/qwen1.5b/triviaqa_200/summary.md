Run `qwen1.5b/triviaqa_200`, 200 questions, precision float32

| method | family | access | cost | status | auroc | ci_low | ci_high | vs_best | d_auroc | d_ci_low | d_ci_high | aurc | prr | ece | n_failed | coverage | seconds_per_question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dummy_random | baseline | text_only | 1 | ok | 0.521 | 0.439 | 0.610 | worse | -0.313 | -0.406 | -0.214 | 0.597 | 0.131 |  | 0 |  | 0.000 |
| cocoa_msp | hybrid | logprobs | 12 | ok | 0.759 | 0.689 | 0.820 | worse | -0.075 | -0.134 | -0.021 | 0.457 | 0.481 |  | 0 |  | 0.000 |
| cocoa_p_true | hybrid | logprobs | 13 | ok | 0.833 | 0.775 | 0.886 | best | 0.000 | 0.000 | 0.000 | 0.423 | 0.571 |  | 0 |  | 0.313 |
| deg_mat | agreement | text_only | 12 | ok | 0.779 | 0.711 | 0.835 | worse | -0.054 | -0.103 | -0.011 | 0.449 | 0.499 |  | 0 |  | 0.000 |
| discrete_semantic_entropy | agreement | text_only | 12 | ok | 0.779 | 0.716 | 0.830 | worse | -0.055 | -0.105 | -0.008 | 0.452 | 0.508 |  | 0 |  | 0.000 |
| eccentricity | agreement | text_only | 12 | ok | 0.748 | 0.674 | 0.813 | worse | -0.085 | -0.145 | -0.033 | 0.469 | 0.460 |  | 0 |  | 0.000 |
| eig_val_laplacian | agreement | text_only | 12 | ok | 0.784 | 0.717 | 0.841 | worse | -0.049 | -0.098 | -0.007 | 0.441 | 0.506 |  | 0 |  | 0.000 |
| greedy_cluster_share | agreement | text_only | 12 | ok | 0.770 | 0.705 | 0.826 | worse | -0.063 | -0.116 | -0.013 | 0.455 | 0.501 |  | 0 |  | 0.000 |
| largest_cluster_share | agreement | text_only | 12 | ok | 0.777 | 0.714 | 0.830 | worse | -0.057 | -0.107 | -0.008 | 0.452 | 0.507 |  | 0 |  | 0.000 |
| lexical_similarity | agreement | text_only | 11 | ok | 0.752 | 0.680 | 0.817 | worse | -0.082 | -0.137 | -0.033 | 0.460 | 0.479 |  | 0 |  | 0.000 |
| likelihood_mean | baseline | logprobs | 1 | ok | 0.790 | 0.723 | 0.846 | tied | -0.044 | -0.104 | 0.012 | 0.433 | 0.618 |  | 0 |  | 0.000 |
| mean_token_entropy | baseline | white_box | 1 | ok | 0.805 | 0.747 | 0.860 | tied | -0.028 | -0.090 | 0.034 | 0.421 | 0.653 |  | 0 |  | 0.000 |
| msp | baseline | logprobs | 1 | ok | 0.748 | 0.673 | 0.817 | worse | -0.085 | -0.158 | -0.016 | 0.462 | 0.552 |  | 0 |  | 0.000 |
| num_semantic_sets | agreement | text_only | 12 | ok | 0.775 | 0.716 | 0.830 | worse | -0.058 | -0.109 | -0.010 | 0.453 | 0.504 |  | 0 |  | 0.004 |
| p_true_brainstorm | self_eval | logprobs | 12 | ok | 0.806 | 0.738 | 0.864 | tied | -0.027 | -0.093 | 0.036 | 0.427 | 0.577 | 0.389 | 0 | 0.999 | 0.505 |
| p_true_empirical | self_eval | text_only | 11 | ok | 0.770 | 0.708 | 0.826 | tied | -0.063 | -0.127 | 0.001 | 0.467 | 0.472 | 0.314 | 0 | 1.000 | 2.252 |
| p_true_zeroshot | self_eval | logprobs | 2 | ok | 0.825 | 0.764 | 0.883 | tied | -0.008 | -0.064 | 0.054 | 0.411 | 0.650 | 0.335 | 0 | 1.000 | 0.313 |
| semantic_entropy | agreement | logprobs | 12 | ok | 0.776 | 0.713 | 0.827 | worse | -0.057 | -0.107 | -0.008 | 0.455 | 0.505 |  | 0 |  | 0.255 |
| verbalized_2s | self_eval | text_only | 2 | ok | 0.538 | 0.488 | 0.587 | worse | -0.296 | -0.365 | -0.226 | 0.606 | 0.081 | 0.563 | 0 |  | 0.560 |
