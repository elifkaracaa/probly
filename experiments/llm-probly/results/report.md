# LLM-probly benchmark report

*Generated 2026-11-24 11:16 UTC by `python -m cli report` (1000 bootstrap rounds).*

AUROC: how well the uncertainty separates wrong from right answers (0.5 = random, 1 = perfect).
**vs best** compares every method with the run's best one by a paired bootstrap: *tied* means the 95 % interval of the AUROC difference contains 0, so the data cannot tell the two apart. **cost** is generation passes per question (1 = the greedy answer alone; registry, CONTRACT.md). ★ marks the pick: the cheapest method that is best or tied with the best. *Pick via API* only allows methods that need text or log-probabilities, which an OpenAI-compatible API returns. If the random reference is tied with the best method, no method beats random on that run.

## Runs

| model | dataset | questions | accuracy | precision, device | methods | pick | pick via API |
|---|---|---|---|---|---|---|---|
| qwen0.5b | triviaqa | 200 | 0.190 | float32, cpu | 17 | likelihood_mean | likelihood_mean |
| qwen1.5b | triviaqa | 200 | 0.375 | float32, cpu | 17 | mean_token_entropy | likelihood_mean |
| qwen3b | triviaqa | 200 | 0.485 | float32, mps | 13 | p_true_zeroshot | p_true_zeroshot |
| qwen7b | triviaqa | 200 | 0.590 | float16, mps | 13 | mean_token_entropy | msp |
| smollm2-1.7b | triviaqa | 200 | 0.360 | float32, mps | 5 | mean_token_entropy | likelihood_mean |
| qwen0.5b | triviaqa | 1000 | 0.207 | float32, cpu | 11 | eig_val_laplacian | eig_val_laplacian |
| qwen1.5b | triviaqa | 1000 | 0.370 | float32, cpu | 11 | mean_token_entropy | likelihood_mean |
| qwen3b | triviaqa | 1000 | 0.445 | float32, mps | 5 | mean_token_entropy | none tied with the best |
| qwen0.5b | mmlu | 200 | 0.195 | float32, cpu | 11 | none (no method beats random) | none (no method beats random) |
| qwen1.5b | mmlu | 200 | 0.360 | float32, cpu | 11 | none (no method beats random) | none (no method beats random) |
| qwen3b | mmlu | 200 | 0.635 | float32, cpu | 5 | mean_token_entropy | msp |
| smollm2-1.7b | mmlu | 200 | 0.450 | float32, cpu | 5 | none (no method beats random) | none (no method beats random) |

## Methods x models, triviaqa 200 (AUROC, best per model in bold)

| method | family | access | cost | qwen0.5b | qwen1.5b | qwen3b | qwen7b | smollm2-1.7b |
|---|---|---|---|---|---|---|---|---|
| msp | baseline | logprobs | 1 | 0.680 | 0.748 | 0.753 | 0.824 | 0.795 |
| likelihood_mean | baseline | logprobs | 1 | **0.791** | 0.790 | 0.786 | 0.817 | 0.809 |
| mean_token_entropy | baseline | white_box | 1 | 0.773 | 0.805 | 0.817 | 0.833 | **0.816** |
| dummy_random | baseline | text_only | 1 | 0.567 | 0.521 | 0.500 | 0.560 | 0.525 |
| lexical_similarity | agreement | text_only | 11 | 0.757 | 0.752 | 0.757 | 0.735 | 0.796 |
| discrete_semantic_entropy | agreement | text_only | 12 | 0.751 | 0.779 | 0.765 | 0.729 |  |
| semantic_entropy | agreement | logprobs | 12 | 0.744 | 0.776 | 0.761 | 0.729 |  |
| num_semantic_sets | agreement | text_only | 12 | 0.759 | 0.775 |  |  |  |
| eig_val_laplacian | agreement | text_only | 12 | 0.758 | 0.784 |  |  |  |
| deg_mat | agreement | text_only | 12 | 0.751 | 0.779 |  |  |  |
| eccentricity | agreement | text_only | 12 | 0.750 | 0.748 |  |  |  |
| largest_cluster_share | agreement | text_only | 12 | 0.738 | 0.777 | 0.758 | 0.728 |  |
| greedy_cluster_share | agreement | text_only | 12 | 0.758 | 0.770 | 0.761 | 0.718 |  |
| p_true_zeroshot | self_eval | logprobs | 2 | 0.669 | **0.825** | **0.877** | **0.870** |  |
| p_true_brainstorm | self_eval | logprobs | 12 | 0.685 | 0.806 | 0.820 | 0.845 |  |
| p_true_empirical | self_eval | text_only | 11 | 0.518 | 0.770 | 0.830 | 0.824 |  |
| verbalized_2s | self_eval | text_only | 2 | 0.584 | 0.538 | 0.713 | 0.734 |  |

## qwen0.5b/triviaqa_200

Accuracy 0.190, float32, cpu.

| method | family | access | cost | AUROC | 95 % interval | vs best | Δ AUROC [paired 95 %] | AURC | PRR | ECE | s / question (method) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| likelihood_mean ★ | baseline | logprobs | 1 | 0.791 | [0.701, 0.877] | best |  | 0.671 | 0.576 |  | 0.00 |
| mean_token_entropy | baseline | white_box | 1 | 0.773 | [0.676, 0.858] | tied | -0.018 [-0.046, +0.008] | 0.680 | 0.548 |  | 0.00 |
| num_semantic_sets | agreement | text_only | 12 | 0.759 | [0.657, 0.847] | tied | -0.032 [-0.126, +0.063] | 0.676 | 0.530 |  | 0.00 |
| greedy_cluster_share | agreement | text_only | 12 | 0.758 | [0.655, 0.845] | tied | -0.033 [-0.128, +0.060] | 0.679 | 0.517 |  | 0.00 |
| eig_val_laplacian | agreement | text_only | 12 | 0.758 | [0.649, 0.844] | tied | -0.033 [-0.127, +0.060] | 0.674 | 0.530 |  | 0.00 |
| lexical_similarity | agreement | text_only | 11 | 0.757 | [0.663, 0.839] | tied | -0.034 [-0.097, +0.033] | 0.692 | 0.467 |  | 0.00 |
| deg_mat | agreement | text_only | 12 | 0.751 | [0.650, 0.839] | tied | -0.040 [-0.131, +0.053] | 0.686 | 0.514 |  | 0.00 |
| discrete_semantic_entropy | agreement | text_only | 12 | 0.751 | [0.646, 0.839] | tied | -0.040 [-0.131, +0.056] | 0.680 | 0.520 |  | 0.00 |
| eccentricity | agreement | text_only | 12 | 0.750 | [0.650, 0.839] | tied | -0.040 [-0.130, +0.053] | 0.681 | 0.513 |  | 0.00 |
| semantic_entropy | agreement | logprobs | 12 | 0.744 | [0.638, 0.835] | tied | -0.047 [-0.137, +0.051] | 0.685 | 0.514 |  | 0.00 |
| largest_cluster_share | agreement | text_only | 12 | 0.738 | [0.636, 0.829] | tied | -0.053 [-0.146, +0.043] | 0.688 | 0.500 |  | 0.00 |
| p_true_brainstorm | self_eval | logprobs | 12 | 0.685 | [0.588, 0.769] | worse | -0.106 [-0.208, -0.001] | 0.730 | 0.291 | 0.662 | 0.29 |
| msp | baseline | logprobs | 1 | 0.680 | [0.576, 0.777] | worse | -0.110 [-0.188, -0.028] | 0.719 | 0.400 |  | 0.00 |
| p_true_zeroshot | self_eval | logprobs | 2 | 0.669 | [0.581, 0.751] | worse | -0.122 [-0.236, -0.019] | 0.744 | 0.275 | 0.690 | 0.11 |
| verbalized_2s | self_eval | text_only | 2 | 0.584 | [0.505, 0.658] | worse | -0.207 [-0.322, -0.102] | 0.777 | 0.090 | 0.722 | 0.19 |
| dummy_random | baseline | text_only | 1 | 0.567 | [0.467, 0.669] | worse | -0.224 [-0.368, -0.086] | 0.771 | 0.160 |  | 0.00 |
| p_true_empirical | self_eval | text_only | 11 | 0.518 | [0.421, 0.611] | worse | -0.273 [-0.396, -0.154] | 0.805 | -0.000 | 0.651 | 0.85 |

## qwen1.5b/triviaqa_200

Accuracy 0.375, float32, cpu.

| method | family | access | cost | AUROC | 95 % interval | vs best | Δ AUROC [paired 95 %] | AURC | PRR | ECE | s / question (method) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p_true_zeroshot | self_eval | logprobs | 2 | 0.825 | [0.764, 0.883] | best |  | 0.411 | 0.650 | 0.335 | 0.31 |
| p_true_brainstorm | self_eval | logprobs | 12 | 0.806 | [0.738, 0.864] | tied | -0.019 [-0.083, +0.044] | 0.427 | 0.577 | 0.389 | 0.50 |
| mean_token_entropy ★ | baseline | white_box | 1 | 0.805 | [0.747, 0.860] | tied | -0.020 [-0.097, +0.051] | 0.421 | 0.653 |  | 0.00 |
| likelihood_mean | baseline | logprobs | 1 | 0.790 | [0.723, 0.846] | tied | -0.035 [-0.118, +0.038] | 0.433 | 0.618 |  | 0.00 |
| eig_val_laplacian | agreement | text_only | 12 | 0.784 | [0.717, 0.841] | tied | -0.041 [-0.129, +0.044] | 0.441 | 0.506 |  | 0.00 |
| deg_mat | agreement | text_only | 12 | 0.779 | [0.711, 0.835] | tied | -0.046 [-0.133, +0.035] | 0.449 | 0.499 |  | 0.00 |
| discrete_semantic_entropy | agreement | text_only | 12 | 0.779 | [0.716, 0.830] | tied | -0.046 [-0.128, +0.035] | 0.452 | 0.508 |  | 0.00 |
| largest_cluster_share | agreement | text_only | 12 | 0.777 | [0.714, 0.830] | tied | -0.049 [-0.133, +0.030] | 0.452 | 0.507 |  | 0.00 |
| semantic_entropy | agreement | logprobs | 12 | 0.776 | [0.713, 0.827] | tied | -0.049 [-0.131, +0.033] | 0.455 | 0.505 |  | 0.26 |
| num_semantic_sets | agreement | text_only | 12 | 0.775 | [0.716, 0.830] | tied | -0.050 [-0.133, +0.032] | 0.453 | 0.504 |  | 0.00 |
| p_true_empirical | self_eval | text_only | 11 | 0.770 | [0.708, 0.826] | worse | -0.055 [-0.092, -0.019] | 0.467 | 0.472 | 0.314 | 2.25 |
| greedy_cluster_share | agreement | text_only | 12 | 0.770 | [0.705, 0.826] | tied | -0.055 [-0.142, +0.025] | 0.455 | 0.501 |  | 0.00 |
| lexical_similarity | agreement | text_only | 11 | 0.752 | [0.680, 0.817] | tied | -0.073 [-0.167, +0.006] | 0.460 | 0.479 |  | 0.00 |
| msp | baseline | logprobs | 1 | 0.748 | [0.673, 0.817] | tied | -0.077 [-0.175, +0.019] | 0.462 | 0.552 |  | 0.00 |
| eccentricity | agreement | text_only | 12 | 0.748 | [0.674, 0.813] | tied | -0.077 [-0.169, +0.009] | 0.469 | 0.460 |  | 0.00 |
| verbalized_2s | self_eval | text_only | 2 | 0.538 | [0.488, 0.587] | worse | -0.287 [-0.366, -0.216] | 0.606 | 0.081 | 0.563 | 0.56 |
| dummy_random | baseline | text_only | 1 | 0.521 | [0.439, 0.610] | worse | -0.304 [-0.399, -0.204] | 0.597 | 0.131 |  | 0.00 |

## qwen3b/triviaqa_200

Accuracy 0.485, float32, mps.

| method | family | access | cost | AUROC | 95 % interval | vs best | Δ AUROC [paired 95 %] | AURC | PRR | ECE | s / question (method) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p_true_zeroshot ★ | self_eval | logprobs | 2 | 0.877 | [0.830, 0.921] | best |  | 0.262 | 0.793 | 0.176 | 0.24 |
| p_true_empirical | self_eval | text_only | 11 | 0.830 | [0.775, 0.880] | worse | -0.047 [-0.081, -0.012] | 0.298 | 0.650 | 0.181 | 3.16 |
| p_true_brainstorm | self_eval | logprobs | 12 | 0.820 | [0.764, 0.873] | worse | -0.057 [-0.107, -0.012] | 0.299 | 0.689 | 0.259 | 0.47 |
| mean_token_entropy | baseline | white_box | 1 | 0.817 | [0.757, 0.871] | worse | -0.061 [-0.123, -0.000] | 0.299 | 0.684 |  | 0.00 |
| likelihood_mean | baseline | logprobs | 1 | 0.786 | [0.719, 0.843] | worse | -0.091 [-0.156, -0.027] | 0.321 | 0.631 |  | 0.00 |
| discrete_semantic_entropy | agreement | text_only | 12 | 0.765 | [0.703, 0.823] | worse | -0.112 [-0.173, -0.052] | 0.353 | 0.467 |  | 0.00 |
| semantic_entropy | agreement | logprobs | 12 | 0.761 | [0.699, 0.819] | worse | -0.116 [-0.176, -0.057] | 0.355 | 0.463 |  | 0.13 |
| greedy_cluster_share | agreement | text_only | 12 | 0.761 | [0.697, 0.818] | worse | -0.117 [-0.178, -0.056] | 0.354 | 0.464 |  | 0.00 |
| largest_cluster_share | agreement | text_only | 12 | 0.758 | [0.696, 0.817] | worse | -0.119 [-0.183, -0.059] | 0.355 | 0.460 |  | 0.00 |
| lexical_similarity | agreement | text_only | 11 | 0.757 | [0.692, 0.816] | worse | -0.121 [-0.189, -0.056] | 0.363 | 0.457 |  | 0.00 |
| msp | baseline | logprobs | 1 | 0.753 | [0.682, 0.818] | worse | -0.124 [-0.195, -0.055] | 0.342 | 0.573 |  | 0.00 |
| verbalized_2s | self_eval | text_only | 2 | 0.713 | [0.659, 0.766] | worse | -0.164 [-0.223, -0.103] | 0.395 | 0.350 | 0.412 | 0.77 |
| dummy_random | baseline | text_only | 1 | 0.500 | [0.417, 0.584] | worse | -0.378 [-0.473, -0.286] | 0.504 | 0.093 |  | 0.00 |

## qwen7b/triviaqa_200

Accuracy 0.590, float16, mps.

| method | family | access | cost | AUROC | 95 % interval | vs best | Δ AUROC [paired 95 %] | AURC | PRR | ECE | s / question (method) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p_true_zeroshot | self_eval | logprobs | 2 | 0.870 | [0.815, 0.920] | best |  | 0.189 | 0.767 | 0.220 | 0.34 |
| p_true_brainstorm | self_eval | logprobs | 12 | 0.845 | [0.785, 0.896] | tied | -0.025 [-0.075, +0.019] | 0.196 | 0.693 | 0.255 | 0.68 |
| mean_token_entropy ★ | baseline | white_box | 1 | 0.833 | [0.772, 0.885] | tied | -0.037 [-0.104, +0.026] | 0.204 | 0.724 |  | 0.00 |
| msp | baseline | logprobs | 1 | 0.824 | [0.764, 0.877] | tied | -0.046 [-0.115, +0.026] | 0.207 | 0.719 |  | 0.00 |
| p_true_empirical | self_eval | text_only | 11 | 0.824 | [0.763, 0.886] | worse | -0.046 [-0.084, -0.008] | 0.218 | 0.633 | 0.171 | 5.93 |
| likelihood_mean | baseline | logprobs | 1 | 0.817 | [0.752, 0.871] | tied | -0.053 [-0.120, +0.015] | 0.209 | 0.704 |  | 0.00 |
| lexical_similarity | agreement | text_only | 11 | 0.735 | [0.668, 0.800] | worse | -0.136 [-0.213, -0.052] | 0.275 | 0.418 |  | 0.00 |
| verbalized_2s | self_eval | text_only | 2 | 0.734 | [0.672, 0.796] | worse | -0.137 [-0.202, -0.070] | 0.279 | 0.409 | 0.320 | 0.91 |
| discrete_semantic_entropy | agreement | text_only | 12 | 0.729 | [0.667, 0.789] | worse | -0.141 [-0.215, -0.071] | 0.275 | 0.391 |  | 0.00 |
| semantic_entropy | agreement | logprobs | 12 | 0.729 | [0.667, 0.790] | worse | -0.141 [-0.215, -0.070] | 0.274 | 0.391 |  | 0.06 |
| largest_cluster_share | agreement | text_only | 12 | 0.728 | [0.664, 0.787] | worse | -0.143 [-0.219, -0.072] | 0.275 | 0.390 |  | 0.00 |
| greedy_cluster_share | agreement | text_only | 12 | 0.718 | [0.651, 0.777] | worse | -0.152 [-0.229, -0.078] | 0.279 | 0.374 |  | 0.00 |
| dummy_random | baseline | text_only | 1 | 0.560 | [0.477, 0.637] | worse | -0.311 [-0.414, -0.208] | 0.372 | 0.168 |  | 0.00 |

## smollm2-1.7b/triviaqa_200

Accuracy 0.360, float32, mps.

| method | family | access | cost | AUROC | 95 % interval | vs best | Δ AUROC [paired 95 %] | AURC | PRR | ECE | s / question (method) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean_token_entropy ★ | baseline | white_box | 1 | 0.816 | [0.741, 0.879] | best |  | 0.424 | 0.694 |  | 0.00 |
| likelihood_mean | baseline | logprobs | 1 | 0.809 | [0.738, 0.871] | tied | -0.007 [-0.030, +0.015] | 0.423 | 0.685 |  | 0.00 |
| lexical_similarity | agreement | text_only | 11 | 0.796 | [0.726, 0.854] | tied | -0.020 [-0.077, +0.047] | 0.445 | 0.585 |  | 0.00 |
| msp | baseline | logprobs | 1 | 0.795 | [0.726, 0.855] | tied | -0.021 [-0.085, +0.044] | 0.431 | 0.671 |  | 0.00 |
| dummy_random | baseline | text_only | 1 | 0.525 | [0.440, 0.612] | worse | -0.292 [-0.406, -0.172] | 0.613 | 0.098 |  | 0.00 |

## qwen0.5b/triviaqa_1000

Accuracy 0.207, float32, cpu.

| method | family | access | cost | AUROC | 95 % interval | vs best | Δ AUROC [paired 95 %] | AURC | PRR | ECE | s / question (method) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eig_val_laplacian ★ | agreement | text_only | 12 | 0.752 | [0.712, 0.794] | best |  | 0.663 | 0.498 |  | 0.00 |
| deg_mat | agreement | text_only | 12 | 0.743 | [0.704, 0.785] | worse | -0.009 [-0.016, -0.001] | 0.668 | 0.478 |  | 0.00 |
| num_semantic_sets | agreement | text_only | 12 | 0.735 | [0.693, 0.778] | worse | -0.017 [-0.028, -0.006] | 0.669 | 0.460 |  | 0.00 |
| discrete_semantic_entropy | agreement | text_only | 12 | 0.732 | [0.688, 0.774] | worse | -0.021 [-0.032, -0.009] | 0.671 | 0.458 |  | 0.00 |
| semantic_entropy | agreement | logprobs | 12 | 0.730 | [0.687, 0.773] | worse | -0.022 [-0.036, -0.009] | 0.671 | 0.456 |  | 1.25 |
| likelihood_mean | baseline | logprobs | 1 | 0.712 | [0.672, 0.747] | worse | -0.040 [-0.085, -0.000] | 0.689 | 0.401 |  | 0.00 |
| eccentricity | agreement | text_only | 12 | 0.710 | [0.667, 0.752] | worse | -0.042 [-0.062, -0.024] | 0.684 | 0.426 |  | 0.00 |
| mean_token_entropy | baseline | white_box | 1 | 0.696 | [0.656, 0.735] | worse | -0.056 [-0.104, -0.012] | 0.698 | 0.372 |  | 0.00 |
| lexical_similarity | agreement | text_only | 11 | 0.681 | [0.642, 0.717] | worse | -0.072 [-0.112, -0.033] | 0.714 | 0.325 |  | 0.00 |
| msp | baseline | logprobs | 1 | 0.639 | [0.595, 0.683] | worse | -0.113 [-0.150, -0.076] | 0.720 | 0.302 |  | 0.00 |
| dummy_random | baseline | text_only | 1 | 0.515 | [0.465, 0.561] | worse | -0.237 [-0.298, -0.174] | 0.780 | 0.065 |  | 0.00 |

## qwen1.5b/triviaqa_1000

Accuracy 0.370, float32, cpu.

| method | family | access | cost | AUROC | 95 % interval | vs best | Δ AUROC [paired 95 %] | AURC | PRR | ECE | s / question (method) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean_token_entropy ★ | baseline | white_box | 1 | 0.799 | [0.770, 0.827] | best |  | 0.436 | 0.632 |  | 0.00 |
| likelihood_mean | baseline | logprobs | 1 | 0.789 | [0.759, 0.818] | tied | -0.010 [-0.020, +0.000] | 0.444 | 0.605 |  | 0.00 |
| eig_val_laplacian | agreement | text_only | 12 | 0.777 | [0.746, 0.806] | tied | -0.021 [-0.053, +0.010] | 0.449 | 0.525 |  | 0.00 |
| deg_mat | agreement | text_only | 12 | 0.775 | [0.746, 0.804] | tied | -0.024 [-0.055, +0.007] | 0.451 | 0.520 |  | 0.00 |
| discrete_semantic_entropy | agreement | text_only | 12 | 0.775 | [0.745, 0.803] | tied | -0.024 [-0.053, +0.006] | 0.452 | 0.506 |  | 0.00 |
| lexical_similarity | agreement | text_only | 11 | 0.773 | [0.743, 0.802] | tied | -0.026 [-0.056, +0.003] | 0.454 | 0.514 |  | 0.00 |
| num_semantic_sets | agreement | text_only | 12 | 0.772 | [0.743, 0.801] | tied | -0.026 [-0.055, +0.002] | 0.453 | 0.503 |  | 0.00 |
| semantic_entropy | agreement | logprobs | 12 | 0.772 | [0.742, 0.801] | tied | -0.027 [-0.057, +0.003] | 0.454 | 0.503 |  | 0.41 |
| eccentricity | agreement | text_only | 12 | 0.753 | [0.721, 0.785] | worse | -0.045 [-0.078, -0.011] | 0.464 | 0.491 |  | 0.00 |
| msp | baseline | logprobs | 1 | 0.739 | [0.706, 0.771] | worse | -0.060 [-0.093, -0.028] | 0.465 | 0.547 |  | 0.00 |
| dummy_random | baseline | text_only | 1 | 0.503 | [0.467, 0.540] | worse | -0.296 [-0.343, -0.252] | 0.619 | 0.047 |  | 0.00 |

## qwen3b/triviaqa_1000

Accuracy 0.445, float32, mps.

| method | family | access | cost | AUROC | 95 % interval | vs best | Δ AUROC [paired 95 %] | AURC | PRR | ECE | s / question (method) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean_token_entropy ★ | baseline | white_box | 1 | 0.816 | [0.790, 0.841] | best |  | 0.342 | 0.668 |  | 0.00 |
| likelihood_mean | baseline | logprobs | 1 | 0.798 | [0.772, 0.824] | worse | -0.018 [-0.027, -0.009] | 0.354 | 0.637 |  | 0.00 |
| msp | baseline | logprobs | 1 | 0.769 | [0.740, 0.799] | worse | -0.047 [-0.070, -0.023] | 0.362 | 0.606 |  | 0.00 |
| lexical_similarity | agreement | text_only | 11 | 0.754 | [0.726, 0.783] | worse | -0.062 [-0.085, -0.038] | 0.397 | 0.452 |  | 0.00 |
| dummy_random | baseline | text_only | 1 | 0.460 | [0.425, 0.495] | worse | -0.356 [-0.400, -0.313] | 0.573 | -0.049 |  | 0.00 |

## qwen0.5b/mmlu_200

Accuracy 0.195, float32, cpu.

| method | family | access | cost | AUROC | 95 % interval | vs best | Δ AUROC [paired 95 %] | AURC | PRR | ECE | s / question (method) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean_token_entropy | baseline | white_box | 1 | 0.580 | [0.471, 0.685] | best |  | 0.759 | 0.203 |  | 0.00 |
| likelihood_mean | baseline | logprobs | 1 | 0.543 | [0.436, 0.648] | tied | -0.037 [-0.088, +0.013] | 0.776 | 0.168 |  | 0.00 |
| num_semantic_sets | agreement | text_only | 12 | 0.533 | [0.450, 0.620] | tied | -0.047 [-0.126, +0.040] | 0.782 | 0.084 |  | 0.00 |
| eig_val_laplacian | agreement | text_only | 12 | 0.522 | [0.426, 0.620] | tied | -0.057 [-0.151, +0.028] | 0.792 | 0.070 |  | 0.00 |
| eccentricity | agreement | text_only | 12 | 0.518 | [0.419, 0.620] | tied | -0.062 [-0.158, +0.034] | 0.791 | 0.065 |  | 0.00 |
| discrete_semantic_entropy | agreement | text_only | 12 | 0.517 | [0.418, 0.617] | tied | -0.063 [-0.152, +0.023] | 0.792 | 0.073 |  | 0.00 |
| dummy_random | baseline | text_only | 1 | 0.516 | [0.420, 0.611] | tied | -0.064 [-0.224, +0.092] | 0.808 | -0.062 |  | 0.00 |
| lexical_similarity | agreement | text_only | 11 | 0.507 | [0.403, 0.611] | tied | -0.072 [-0.168, +0.016] | 0.794 | 0.068 |  | 0.00 |
| semantic_entropy | agreement | logprobs | 12 | 0.504 | [0.398, 0.609] | tied | -0.075 [-0.162, +0.011] | 0.796 | 0.062 |  | 0.04 |
| deg_mat | agreement | text_only | 12 | 0.503 | [0.400, 0.608] | tied | -0.077 [-0.173, +0.010] | 0.795 | 0.053 |  | 0.00 |
| msp | baseline | logprobs | 1 | 0.501 | [0.394, 0.605] | worse | -0.079 [-0.163, -0.003] | 0.798 | 0.094 |  | 0.00 |

## qwen1.5b/mmlu_200

Accuracy 0.360, float32, cpu.

| method | family | access | cost | AUROC | 95 % interval | vs best | Δ AUROC [paired 95 %] | AURC | PRR | ECE | s / question (method) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dummy_random | baseline | text_only | 1 | 0.553 | [0.469, 0.638] | best |  | 0.610 | 0.153 |  | 0.00 |
| eccentricity | agreement | text_only | 12 | 0.550 | [0.481, 0.625] | tied | -0.003 [-0.110, +0.099] | 0.617 | 0.076 |  | 0.00 |
| num_semantic_sets | agreement | text_only | 12 | 0.526 | [0.457, 0.598] | tied | -0.027 [-0.129, +0.074] | 0.628 | 0.052 |  | 0.00 |
| lexical_similarity | agreement | text_only | 11 | 0.526 | [0.451, 0.602] | tied | -0.027 [-0.133, +0.072] | 0.625 | 0.055 |  | 0.00 |
| eig_val_laplacian | agreement | text_only | 12 | 0.524 | [0.451, 0.597] | tied | -0.029 [-0.133, +0.068] | 0.624 | 0.054 |  | 0.00 |
| deg_mat | agreement | text_only | 12 | 0.524 | [0.449, 0.598] | tied | -0.029 [-0.135, +0.070] | 0.626 | 0.054 |  | 0.00 |
| discrete_semantic_entropy | agreement | text_only | 12 | 0.520 | [0.446, 0.597] | tied | -0.033 [-0.139, +0.067] | 0.630 | 0.047 |  | 0.00 |
| semantic_entropy | agreement | logprobs | 12 | 0.520 | [0.447, 0.597] | tied | -0.033 [-0.138, +0.066] | 0.631 | 0.047 |  | 0.02 |
| mean_token_entropy | baseline | white_box | 1 | 0.508 | [0.421, 0.594] | tied | -0.045 [-0.175, +0.075] | 0.638 | 0.007 |  | 0.00 |
| likelihood_mean | baseline | logprobs | 1 | 0.505 | [0.419, 0.587] | tied | -0.048 [-0.167, +0.071] | 0.646 | 0.006 |  | 0.00 |
| msp | baseline | logprobs | 1 | 0.505 | [0.419, 0.587] | tied | -0.048 [-0.167, +0.071] | 0.646 | 0.006 |  | 0.00 |

## qwen3b/mmlu_200

Accuracy 0.635, float32, cpu.

| method | family | access | cost | AUROC | 95 % interval | vs best | Δ AUROC [paired 95 %] | AURC | PRR | ECE | s / question (method) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean_token_entropy ★ | baseline | white_box | 1 | 0.769 | [0.702, 0.834] | best |  | 0.208 | 0.592 |  | 0.00 |
| msp | baseline | logprobs | 1 | 0.768 | [0.700, 0.834] | tied | -0.001 [-0.005, +0.003] | 0.209 | 0.587 |  | 0.00 |
| likelihood_mean | baseline | logprobs | 1 | 0.768 | [0.700, 0.834] | tied | -0.001 [-0.005, +0.003] | 0.209 | 0.587 |  | 0.00 |
| lexical_similarity | agreement | text_only | 11 | 0.601 | [0.553, 0.650] | worse | -0.168 [-0.233, -0.106] | 0.318 | 0.172 |  | 0.00 |
| dummy_random | baseline | text_only | 1 | 0.551 | [0.468, 0.629] | worse | -0.218 [-0.330, -0.111] | 0.333 | 0.150 |  | 0.00 |

## smollm2-1.7b/mmlu_200

Accuracy 0.450, float32, cpu.

| method | family | access | cost | AUROC | 95 % interval | vs best | Δ AUROC [paired 95 %] | AURC | PRR | ECE | s / question (method) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| msp | baseline | logprobs | 1 | 0.627 | [0.546, 0.702] | best |  | 0.452 | 0.366 |  | 0.00 |
| lexical_similarity | agreement | text_only | 11 | 0.625 | [0.548, 0.701] | tied | -0.002 [-0.064, +0.067] | 0.460 | 0.289 |  | 0.00 |
| likelihood_mean | baseline | logprobs | 1 | 0.583 | [0.510, 0.668] | tied | -0.044 [-0.135, +0.058] | 0.476 | 0.310 |  | 0.00 |
| mean_token_entropy | baseline | white_box | 1 | 0.564 | [0.488, 0.650] | tied | -0.063 [-0.164, +0.050] | 0.498 | 0.221 |  | 0.00 |
| dummy_random | baseline | text_only | 1 | 0.527 | [0.440, 0.605] | tied | -0.100 [-0.214, +0.017] | 0.523 | 0.103 |  | 0.00 |
