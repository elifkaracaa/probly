# triviaqa, 200 questions: methods x model size

## Models

| model | questions | accuracy | precision |
|---|---|---|---|
| qwen0.5b | 200 | 0.190 | float32 |
| qwen1.5b | 200 | 0.375 | float32 |
| qwen3b | 200 | 0.485 | float32 |
| qwen7b | 200 | 0.590 | float16 |
| smollm2-1.7b | 200 | 0.360 | float32 |


## AUROC (higher is better, 0.5 = random)

| method | qwen0.5b | qwen1.5b | qwen3b | qwen7b | smollm2-1.7b |
|---|---|---|---|---|---|
| cocoa_msp | 0.715 | 0.759 | 0.737 | 0.733 |  |
| cocoa_p_true | 0.734 | 0.833 | 0.786 | 0.752 |  |
| deg_mat | 0.751 | 0.779 |  |  |  |
| discrete_semantic_entropy | 0.751 | 0.779 | 0.765 | 0.729 |  |
| dummy_random | 0.567 | 0.521 | 0.500 | 0.560 | 0.525 |
| eccentricity | 0.750 | 0.748 |  |  |  |
| eig_val_laplacian | 0.758 | 0.784 |  |  |  |
| greedy_cluster_share | 0.758 | 0.770 | 0.761 | 0.718 |  |
| largest_cluster_share | 0.738 | 0.777 | 0.758 | 0.728 |  |
| lexical_similarity | 0.757 | 0.752 | 0.757 | 0.735 | 0.796 |
| likelihood_mean | 0.791 | 0.790 | 0.786 | 0.817 | 0.809 |
| mean_token_entropy | 0.773 | 0.805 | 0.817 | 0.833 | 0.816 |
| msp | 0.680 | 0.748 | 0.753 | 0.824 | 0.795 |
| num_semantic_sets | 0.759 | 0.775 |  |  |  |
| p_true_brainstorm | 0.685 | 0.806 | 0.820 | 0.845 |  |
| p_true_empirical | 0.518 | 0.770 | 0.830 | 0.824 |  |
| p_true_zeroshot | 0.669 | 0.825 | 0.877 | 0.870 |  |
| semantic_entropy | 0.744 | 0.776 | 0.761 | 0.729 |  |
| verbalized_2s | 0.584 | 0.538 | 0.713 | 0.734 |  |


## PRR (higher is better, 1 = perfect, 0 = random)

| method | qwen0.5b | qwen1.5b | qwen3b | qwen7b | smollm2-1.7b |
|---|---|---|---|---|---|
| cocoa_msp | 0.465 | 0.481 | 0.433 | 0.414 |  |
| cocoa_p_true | 0.468 | 0.571 | 0.490 | 0.432 |  |
| deg_mat | 0.514 | 0.499 |  |  |  |
| discrete_semantic_entropy | 0.520 | 0.508 | 0.467 | 0.391 |  |
| dummy_random | 0.160 | 0.131 | 0.093 | 0.168 | 0.098 |
| eccentricity | 0.513 | 0.460 |  |  |  |
| eig_val_laplacian | 0.530 | 0.506 |  |  |  |
| greedy_cluster_share | 0.517 | 0.501 | 0.464 | 0.374 |  |
| largest_cluster_share | 0.500 | 0.507 | 0.460 | 0.390 |  |
| lexical_similarity | 0.467 | 0.479 | 0.457 | 0.418 | 0.585 |
| likelihood_mean | 0.576 | 0.618 | 0.631 | 0.704 | 0.685 |
| mean_token_entropy | 0.548 | 0.653 | 0.684 | 0.724 | 0.694 |
| msp | 0.400 | 0.552 | 0.573 | 0.719 | 0.671 |
| num_semantic_sets | 0.530 | 0.504 |  |  |  |
| p_true_brainstorm | 0.291 | 0.577 | 0.689 | 0.693 |  |
| p_true_empirical | -0.000 | 0.472 | 0.650 | 0.633 |  |
| p_true_zeroshot | 0.275 | 0.650 | 0.793 | 0.767 |  |
| semantic_entropy | 0.514 | 0.505 | 0.463 | 0.391 |  |
| verbalized_2s | 0.090 | 0.081 | 0.350 | 0.409 |  |


## AURC (lower is better)

| method | qwen0.5b | qwen1.5b | qwen3b | qwen7b | smollm2-1.7b |
|---|---|---|---|---|---|
| cocoa_msp | 0.697 | 0.457 | 0.370 | 0.278 |  |
| cocoa_p_true | 0.704 | 0.423 | 0.347 | 0.272 |  |
| deg_mat | 0.686 | 0.449 |  |  |  |
| discrete_semantic_entropy | 0.680 | 0.452 | 0.353 | 0.275 |  |
| dummy_random | 0.771 | 0.597 | 0.504 | 0.372 | 0.613 |
| eccentricity | 0.681 | 0.469 |  |  |  |
| eig_val_laplacian | 0.674 | 0.441 |  |  |  |
| greedy_cluster_share | 0.679 | 0.455 | 0.354 | 0.279 |  |
| largest_cluster_share | 0.688 | 0.452 | 0.355 | 0.275 |  |
| lexical_similarity | 0.692 | 0.460 | 0.363 | 0.275 | 0.445 |
| likelihood_mean | 0.671 | 0.433 | 0.321 | 0.209 | 0.423 |
| mean_token_entropy | 0.680 | 0.421 | 0.299 | 0.204 | 0.424 |
| msp | 0.719 | 0.462 | 0.342 | 0.207 | 0.431 |
| num_semantic_sets | 0.676 | 0.453 |  |  |  |
| p_true_brainstorm | 0.730 | 0.427 | 0.299 | 0.196 |  |
| p_true_empirical | 0.805 | 0.467 | 0.298 | 0.218 |  |
| p_true_zeroshot | 0.744 | 0.411 | 0.262 | 0.189 |  |
| semantic_entropy | 0.685 | 0.455 | 0.355 | 0.274 |  |
| verbalized_2s | 0.777 | 0.606 | 0.395 | 0.279 |  |


## ECE, only probability methods (lower is better)

| method | qwen0.5b | qwen1.5b | qwen3b | qwen7b | smollm2-1.7b |
|---|---|---|---|---|---|
| p_true_brainstorm | 0.662 | 0.389 | 0.259 | 0.255 |  |
| p_true_empirical | 0.651 | 0.314 | 0.181 | 0.171 |  |
| p_true_zeroshot | 0.690 | 0.335 | 0.176 | 0.220 |  |
| verbalized_2s | 0.722 | 0.563 | 0.412 | 0.320 |  |
