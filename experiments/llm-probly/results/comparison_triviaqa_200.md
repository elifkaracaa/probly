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
| discrete_semantic_entropy | 0.751 | 0.779 | 0.765 | 0.729 |  |
| dummy_random | 0.567 | 0.521 | 0.500 | 0.560 | 0.525 |
| lexical_similarity | 0.757 | 0.752 | 0.757 | 0.735 | 0.796 |
| likelihood_mean | 0.791 | 0.790 | 0.786 | 0.817 | 0.809 |
| mean_token_entropy | 0.773 | 0.805 | 0.817 | 0.833 | 0.816 |
| msp | 0.680 | 0.748 | 0.753 | 0.824 | 0.795 |
| p_true_brainstorm | 0.685 | 0.806 | 0.820 | 0.845 |  |
| p_true_empirical | 0.518 | 0.770 | 0.830 | 0.824 |  |
| p_true_zeroshot | 0.669 | 0.825 | 0.877 | 0.870 |  |
| semantic_entropy | 0.744 | 0.776 | 0.761 | 0.729 |  |
| verbalized_2s | 0.584 | 0.538 | 0.713 | 0.734 |  |


## PRR (higher is better, 1 = perfect, 0 = random)

| method | qwen0.5b | qwen1.5b | qwen3b | qwen7b | smollm2-1.7b |
|---|---|---|---|---|---|
| discrete_semantic_entropy | 0.520 | 0.508 | 0.467 | 0.391 |  |
| dummy_random | 0.160 | 0.131 | 0.093 | 0.168 | 0.098 |
| lexical_similarity | 0.467 | 0.479 | 0.457 | 0.418 | 0.585 |
| likelihood_mean | 0.576 | 0.618 | 0.631 | 0.704 | 0.685 |
| mean_token_entropy | 0.548 | 0.653 | 0.684 | 0.724 | 0.694 |
| msp | 0.400 | 0.552 | 0.573 | 0.719 | 0.671 |
| p_true_brainstorm | 0.291 | 0.577 | 0.689 | 0.693 |  |
| p_true_empirical | -0.000 | 0.472 | 0.650 | 0.633 |  |
| p_true_zeroshot | 0.275 | 0.650 | 0.793 | 0.767 |  |
| semantic_entropy | 0.514 | 0.505 | 0.463 | 0.391 |  |
| verbalized_2s | 0.090 | 0.081 | 0.350 | 0.409 |  |


## AURC (lower is better)

| method | qwen0.5b | qwen1.5b | qwen3b | qwen7b | smollm2-1.7b |
|---|---|---|---|---|---|
| discrete_semantic_entropy | 0.680 | 0.452 | 0.353 | 0.275 |  |
| dummy_random | 0.771 | 0.597 | 0.504 | 0.372 | 0.613 |
| lexical_similarity | 0.692 | 0.460 | 0.363 | 0.275 | 0.445 |
| likelihood_mean | 0.671 | 0.433 | 0.321 | 0.209 | 0.423 |
| mean_token_entropy | 0.680 | 0.421 | 0.299 | 0.204 | 0.424 |
| msp | 0.719 | 0.462 | 0.342 | 0.207 | 0.431 |
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
