# Sample-count curve: `qwen0.5b/triviaqa_200`

NLI setup: `deberta-large`. AUROC (mean over random subsets of the 10 cached samples; std in brackets).

| method | K=2 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 | K=9 | K=10 |
|---|---|---|---|---|---|---|---|---|---|
| SE | 0.638 (0.022) | 0.696 (0.031) | 0.720 (0.017) | 0.738 (0.017) | 0.737 (0.013) | 0.740 (0.011) | 0.746 (0.012) | 0.745 (0.005) | 0.744 (0.000) |
| discrete SE | 0.672 (0.026) | 0.715 (0.032) | 0.735 (0.016) | 0.748 (0.016) | 0.744 (0.013) | 0.747 (0.011) | 0.754 (0.012) | 0.752 (0.006) | 0.751 (0.000) |
| lexical similarity | 0.693 (0.028) | 0.714 (0.023) | 0.733 (0.015) | 0.748 (0.008) | 0.745 (0.008) | 0.745 (0.008) | 0.753 (0.008) | 0.755 (0.005) | 0.757 (0.000) |

## Early stop

Rule: generate `check_at` samples; stop if they all fall into one semantic cluster,
otherwise generate all 10. AUROC change is against all 10 samples.

Accuracy columns: share of correct greedy answers among the stopped and the not-stopped
questions. A stopped wrong answer gets entropy 0 and ties with the stopped correct ones,
which is where most of the AUROC change comes from.

| method | check_at | stopped | accuracy stopped | accuracy not stopped | samples / question | saved generations | AUROC | AUROC change |
|---|---|---|---|---|---|---|---|---|
| SE | 2 | 36% | 32% | 12% | 7.08 | 29% | 0.661 | -0.083 |
| discrete SE | 2 | 36% | 32% | 12% | 7.08 | 29% | 0.667 | -0.084 |
| lexical similarity | 2 | 36% | 32% | 12% | 7.08 | 29% | 0.676 | -0.080 |
| SE | 3 | 26% | 42% | 11% | 8.18 | 18% | 0.710 | -0.034 |
| discrete SE | 3 | 26% | 42% | 11% | 8.18 | 18% | 0.717 | -0.034 |
| lexical similarity | 3 | 26% | 42% | 11% | 8.18 | 18% | 0.738 | -0.019 |
| SE | 4 | 21% | 50% | 11% | 8.74 | 13% | 0.729 | -0.014 |
| discrete SE | 4 | 21% | 50% | 11% | 8.74 | 13% | 0.736 | -0.014 |
| lexical similarity | 4 | 21% | 50% | 11% | 8.74 | 13% | 0.762 | +0.005 |
| SE | 5 | 18% | 54% | 11% | 9.07 | 9% | 0.736 | -0.007 |
| discrete SE | 5 | 18% | 54% | 11% | 9.07 | 9% | 0.743 | -0.008 |
| lexical similarity | 5 | 18% | 54% | 11% | 9.07 | 9% | 0.765 | +0.008 |
