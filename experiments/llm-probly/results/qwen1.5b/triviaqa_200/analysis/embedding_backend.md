# Embedding backend vs NLI: `qwen1.5b/triviaqa_200`

Embedding model `sentence-transformers/all-mpnet-base-v2`; threshold chosen on `qwen0.5b`: **0.9** (held-out model).

## Speed (this machine, same questions)

| backend | seconds per question | speed-up |
|---|---|---|
| NLI (deberta-large, A2 + A3) | 0.303 | 1x |
| embedding | 0.011 | 26x |

## AUROC

| backend | semantic_entropy | discrete_semantic_entropy | eig_val_laplacian | deg_mat | num_semantic_sets | mean clusters | same clusters as NLI |
|---|---|---|---|---|---|---|---|
| NLI (deberta-large) | 0.776 | 0.779 | 0.784 | 0.779 | 0.775 | 3.08 |  |
| embedding, t = 0.6 | 0.715 | 0.716 | 0.754 | 0.752 | 0.709 | 2.30 | 44% |
| embedding, t = 0.65 | 0.729 | 0.730 | 0.754 | 0.752 | 0.724 | 2.50 | 48% |
| embedding, t = 0.7 | 0.738 | 0.742 | 0.753 | 0.752 | 0.736 | 2.65 | 47% |
| embedding, t = 0.75 | 0.746 | 0.751 | 0.753 | 0.752 | 0.745 | 2.85 | 49% |
| embedding, t = 0.8 | 0.750 | 0.752 | 0.753 | 0.750 | 0.753 | 3.05 | 50% |
| embedding, t = 0.85 | 0.751 | 0.755 | 0.753 | 0.751 | 0.751 | 3.29 | 52% |
| embedding, t = 0.9 | 0.744 | 0.749 | 0.752 | 0.751 | 0.748 | 3.60 | 53% |
| embedding, t = 0.95 | 0.740 | 0.746 | 0.751 | 0.750 | 0.744 | 3.88 | 49% |
