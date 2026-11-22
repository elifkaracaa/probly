# NLI setups (A4)

TriviaQA questions per model: {'qwen0.5b': 200, 'qwen1.5b': 200}. Agreement is weighted to the real mix of answer pairs (see `common/nli_labels.weighted_agreement`). "SE vs reference" is the paired bootstrap difference of SE AUROC against deberta-base without the question (positive = better; verdict A means this setup is better, B the reference, tie = interval contains 0).

| setup | agreement with hand labels (probly rule) | (strict rule) | SE AUROC qwen0.5b | DSE AUROC qwen0.5b | SE vs reference qwen0.5b | s/question qwen0.5b | SE AUROC qwen1.5b | DSE AUROC qwen1.5b | SE vs reference qwen1.5b | s/question qwen1.5b |
|---|---|---|---|---|---|---|---|---|---|---|
| deberta-base | 0.936 | 0.873 | 0.740 [0.63-0.83] | 0.746 | reference | 0.22 | 0.781 [0.72-0.84] | 0.783 | reference | 0.07 |
| deberta-base+question | 0.959 | 0.908 | 0.754 [0.65-0.84] | 0.760 | +0.015 [-0.031, +0.065] tie | 0.46 | 0.772 [0.71-0.83] | 0.772 | -0.009 [-0.029, +0.009] tie | 0.19 |
| deberta-large | 0.971 | 0.875 | 0.744 [0.64-0.83] | 0.752 | +0.004 [-0.019, +0.027] tie | 0.71 | 0.776 [0.71-0.83] | 0.778 | -0.005 [-0.030, +0.018] tie | 0.23 |
| deberta-large+question | 0.938 | 0.900 | 0.772 [0.67-0.86] | 0.777 | +0.032 [-0.011, +0.082] tie | 1.38 | 0.781 [0.72-0.84] | 0.782 | -0.001 [-0.022, +0.022] tie | 0.55 |
