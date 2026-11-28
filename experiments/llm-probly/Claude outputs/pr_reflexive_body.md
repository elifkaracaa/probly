## What

More tests for the week-1 reference p(True) in `common/reflexive.py`, and its docstrings now name the methods (zero-shot, brainstorm) instead of using role codes.

## Why

The module was at 87% coverage. Not tested: the chat-template prompt, the error for an unknown mode and `true_logit`. It is the reference that the library implementation is checked against, so its prompts should be pinned down.

## How

The existing fake tokenizer and model are reused, so nothing is downloaded. New tests check:

- `build_prompt`: `plain_ab` ends with an open bracket, `chat_groups` goes through the chat template, an unknown mode raises `ValueError`
- `build_text`: the brainstorm line appears only when ideas are given
- `true_logit`: returns the logit of the single token "True"

The only change in `common/reflexive.py` is wording in docstrings (no behavior change).

## Result

5 new tests pass. Coverage of `common/reflexive.py` goes from 87% to 100%.
