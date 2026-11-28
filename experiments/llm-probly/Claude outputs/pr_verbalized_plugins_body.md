## What

Tests for the empirical p(True) and verbalized-confidence plugins (`methods/p_true_empirical.py`, `methods/verbalized_2s.py`) and for `verbalized_2s()` in `common/verbalized.py`.

## Why

The `score()` functions of both plugins and the model call in `common/verbalized.py` were never run in the tests (coverage 36%, 37% and 77%). Only the reply parsing was tested.

## How

The sampler and the evaluator are replaced by small stubs with fixed replies, so no model is loaded and nothing is downloaded. The tests check:

- `verbalized_2s()`: reads the number in the reply, an unreadable reply gives confidence 0 and `failed`, the chat has the question, the answer and the ask prompt, decoding is greedy, and `ask` replaces the default prompt
- verbalized plugin: uncertainty is `1 - confidence`, unreadable replies give 1.0 and `failed`, a stated 0 is not a failure, result fields
- empirical plugin: uncertainty is `1 - share of True judgments`, `parsed_fraction`, a question with no readable reply gives 1.0 and `failed`, result fields including device, dtype and implementation
- both plugins raise `ValueError` without a model

## Result

9 tests pass. `p_true_empirical.py` and `verbalized_2s.py` go to 100% coverage (from 36% and 37%).
