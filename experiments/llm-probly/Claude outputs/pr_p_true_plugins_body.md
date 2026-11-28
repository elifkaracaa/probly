## What

Tests for the zero-shot and brainstorm p(True) plugins (`methods/p_true_zeroshot.py`, `methods/p_true_brainstorm.py`).

## Why

Both plugins were only at 47% and 35% coverage: their `score()` functions were never run in the tests. The library class `HFSelfEvaluator` is already tested separately.

## How

`HFSelfEvaluator` is replaced by a small stub with fixed probabilities, so no model is loaded and nothing is downloaded. The tests check what the plugins add on top of the library:

- both raise `ValueError` without a model and tokenizer
- uncertainty is `1 - p_true`, and the result fields (method, model, answer, probability, ids, coverage, seconds) are right
- a NaN probability gives uncertainty 1.0 and `failed = True`, while `p_true = 0` stays valid
- brainstorm passes the cached samples to the evaluator
- brainstorm rows without samples (`None` or empty) are failed, get coverage NaN and are not sent to the evaluator

## Result

7 tests pass. Coverage of both files goes to 100% (from 47% and 35%).
