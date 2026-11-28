## What

Tests for the shared sanity test (`common/sanity.py`), and its docstring now names the methods instead of using role codes.

## Why

`common/sanity.py` had 0% coverage. It is the check that a reflexive method gives clearly higher confidence to correct answers than to wrong ones, and its results are read by `sanity_table`, so it should be tested.

## How

Small fake confidence functions replace the model, so nothing is downloaded. The tests check:

- the question set has a right and a different wrong answer per question
- a perfect method gets AUROC 1.0 and a clear gap, a constant method 0.5 and no gap, an inverted method 0.0 and a negative gap
- each question is asked twice, right answer first
- samples are passed per question, missing ones as `None`
- `save_sanity` creates the folder and writes `<method>_<short model name>.json` with the result

The only change in `common/sanity.py` is the first docstring line (no behavior change).

## Result

7 new tests pass. Coverage of `common/sanity.py` goes from 0% to 100%.
