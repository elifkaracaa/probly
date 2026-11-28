# llm-probly vs LM-Polygraph

[LM-Polygraph](https://github.com/IINemo/lm-polygraph) is a large library and
benchmark of uncertainty methods for LLMs. llm-probly does not try to match its
number of methods. It adds a layer on top of probly: it tells you which method
to use for your model, what that method costs, and what the confidence means.

> Written on Day 4. PRs marked *(open)* are still in review.

## What we offer

| Feature | llm-probly | LM-Polygraph | Evidence |
|---|---|---|---|
| Picks the method for your model | Method selector + saved profile per model | Benchmarks methods, the user chooses | [#31](https://github.com/elifkaracaa/probly/pull/31) *(open)* |
| Cost per method | Registry stores family, access level and cost; report has a cost column | Rough low/high compute label per method | [#47](https://github.com/elifkaracaa/probly/pull/47), [#53](https://github.com/elifkaracaa/probly/pull/53) *(open)* |
| Generate once, score every method from the cache | Yes | – | [#43](https://github.com/elifkaracaa/probly/pull/43) |
| Fewer generations for semantic entropy | Sample-count curve + early-stop rule | – | [#44](https://github.com/elifkaracaa/probly/pull/44) *(open)* |
| Faster semantic entropy | One NLI matrix per question, dedup, batching | – | [#30](https://github.com/elifkaracaa/probly/pull/30), [#48](https://github.com/elifkaracaa/probly/pull/48), [#51](https://github.com/elifkaracaa/probly/pull/51) *(open)* |
| "Is method X better than Y?" with an interval | Paired bootstrap | – | [#37](https://github.com/elifkaracaa/probly/pull/37) |
| API returns answer, chosen method, confidence, cost | `POST /uncertainty` | – | [#39](https://github.com/elifkaracaa/probly/pull/39) *(open)* |
| Built into probly | probly's conventions, tests, docs | Standalone library | [#32](https://github.com/elifkaracaa/probly/pull/32), [#46](https://github.com/elifkaracaa/probly/pull/46) *(open)* |
| Demo and runnable example | Generated from the real profiles | Web demo | [#45](https://github.com/elifkaracaa/probly/pull/45), [#50](https://github.com/elifkaracaa/probly/pull/50) *(open)* |

## Where LM-Polygraph is ahead

| Area | LM-Polygraph | llm-probly |
|---|---|---|
| Number of methods | Many white-box and black-box methods ([list](https://pypi.org/project/lm-polygraph/)) | Smaller set ([registry](https://github.com/elifkaracaa/probly/pull/47)) |
| Claim-level uncertainty | Yes | Out of scope |
| Density / training-based methods | Yes (e.g. Mahalanobis, RDE) | Out of scope |
| Normalized [0, 1] confidence | Yes, several normalization methods | Calibration step in progress |
| API-hosted models | Any OpenAI-compatible service | Text-only methods ([#41](https://github.com/elifkaracaa/probly/pull/41) *(open)*) |
| Published benchmark | [TACL 2025](https://aclanthology.org/2025.tacl-1.11.pdf) | TriviaQA-1000 + MMLU, 0.5B to 7B + SmolLM2 ([#43](https://github.com/elifkaracaa/probly/pull/43)) |

## What we closed this week

| Gap | Evidence |
|---|---|
| Larger benchmark (TriviaQA 1000, MMLU, second model family) | [#35](https://github.com/elifkaracaa/probly/pull/35), [#36](https://github.com/elifkaracaa/probly/pull/36), [#43](https://github.com/elifkaracaa/probly/pull/43) |
| One command runs a config end to end | [#47](https://github.com/elifkaracaa/probly/pull/47) |
| Reflexive methods moved into the library, batched | [#46](https://github.com/elifkaracaa/probly/pull/46), [#49](https://github.com/elifkaracaa/probly/pull/49) *(open)* |
| Faster semantic entropy | [#30](https://github.com/elifkaracaa/probly/pull/30), [#48](https://github.com/elifkaracaa/probly/pull/48), [#51](https://github.com/elifkaracaa/probly/pull/51) *(open)* |
| Models behind an API | [#41](https://github.com/elifkaracaa/probly/pull/41) *(open)* |
