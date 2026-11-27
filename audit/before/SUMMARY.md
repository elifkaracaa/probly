# Audit before cleanup (Day 1)

Commit: 33cc81ac (2026-11-24, "Merge pull request #31") - main before any cleanup change
Measured: 2026-11-27 by Elçin (T5), MacBook Pro (macOS, zsh)

| Metric | Plan (25 Nov) | Measured | Raw file |
|---|---|---|---|
| Pre-commit (Ruff, ty, backend naming) | Pass | Pass | prek.txt |
| Excluded from Ruff and ty | 2 files + 2 folders | 2 files + 2 folders | lint_exclusions.txt |
| Experiment tests | 304 passed, 1 skipped, 65 s | 304 passed, 1 skipped, 50.7 s | tests.txt |
| Experiment tests run in CI | no | no | - |
| Coverage of imported code | 86% (5,273 stmts) | 86% (5,273 stmts, 716 missed) | tests.txt |
| Scripts without any test | 12 | 12 | untested_scripts.txt |
| Public functions without docstring (D103) | 10 | 10 | docstrings.txt |
| noqa / type: ignore comments | 50 | 49 | noqa.txt |
| Role codes in .py and .md files | 277 | 277 in code and docs (378 incl. archive/ and results/) | role_codes.txt |
| Open / stale merged branches | 13 / 35 | 12 / 33 | branches_open.txt, branches_merged.txt |
| p_true_empirical s/question (1.5B, TriviaQA 200) | 2.25 | 2.25 (10 model calls) | results/qwen1.5b/triviaqa_200/p_true_empirical.json |
| README lines | 311 | 311 | readme_lines.txt |

## Notes
- Role codes: counted with `git grep` (tracked files only). The plan's count leaves out
  archive/ and results/; we get exactly 277 with the same scope. archive/ will be deleted,
  results/ files are generated reports. For "after", report both numbers.
- noqa: 49 vs 50 in the plan; same commit, difference not explained yet (likely counting scope).
- Docstrings: `ruff --isolated` also reports rules that Google style does not use
  (D413, D401, D400). Only D103 counts.
- Branches: 12 open instead of 13. Which branch is missing? Check before deleting anything.
  The merged count excludes origin/main itself.
- Test runtime depends on the machine: measure "after" on the same machine.
- macOS / zsh: quote globs in the audit commands (`--include='*.py'`), otherwise zsh fails.
- Not measured today: non-English files (manual check, T3), steps from clone to first
  result (Day 3 fresh-README test).
