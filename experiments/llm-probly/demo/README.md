# LLM-probly demo

One command regenerates every demo output from saved results; no model is loaded.

- `build_demo.py`: the README demo block, a demo page (`DEMO.md`) and two SVG cards (`assets/`) from the
  selector profiles in `profiles/`.
- `animation.py`: the animated README demo (light and dark SVG): example questions with the scores of five
  methods, read from the cached answers and the score files.

```bash
# from experiments/llm-probly: write DEMO.md and the SVG cards
uv run python -m demo.build_demo

# also update the README between the demo markers and write the animation for two questions
uv run python -m demo.build_demo --readme README.md --animation-run qwen1.5b --animation-n 200 --animation-ids 48 169

# find good example questions for the animation
uv run python -m demo.animation --list --run qwen1.5b --n 200
```

Rerun it whenever the profiles or the results change; never edit the generated files by hand.
