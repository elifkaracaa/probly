# LLM-probly demo

Regenerates the README demo block, a demo page (`DEMO.md`) and two SVG cards (`assets/`) from the
selector profiles in `experiments/llm-probly/profiles/`. No model is loaded.

```bash
# from experiments/llm-probly: write DEMO.md and the SVG cards
uv run python -m demo.build_demo

# also update the README between the demo markers
uv run python -m demo.build_demo --readme README.md
```

Rerun it whenever the profiles change; never edit the generated files by hand.
