# LLM-probly demo (F3)

Regenerates the README demo block, a demo page (`DEMO.md`) and two SVG cards (`assets/`) from the selector
profiles in `experiments/llm-probly/profiles/` (D4). Standard library only.

```bash
# from the repository root: write DEMO.md and the SVG cards
uv run python experiments/llm-probly/demo/build_demo.py

# also update the README between the demo markers
uv run python experiments/llm-probly/demo/build_demo.py --readme README.md
```

Rerun it whenever the profiles change; never edit the generated files by hand.
