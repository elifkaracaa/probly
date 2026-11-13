"""Run the whole benchmark chain for one config (PIPE).

Run from ``experiments/llm-probly``::

    uv run python -m scripts.run_benchmark --config configs/qwen0.5b_triviaqa_5.yaml

Steps, in order. Each step is skipped if its output file already exists
(use ``--force`` to recompute the method results):

1. generate  -> data/<model>/<dataset>_<n>.jsonl      (scripts/generate.py)
2. label     -> adds "correct" to that file           (scripts/label.py)
3. methods   -> results/<model>/<dataset>/<method>.json, one per plugin in methods/
4. evaluate  -> AUROC, AURC, bootstrap interval per method (common/summary.py)
5. table     -> results/<model>/<dataset>/summary.md

See CONTRACT.md for the plugin interface.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from common.io import RESULTS_ROOT, save_scores
from common.summary import summary_table

ROOT = Path(__file__).resolve().parent.parent  # experiments/llm-probly
DATA_DIR = ROOT / "data"
METHODS_DIR = ROOT / "methods"
SUPPORTED_DATASETS = ("triviaqa",)  # generate.py only knows TriviaQA for now


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a YAML file in configs/")
    parser.add_argument("--methods", help="comma-separated method names; overrides the config")
    parser.add_argument("--skip-model-methods", action="store_true", help="skip plugins with NEEDS_MODEL = True")
    parser.add_argument("--force", action="store_true", help="recompute method results even if they exist")
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Read the YAML config and check that the required keys are there."""
    config = yaml.safe_load(Path(path).read_text())
    missing = [key for key in ("model", "dataset", "n") if key not in config]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if config["dataset"] not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset {config['dataset']!r} is not supported yet, only {SUPPORTED_DATASETS}")
    config.setdefault("device", "cpu")
    config.setdefault("methods", "all")
    return config


def read_rows(path: Path) -> list[dict]:
    """Read a JSONL cache file (one question per line)."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_script(module: str, *args: str) -> None:
    """Run one of the existing scripts, e.g. scripts.generate, as its own process."""
    command = [sys.executable, "-m", module, *args]
    print("   $", " ".join(command[1:]))
    subprocess.run(command, cwd=ROOT, check=True)  # check=True: stop if the script fails


# ----- steps 1 and 2: data -------------------------------------------------


def prepare_data(config: dict) -> list[dict]:
    """Make sure the cache file exists and is labelled, then return its rows."""
    model, dataset, n = config["model"], config["dataset"], config["n"]
    path = DATA_DIR / model / f"{dataset}_{n}.jsonl"

    if path.exists():
        print(f"1. generate: skip, {path.relative_to(ROOT)} exists")
    else:
        print("1. generate")
        run_script("scripts.generate", "--model", model, "--n", str(n), "--device", config["device"])

    rows = read_rows(path)
    if all("correct" in row for row in rows):
        print("2. label:    skip, already labelled")
    else:
        print("2. label")
        run_script("scripts.label", "--model", model, "--n", str(n))
        rows = read_rows(path)
    return rows


# ----- step 3: methods -----------------------------------------------------


def find_plugins() -> dict:
    """Import every file in methods/ and return {NAME: module}."""
    plugins = {}
    for file in sorted(METHODS_DIR.glob("*.py")):
        if file.name.startswith("_"):  # skip __init__.py and helper files
            continue
        module = importlib.import_module(f"methods.{file.stem}")
        for attr in ("NAME", "FAMILY", "NEEDS_MODEL", "score"):
            if not hasattr(module, attr):
                raise AttributeError(f"methods/{file.name} has no {attr} (see CONTRACT.md)")
        plugins[module.NAME] = module
    return plugins


def choose_plugins(plugins: dict, config: dict, args: argparse.Namespace) -> dict:
    """Keep only the methods this run should use."""
    wanted = args.methods.split(",") if args.methods else config["methods"]
    if wanted != "all":
        unknown = [name for name in wanted if name not in plugins]
        if unknown:
            raise ValueError(f"unknown methods {unknown}; available: {sorted(plugins)}")
        plugins = {name: plugins[name] for name in wanted}
    if args.skip_model_methods:
        plugins = {name: p for name, p in plugins.items() if not p.NEEDS_MODEL}
    return plugins


def load_llm(config: dict):
    """Load the LLM once for all plugins that need it."""
    # Imported here, not at the top: torch and transformers are slow to import
    # and not needed when no plugin uses the model.
    from probly.representer.sampler.huggingface import load_model
    from scripts.generate import MODELS

    model, tok = load_model(MODELS[config["model"]])
    model = model.to(config["device"])
    model.eval()
    return model, tok


def run_methods(plugins: dict, rows: list[dict], config: dict, run: str, force: bool) -> None:
    """Call score() for every plugin and save its record."""
    ids = [row["id"] for row in rows]
    todo = {name: p for name, p in plugins.items() if force or not (RESULTS_ROOT / run / f"{name}.json").exists()}
    for name in plugins:
        if name not in todo:
            print(f"3. {name}: skip, result exists")
    if not todo:
        return

    model = tok = None
    if any(p.NEEDS_MODEL for p in todo.values()):
        print("   loading the model ...")
        model, tok = load_llm(config)

    for name, plugin in todo.items():
        print(f"3. {name}")
        start = time.time()
        record = plugin.score(rows, model, tok) if plugin.NEEDS_MODEL else plugin.score(rows)
        path = save_scores(record, run=run, expected_ids=ids)  # validates against the contract
        print(f"   saved {path.relative_to(ROOT)} ({time.time() - start:.1f} s)")


# ----- steps 4 and 5: evaluate and table -------------------------------------

def to_markdown(table) -> str:
    """Write a DataFrame as a Markdown table (numbers with 3 decimals)."""

    def cell(value) -> str:
        if isinstance(value, float):
            return "" if value != value else f"{value:.3f}"  # value != value is True only for NaN
        return str(value)

    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines) + "\n"

def make_table(rows: list[dict], plugins: dict, run: str) -> None:
    """Evaluate every method of this run and write the summary table."""
    print("4. evaluate")
    table = summary_table(rows, run=run, methods=tuple(plugins))
    table.insert(1, "family", [getattr(plugins.get(m), "FAMILY", "") for m in table["method"]])

    out = RESULTS_ROOT / run / "summary.md"
    out.write_text(to_markdown(table))
    print(f"5. table:    {out.relative_to(ROOT)}\n")
    print(table.to_string(index=False))


def main() -> None:
    """Run all steps for one config."""
    args = parse_args()
    config = load_config(args.config)
    run = f"{config['model']}/{config['dataset']}_{config['n']}"  # results/<run>/
    print(f"run: {run}")

    rows = prepare_data(config)
    plugins = choose_plugins(find_plugins(), config, args)
    run_methods(plugins, rows, config, run, args.force)
    make_table(rows, plugins, run)


if __name__ == "__main__":
    main()
