"""Pytest setup for the LLM-probly experiment tests.

The tests import the experiment packages directly (``common``, ``methods``,
``selector``, ``scripts``). probly runs pytest with ``--import-mode=importlib``,
which does not add test folders to ``sys.path``. This file adds the experiment
folder, so that ``uv run pytest experiments/llm-probly/tests`` works from the
repository root and from this folder alike.
"""

from __future__ import annotations

from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parent

if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
