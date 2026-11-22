"""Tests for the F3 demo builder."""
# ruff: noqa: S101, ANN001, ANN201

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
PROFILES = Path(__file__).resolve().parent / "fixtures" / "demo_profiles"

_spec = importlib.util.spec_from_file_location("build_demo", DEMO_DIR / "build_demo.py")
assert _spec is not None
assert _spec.loader is not None
build_demo = importlib.util.module_from_spec(_spec)
sys.modules["build_demo"] = build_demo
_spec.loader.exec_module(build_demo)


def _profile_data() -> dict:
    return json.loads((PROFILES / "qwen1.5b.json").read_text())


def test_load_real_d4_profile():
    (p,) = build_demo.load_profiles(PROFILES)
    assert p.model == "qwen1.5b"
    assert p.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert p.method == "perplexity"
    assert p.family == "baseline"
    assert p.metrics_split == "calibration"
    assert p.metrics.auroc == pytest.approx(0.7683)
    assert p.calibration == "isotonic"
    assert p.ece_before is None
    assert p.ece_after == pytest.approx(0.0808)
    assert p.best_candidate[0] == "p_true_zeroshot"


def test_held_out_metrics_win_over_calibration_metrics(tmp_path):
    data = _profile_data()
    data["held_out_metrics"] = {"auroc": 0.75, "auroc_low": 0.7, "auroc_high": 0.8, "prr": 0.5}
    (tmp_path / "m.json").write_text(json.dumps(data))
    (p,) = build_demo.load_profiles(tmp_path)
    assert p.metrics.auroc == 0.75
    assert p.metrics_split == "held-out"


def test_wrong_schema_version_raises(tmp_path):
    data = _profile_data()
    data["schema_version"] = 2
    (tmp_path / "m.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema_version"):
        build_demo.load_profiles(tmp_path)


def test_missing_method_raises(tmp_path):
    data = _profile_data()
    del data["method"]
    (tmp_path / "m.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not a valid profile"):
        build_demo.load_profiles(tmp_path)


def test_empty_profile_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_demo.load_profiles(tmp_path)


def test_uncalibrated_profile_renders_dashes(tmp_path):
    data = _profile_data()
    data["calibration"] = {"type": "none", "parameters": {}, "ece_before": None, "ece_after": None}
    data["candidates"] = {}
    (tmp_path / "m.json").write_text(json.dumps(data))
    profiles = build_demo.load_profiles(tmp_path)
    assert "| none | - |" in build_demo.render_summary_table(profiles)
    assert 'width="0.0"' in build_demo.render_svg(profiles, dark=True)


def test_replace_block_keeps_rest_of_readme():
    text = f"intro\n{build_demo.START_MARKER}\nold\n{build_demo.END_MARKER}\noutro\n"
    out = build_demo.replace_block(text, "new")
    assert out == f"intro\n{build_demo.START_MARKER}\nnew\n{build_demo.END_MARKER}\noutro\n"
    assert build_demo.replace_block(out, "new") == out


def test_replace_block_appends_when_markers_missing():
    out = build_demo.replace_block("intro\n", "new")
    assert out.startswith("intro\n\n")
    assert out.rstrip().endswith(build_demo.END_MARKER)


def test_build_writes_all_outputs_and_is_idempotent(tmp_path):
    profiles = tmp_path / "profiles"
    shutil.copytree(PROFILES, profiles)
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n")
    page = tmp_path / "DEMO.md"
    svg_dir = tmp_path / "svg"
    args = (profiles, readme, page, svg_dir, "svg")
    assert len(build_demo.build(*args)) == 4
    first = readme.read_text()
    assert "`perplexity`" in first
    assert "p_true_zeroshot (0.808)" in first  # best single method shown next to the choice
    assert "**perplexity** (chosen)" in page.read_text()
    assert (svg_dir / "llm_selector_demo_dark.svg").read_text().startswith("<svg")
    build_demo.build(*args)
    assert readme.read_text() == first


def test_svg_escapes_names(tmp_path):
    data = _profile_data()
    data["model_id"] = "org/a<b"
    data["method"]["name"] = "x&y"
    (tmp_path / "m.json").write_text(json.dumps(data))
    svg = build_demo.render_svg(build_demo.load_profiles(tmp_path), dark=False)
    assert "a&lt;b" in svg
    assert "x&amp;y" in svg
