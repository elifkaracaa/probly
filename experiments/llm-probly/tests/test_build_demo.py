"""Tests for the demo builder (demo/build_demo.py)."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from demo import build_demo
import pytest

PROFILES = Path(__file__).resolve().parent / "fixtures" / "demo_profiles"


def _profile_data() -> dict:
    return json.loads((PROFILES / "qwen1.5b.json").read_text())


def _write(tmp_path: Path, data: dict) -> Path:
    (tmp_path / "m.json").write_text(json.dumps(data))
    return tmp_path


def test_load_saved_profile():
    (p,) = build_demo.load_profiles(PROFILES)
    metrics, split = build_demo.shown_metrics(p)
    assert p.model == "qwen1.5b"
    assert p.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert p.method.name == "perplexity"
    assert p.method.family == "baseline"
    assert split == "calibration"
    assert metrics.auroc == pytest.approx(0.7683)
    assert p.calibration.type == "isotonic"
    assert p.calibration.ece_before is None
    assert p.calibration.ece_after == pytest.approx(0.0808)
    best = build_demo.best_candidate(p)
    assert best is not None
    assert best[0] == "p_true_zeroshot"


def test_held_out_metrics_win_over_calibration_metrics(tmp_path):
    data = _profile_data()
    data["held_out_metrics"] = {
        "auroc": 0.75,
        "auroc_low": 0.7,
        "auroc_high": 0.8,
        "prr": 0.5,
        "prr_low": 0.4,
        "prr_high": 0.6,
    }
    (p,) = build_demo.load_profiles(_write(tmp_path, data))
    metrics, split = build_demo.shown_metrics(p)
    assert metrics.auroc == 0.75
    assert split == "held-out"


def test_wrong_schema_version_raises(tmp_path):
    data = _profile_data()
    data["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        build_demo.load_profiles(_write(tmp_path, data))


def test_missing_method_raises(tmp_path):
    data = _profile_data()
    del data["method"]
    with pytest.raises(ValueError, match="not a valid profile"):
        build_demo.load_profiles(_write(tmp_path, data))


def test_selector_validation_applies_to_the_demo(tmp_path):
    data = _profile_data()
    data["method"]["family"] = "unknown"
    with pytest.raises(ValueError, match=r"method\.family"):
        build_demo.load_profiles(_write(tmp_path, data))


def test_empty_profile_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_demo.load_profiles(tmp_path)


def test_uncalibrated_profile_renders_dashes(tmp_path):
    data = _profile_data()
    data["calibration"] = {"type": "none", "parameters": {}, "ece_before": None, "ece_after": None}
    data["candidates"] = {}
    profiles = build_demo.load_profiles(_write(tmp_path, data))
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
    svg = build_demo.render_svg(build_demo.load_profiles(_write(tmp_path, data)), dark=False)
    assert "a&lt;b" in svg
    assert "x&amp;y" in svg
