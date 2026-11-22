"""Build the LLM-probly demo from the selector profiles (task F3).

The script reads every per-model profile written by the selector (D4, ``profiles/<model>.json``, see
``selector/profile.py`` and CONTRACT.md) and regenerates three outputs from them:

1. A Markdown block in the README, between ``<!-- llm-probly-demo:start -->`` and
   ``<!-- llm-probly-demo:end -->``. Everything else in the README is left untouched.
2. A standalone demo page (Markdown) with one section per model.
3. A light and a dark SVG card with the selector's choice and the calibrated reliability per model.

Each profile already holds the metrics of every candidate method, so no separate results file is needed.
Only the standard library is used, so the demo can be rebuilt on any laptop without a model.

Example:
    uv run python experiments/llm-probly/demo/build_demo.py --readme README.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import html
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
START_MARKER = "<!-- llm-probly-demo:start -->"
END_MARKER = "<!-- llm-probly-demo:end -->"


@dataclass
class Metrics:
    """AUROC and PRR of one method with their 95 % intervals (D4 ``Metrics``).

    Attributes:
        auroc: AUROC.
        auroc_low: Lower end of the AUROC interval.
        auroc_high: Upper end of the AUROC interval.
        prr: Prediction-rejection ratio.
    """

    auroc: float
    auroc_low: float | None = None
    auroc_high: float | None = None
    prr: float | None = None


@dataclass
class DemoProfile:
    """The fields of one D4 profile that the demo shows.

    Attributes:
        model: Short model name, e.g. ``qwen1.5b``.
        model_id: Hugging Face id, e.g. ``Qwen/Qwen2.5-1.5B-Instruct``.
        dataset: Dataset the selector ran on.
        n_questions: Number of questions in the run.
        calibration_size: Number of questions the selector chose and fitted on.
        method: Name of the chosen method (or ``stack``).
        family: Family of the chosen method.
        access: What the chosen method needs from the model.
        cost: Relative cost per question of the chosen method.
        reason: One sentence on why the method was chosen.
        metrics: Metrics of the chosen method (held-out if available, else calibration part).
        metrics_split: ``"held-out"`` or ``"calibration"``, the part ``metrics`` comes from.
        calibration: Calibration type (``none``, ``platt`` or ``isotonic``).
        ece_before: ECE before calibration, or ``None``.
        ece_after: ECE after calibration, or ``None``.
        candidates: Metrics of every compared method, by name.
        source: Path of the profile file.
    """

    model: str
    model_id: str
    dataset: str
    n_questions: int
    calibration_size: int
    method: str
    family: str
    access: str
    cost: float
    reason: str
    metrics: Metrics
    metrics_split: str
    calibration: str
    ece_before: float | None
    ece_after: float | None
    candidates: dict[str, Metrics] = field(default_factory=dict)
    source: Path | None = None

    @property
    def best_candidate(self) -> tuple[str, Metrics] | None:
        """The candidate with the highest AUROC, or ``None`` if there are no candidates."""
        return max(self.candidates.items(), key=lambda item: item[1].auroc, default=None)


def _metrics(raw: dict[str, Any]) -> Metrics:
    return Metrics(
        auroc=float(raw["auroc"]),
        auroc_low=raw.get("auroc_low"),
        auroc_high=raw.get("auroc_high"),
        prr=raw.get("prr"),
    )


def load_profile(path: Path) -> DemoProfile:
    """Read one D4 profile file.

    Args:
        path: Path to ``profiles/<model>.json``.

    Returns:
        The fields the demo needs.

    Raises:
        ValueError: If the schema version differs or a required field is missing.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        msg = f"{path}: schema_version is {data.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        raise ValueError(msg)
    try:
        method = data["method"]
        held_out = data.get("held_out_metrics")
        calibration = data.get("calibration") or {}
        return DemoProfile(
            model=data["model"],
            model_id=data.get("model_id") or data["model"],
            dataset=data["dataset"],
            n_questions=int(data["n_questions"]),
            calibration_size=int(data["calibration_size"]),
            method=method["name"],
            family=method["family"],
            access=method["access"],
            cost=float(method["cost"]),
            reason=data["reason"],
            metrics=_metrics(held_out or data["calibration_metrics"]),
            metrics_split="held-out" if held_out else "calibration",
            calibration=calibration.get("type", "none"),
            ece_before=calibration.get("ece_before"),
            ece_after=calibration.get("ece_after"),
            candidates={name: _metrics(m) for name, m in (data.get("candidates") or {}).items()},
            source=path,
        )
    except (KeyError, TypeError, ValueError) as error:
        msg = f"{path}: not a valid profile ({error!r})"
        raise ValueError(msg) from error


def load_profiles(directory: Path) -> list[DemoProfile]:
    """Load every ``*.json`` profile in a directory, sorted by model name.

    Args:
        directory: The ``profiles/`` directory.

    Returns:
        The profiles.

    Raises:
        FileNotFoundError: If the directory has no profile files.
    """
    files = sorted(directory.glob("*.json"))
    if not files:
        msg = f"no profile files (*.json) in {directory}"
        raise FileNotFoundError(msg)
    return sorted((load_profile(f) for f in files), key=lambda p: p.model)


def fmt(value: float | None, digits: int = 3) -> str:
    """Format a number for the demo, or a dash if it is missing.

    Args:
        value: Number or ``None``.
        digits: Decimal places.

    Returns:
        The formatted string.
    """
    return "-" if value is None else f"{value:.{digits}f}"


def fmt_auroc(m: Metrics) -> str:
    """Format an AUROC with its interval, e.g. ``0.768 [0.672, 0.859]``.

    Args:
        m: Metrics.

    Returns:
        The formatted string.
    """
    if m.auroc_low is None or m.auroc_high is None:
        return fmt(m.auroc)
    return f"{fmt(m.auroc)} [{fmt(m.auroc_low)}, {fmt(m.auroc_high)}]"


def fmt_ece(p: DemoProfile) -> str:
    """Format the ECE before and after calibration.

    Args:
        p: Profile.

    Returns:
        ``before -> after``, or only ``after`` when there is no before value.
    """
    if p.ece_before is None:
        return fmt(p.ece_after)
    return f"{fmt(p.ece_before)} -> {fmt(p.ece_after)}"


def render_summary_table(profiles: list[DemoProfile]) -> str:
    """Render the 'which method for which model' Markdown table.

    Args:
        profiles: Selector profiles.

    Returns:
        A Markdown table.
    """
    lines = [
        "| Model | Chosen method | AUROC [95 % CI] | Best single method | Calibration | ECE | Relative cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in profiles:
        best = p.best_candidate
        best_txt = f"{best[0]} ({fmt(best[1].auroc)})" if best else "-"
        lines.append(
            f"| {p.model_id} | `{p.method}` | {fmt_auroc(p.metrics)} | {best_txt} | {p.calibration} "
            f"| {fmt_ece(p)} | {p.cost:g} |"
        )
    return "\n".join(lines)


def render_readme_block(profiles: list[DemoProfile], svg_base: str) -> str:
    """Render the README demo block (without the markers).

    Args:
        profiles: Selector profiles.
        svg_base: URL or relative path prefix of the SVG cards.

    Returns:
        Markdown text.
    """
    picture = (
        '<div align="center">\n  <picture>\n'
        f'    <source srcset="{svg_base}/llm_selector_demo_dark.svg" media="(prefers-color-scheme: dark)">\n'
        f'    <source srcset="{svg_base}/llm_selector_demo_light.svg" media="(prefers-color-scheme: light)">\n'
        f'    <img src="{svg_base}/llm_selector_demo_light.svg" alt="Selector choice and calibrated reliability '
        'per model" width="100%" />\n  </picture>\n</div>'
    )
    intro = (
        "For each model, the selector picks the cheapest uncertainty method that is not significantly worse "
        "than the best one, and calibrates its score into a confidence in [0, 1]."
    )
    note = "_Generated by `experiments/llm-probly/demo/build_demo.py` from the saved profiles. Do not edit by hand._"
    return "\n\n".join([intro, picture, render_summary_table(profiles), note])


def render_page(profiles: list[DemoProfile]) -> str:
    """Render the standalone demo page with one section per model.

    Args:
        profiles: Selector profiles.

    Returns:
        Markdown text.
    """
    parts = [
        "# LLM-probly demo: which uncertainty method for which model",
        "Question -> answer -> chosen method -> calibrated confidence. Generated from the selector profiles.",
        "## Overview",
        render_summary_table(profiles),
    ]
    for p in profiles:
        section = [
            f"## {p.model_id}",
            f"- **Chosen method:** `{p.method}` ({p.family}, needs {p.access}, relative cost {p.cost:g})",
            f"- **Why:** {p.reason}",
            f"- **AUROC:** {fmt_auroc(p.metrics)} on the {p.metrics_split} part",
            f"- **PRR:** {fmt(p.metrics.prr)}",
            f"- **Calibration:** {p.calibration}, ECE {fmt_ece(p)}",
            f"- **Data:** {p.dataset}, {p.n_questions} questions, {p.calibration_size} used to choose and fit",
        ]
        if p.candidates:
            section.append("\n| Method | AUROC [95 % CI] | PRR |\n|---|---|---|")
            ranked = sorted(p.candidates.items(), key=lambda item: item[1].auroc, reverse=True)
            section.extend(
                f"| {f'**{name}** (chosen)' if name == p.method else name} | {fmt_auroc(m)} | {fmt(m.prr)} |"
                for name, m in ranked
            )
        parts.append("\n".join(section))
    return "\n\n".join(parts) + "\n"


def render_svg(profiles: list[DemoProfile], dark: bool) -> str:
    """Render an SVG card with one row per model: chosen method and a reliability bar.

    The bar shows ``1 - ECE`` after calibration; it is empty when the profile is not calibrated yet.

    Args:
        profiles: Selector profiles.
        dark: Use the dark color scheme.

    Returns:
        SVG markup.
    """
    colors = ("#e6e6e6", "#9a9a9a", "#333", "#5fa8ff") if dark else ("#1a1a1a", "#666", "#e5e5e5", "#1f6feb")
    fg, muted, track, accent = colors
    row_h, width = 44, 760
    height = 56 + row_h * len(profiles)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="system-ui, sans-serif">',
        f'<text x="20" y="30" font-size="16" font-weight="600" fill="{fg}">Selector choice per model</text>',
        f'<text x="560" y="30" font-size="12" fill="{muted}">reliability (1 - ECE)</text>',
    ]
    for i, p in enumerate(profiles):
        y = 60 + i * row_h
        reliability = None if p.ece_after is None else max(0.0, min(1.0, 1.0 - p.ece_after))
        name = html.escape(p.model_id.rsplit("/", 1)[-1])
        bar = 160 * (reliability or 0.0)
        label = fmt(reliability, 2)
        out.extend(
            [
                f'<text x="20" y="{y + 6}" font-size="14" fill="{fg}">{name}</text>',
                f'<text x="250" y="{y + 6}" font-size="14" fill="{accent}">{html.escape(p.method)}</text>',
                f'<rect x="560" y="{y - 6}" width="160" height="12" rx="6" fill="{track}"/>',
                f'<rect x="560" y="{y - 6}" width="{bar:.1f}" height="12" rx="6" fill="{accent}"/>',
                f'<text x="755" y="{y + 5}" font-size="12" fill="{muted}" text-anchor="end">{label}</text>',
            ]
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def replace_block(text: str, block: str) -> str:
    """Replace the content between the demo markers, or append a marked block if the markers are missing.

    Args:
        text: Current README text.
        block: New block content.

    Returns:
        The updated README text.
    """
    new = f"{START_MARKER}\n{block}\n{END_MARKER}"
    start, end = text.find(START_MARKER), text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        head = text.rstrip("\n")
        return f"{head}\n\n{new}\n" if head else f"{new}\n"
    return text[:start] + new + text[end + len(END_MARKER) :]


def build(
    profiles_dir: Path,
    readme: Path | None,
    page: Path | None,
    svg_dir: Path | None,
    svg_base: str,
) -> list[Path]:
    """Regenerate every demo output from the profiles.

    Args:
        profiles_dir: Directory with the D4 profile files.
        readme: README to update in place, or ``None`` to skip.
        page: Demo page to write, or ``None`` to skip.
        svg_dir: Directory for the SVG cards, or ``None`` to skip.
        svg_base: Prefix used for the SVG links in the README block.

    Returns:
        Paths of the files that were written.
    """
    profiles = load_profiles(profiles_dir)
    written: list[Path] = []
    if svg_dir is not None:
        svg_dir.mkdir(parents=True, exist_ok=True)
        for name, dark in (("llm_selector_demo_light.svg", False), ("llm_selector_demo_dark.svg", True)):
            (svg_dir / name).write_text(render_svg(profiles, dark), encoding="utf-8")
            written.append(svg_dir / name)
    if page is not None:
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(render_page(profiles), encoding="utf-8")
        written.append(page)
    if readme is not None:
        current = readme.read_text(encoding="utf-8") if readme.exists() else ""
        readme.write_text(replace_block(current, render_readme_block(profiles, svg_base)), encoding="utf-8")
        written.append(readme)
    return written


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Args:
        argv: Arguments, or ``None`` to read ``sys.argv``.
    """
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build the LLM-probly demo from the selector profiles.")
    parser.add_argument("--profiles", type=Path, default=here.parent / "profiles")
    parser.add_argument("--readme", type=Path, default=None, help="README to update (skipped if not given)")
    parser.add_argument("--page", type=Path, default=here / "DEMO.md")
    parser.add_argument("--svg-dir", type=Path, default=here / "assets")
    parser.add_argument("--svg-base", default="experiments/llm-probly/demo/assets", help="SVG link prefix in README")
    args = parser.parse_args(argv)
    for path in build(args.profiles, args.readme, args.page, args.svg_dir, args.svg_base):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
