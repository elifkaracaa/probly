"""Build the LLM-probly demo from the selector profiles.

The script reads every per-model profile written by the method selector (``profiles/<model>.json``, loaded
with ``selector.profile.load_profile``, format in CONTRACT.md) and regenerates three outputs from them:

1. A Markdown block in the README, between ``<!-- llm-probly-demo:start -->`` and
   ``<!-- llm-probly-demo:end -->``. Everything else in the README is left untouched.
2. A standalone demo page (Markdown) with one section per model.
3. A light and a dark SVG card with the selector's choice and the calibrated reliability per model.

With ``--animation-ids`` it also writes the animated README demo (``demo/animation.py``) for those
questions, so all demo outputs come from one command.

Each profile already holds the metrics of every candidate method, so no separate results file is needed,
and no model is loaded.

Example (from ``experiments/llm-probly``)::

    uv run python -m demo.build_demo --readme README.md
    uv run python -m demo.build_demo --animation-run qwen1.5b --animation-n 200 --animation-ids 48 169
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from selector.profile import Metrics, Profile, load_profile

from demo.animation import write_animation

START_MARKER = "<!-- llm-probly-demo:start -->"
END_MARKER = "<!-- llm-probly-demo:end -->"


def load_profiles(directory: Path) -> list[Profile]:
    """Load and check every ``*.json`` profile in a directory, sorted by model name.

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


def shown_metrics(p: Profile) -> tuple[Metrics, str]:
    """Pick the metrics of the chosen method that the demo shows.

    Args:
        p: Profile.

    Returns:
        The held-out metrics if the profile has them, else the calibration-part metrics, and the name of
        the part they come from (``"held-out"`` or ``"calibration"``).
    """
    if p.held_out_metrics is not None:
        return p.held_out_metrics, "held-out"
    return p.calibration_metrics, "calibration"


def best_candidate(p: Profile) -> tuple[str, Metrics] | None:
    """Find the compared method with the highest AUROC.

    Args:
        p: Profile.

    Returns:
        Name and metrics of the best candidate, or ``None`` if the profile has no candidates.
    """
    return max(p.candidates.items(), key=lambda item: item[1].auroc, default=None)


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
    return f"{fmt(m.auroc)} [{fmt(m.auroc_low)}, {fmt(m.auroc_high)}]"


def fmt_ece(p: Profile) -> str:
    """Format the ECE before and after calibration.

    Args:
        p: Profile.

    Returns:
        ``before -> after``, or only ``after`` when there is no before value.
    """
    cal = p.calibration
    if cal.ece_before is None:
        return fmt(cal.ece_after)
    return f"{fmt(cal.ece_before)} -> {fmt(cal.ece_after)}"


def render_summary_table(profiles: list[Profile]) -> str:
    """Render the 'which method for which model' Markdown table.

    The chosen method and the best single method are compared on the same data (the calibration
    part, where the selector chose); the held-out AUROC of the chosen method has its own column.

    Args:
        profiles: Selector profiles.

    Returns:
        A Markdown table.
    """
    lines = [
        (
            "| Model | Chosen method | AUROC, calibration part [95 % CI] | Best single method, calibration part "
            "| Held-out AUROC | Calibration | ECE | Relative cost |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in profiles:
        best = best_candidate(p)
        best_txt = f"{best[0]} ({fmt(best[1].auroc)})" if best else "-"
        held_out = fmt(p.held_out_metrics.auroc) if p.held_out_metrics is not None else "-"
        lines.append(
            f"| {p.model_id} | `{p.method.name}` | {fmt_auroc(p.calibration_metrics)} | {best_txt} | {held_out} "
            f"| {p.calibration.type} | {fmt_ece(p)} | {p.method.cost:g} |"
        )
    return "\n".join(lines)


def render_readme_block(profiles: list[Profile], svg_base: str) -> str:
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


def render_page(profiles: list[Profile]) -> str:
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
        metrics, split = shown_metrics(p)
        section = [
            f"## {p.model_id}",
            (
                f"- **Chosen method:** `{p.method.name}` ({p.method.family}, needs {p.method.access}, "
                f"relative cost {p.method.cost:g})"
            ),
            f"- **Why:** {p.reason}",
            f"- **AUROC:** {fmt_auroc(metrics)} on the {split} part",
            f"- **PRR:** {fmt(metrics.prr)}",
            f"- **Calibration:** {p.calibration.type}, ECE {fmt_ece(p)}",
            f"- **Data:** {p.dataset}, {p.n_questions} questions, {p.calibration_size} used to choose and fit",
        ]
        if p.candidates:
            section.append("\n| Method | AUROC [95 % CI] | PRR |\n|---|---|---|")
            ranked = sorted(p.candidates.items(), key=lambda item: item[1].auroc, reverse=True)
            section.extend(
                f"| {f'**{name}** (chosen)' if name == p.method.name else name} | {fmt_auroc(m)} | {fmt(m.prr)} |"
                for name, m in ranked
            )
        parts.append("\n".join(section))
    return "\n\n".join(parts) + "\n"


def render_svg(profiles: list[Profile], dark: bool) -> str:
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
        ece_after = p.calibration.ece_after
        reliability = None if ece_after is None else max(0.0, min(1.0, 1.0 - ece_after))
        name = html.escape(p.model_id.rsplit("/", 1)[-1])
        bar = 160 * (reliability or 0.0)
        label = fmt(reliability, 2)
        out.extend(
            [
                f'<text x="20" y="{y + 6}" font-size="14" fill="{fg}">{name}</text>',
                f'<text x="250" y="{y + 6}" font-size="14" fill="{accent}">{html.escape(p.method.name)}</text>',
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
    animation: tuple[str, int, list[int]] | None = None,
) -> list[Path]:
    """Regenerate every demo output from the profiles.

    Args:
        profiles_dir: Directory with the selector profile files.
        readme: README to update in place, or ``None`` to skip.
        page: Demo page to write, or ``None`` to skip.
        svg_dir: Directory for the SVG cards, or ``None`` to skip.
        svg_base: Prefix used for the SVG links in the README block.
        animation: ``(run, n, ids)`` to also write the animated README demo, or ``None`` to skip.

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
    if animation is not None:
        run, n, ids = animation
        tag = f"_{run.removeprefix('qwen')}_{n}q"
        written.extend(write_animation(run, n, ids, Path(__file__).resolve().parent, tag))
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
    parser.add_argument(
        "--svg-base", default="demo/assets", help="SVG link prefix in the README block, relative to the README"
    )
    parser.add_argument("--animation-run", default="qwen1.5b", help="model folder for the animated demo")
    parser.add_argument("--animation-n", type=int, default=200, help="number of questions in that cache")
    parser.add_argument(
        "--animation-ids", type=int, nargs="+", help="question ids for the animated demo (skipped if not given)"
    )
    args = parser.parse_args(argv)
    animation = (args.animation_run, args.animation_n, args.animation_ids) if args.animation_ids else None
    for path in build(args.profiles, args.readme, args.page, args.svg_dir, args.svg_base, animation):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
