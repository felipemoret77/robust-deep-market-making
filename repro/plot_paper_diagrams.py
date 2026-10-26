#!/usr/bin/env python3
"""Render both paper diagrams from explicit vector source, without raster inputs.

  python repro/plot_paper_diagrams.py --output outputs/paper_diagrams_proposals

PNG, PDF and SVG are generated. The selected paper images are never overwritten.
The bandit source was recovered from presentation TikZ; the regime illustration
was recreated as declarative coordinates. See data/paper/diagrams/provenance.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as VectorPath

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/paper/diagrams"
NAMES = ("regime_model_diagram", "scenario_bandit_diagram")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canvas(spec, dpi):
    width, height = spec["size_pixels"]
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    return fig, ax


def arrow(ax, vertices, *, dashed=False, linewidth=.7):
    path = VectorPath(vertices, [VectorPath.MOVETO] + [VectorPath.LINETO] * (len(vertices) - 1))
    patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=8.5,
                            color="black", linewidth=linewidth,
                            linestyle=(0, (4.3, 4.3)) if dashed else "solid",
                            capstyle="butt", joinstyle="miter")
    ax.add_patch(patch)


def brace(ax, left, right, y, color):
    """Cubic Bezier underbrace, kept as paths in both vector exports."""
    center = (left + right) / 2
    vertices = [(left, y), (left + 3, y + 7), (left + 8, y + 9), (left + 20, y + 9),
                (center - 20, y + 9), (center - 8, y + 9), (center - 3, y + 10), (center, y + 17),
                (center + 3, y + 10), (center + 8, y + 9), (center + 20, y + 9),
                (right - 20, y + 9), (right - 8, y + 9), (right - 3, y + 7), (right, y)]
    codes = ([VectorPath.MOVETO] + [VectorPath.CURVE4] * 3 + [VectorPath.LINETO]
             + [VectorPath.CURVE4] * 6 + [VectorPath.LINETO] + [VectorPath.CURVE4] * 3)
    ax.add_patch(PathPatch(VectorPath(vertices, codes), facecolor="none", edgecolor=color, linewidth=.45))


def regime_diagram(spec, dpi):
    fig, ax = canvas(spec, dpi)
    color = spec["colors"]
    boundaries = spec["time_boundaries"]
    levels = spec["probability_rows"]
    top, bottom = spec["band_rows"]
    x0, baseline = spec["axis_origin"]
    ax.add_patch(Rectangle((x0, top), boundaries[-1] - x0, bottom - top, facecolor=color["band"], edgecolor="none"))
    ax.plot([x0, boundaries[-1]], [spec["balanced_row"]] * 2, color="#aaaaaa", linewidth=.5, dashes=(6, 6))
    arrow(ax, [(x0, baseline), (1635, baseline)])
    arrow(ax, [(x0, baseline), (x0, 76)])
    for i, level in enumerate(levels):
        side = "buy" if level < spec["balanced_row"] else "sell"
        ax.plot(boundaries[i:i + 2], [level, level], color=color[side], linewidth=1.8, solid_capstyle="butt")
        if i < len(levels) - 1:
            ax.plot([boundaries[i + 1]] * 2, [level, levels[i + 1]], color=color["connector"], linewidth=.75, dashes=(1, 2))
            ax.plot([boundaries[i + 1]] * 2, [baseline - 7, baseline + 7], color="#aaaaaa", linewidth=.5)
    for label, level in zip(spec["labels"], levels):
        ax.text(*label["xy"], label["text"], ha="center", va="center", fontsize=8.5,
                color=color["buy"] if level < spec["balanced_row"] else color["sell"])
    for text, y, ink in ((r"$0.5+\omega$", top, color["annotation"]),
                         (r"$0.5$", spec["balanced_row"], "#555555"),
                         (r"$0.5-\omega$", bottom, color["annotation"])):
        ax.text(127, y, text, fontsize=8.5, ha="right", va="center", color=ink)
    ax.text(x0, 37, r"$p_{\mathrm{buy}}$", fontsize=9.5, ha="center", va="center")
    ax.text(1653, baseline, r"MO count $m$", fontsize=8.5, ha="left", va="center")
    for item in spec["braces"]:
        brace(ax, item["from"], item["to"], 484, color["annotation"])
        ax.text((item["from"] + item["to"]) / 2, 528, item["label"], ha="center", va="center",
                fontsize=8.5, color=color["annotation"])
    return fig


def bandit_diagram(spec, dpi):
    fig, ax = canvas(spec, dpi)
    nodes = spec["nodes"]
    for node in nodes:
        left, top, right, bottom = node["bounds"]
        ax.add_patch(FancyBboxPatch((left, top), right - left, bottom - top,
                     boxstyle="round,pad=0,rounding_size=9", linewidth=.45,
                     facecolor=node["color"], edgecolor="black"))
        line_height = 39.5
        center = (top + bottom) / 2
        for j, label in enumerate(node["lines"]):
            y = center + (j - (len(node["lines"]) - 1) / 2) * line_height
            ax.text((left + right) / 2, y, label, fontsize=7.6, ha="center", va="center")
    for upper, lower, label in zip(nodes, nodes[1:], spec["arrow_labels"]):
        start, end = upper["bounds"][3], lower["bounds"][1]
        arrow(ax, [(459, start + 1), (459, end - 3)])
        ax.text(484, (start + end) / 2, label, fontsize=7.6, ha="left", va="center")
    arrow(ax, spec["feedback_right"], dashed=True)
    arrow(ax, spec["feedback_left"], dashed=True)
    ax.text(777, 270, r"updated $d_i$", fontsize=7.6, ha="center", va="center")
    ax.text(132, 41, "refresh easiest", fontsize=7.6, ha="center", va="center")
    ax.text(132, 80, r"10% from $P_0$", fontsize=7.6, ha="center", va="center")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_diagrams_proposals")
    parser.add_argument("--diagram", choices=["all", *NAMES], default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    for protected in (ROOT / "paper", ROOT / "data", ROOT / "src", ROOT / "repro", ROOT / "config"):
        if output == protected or protected in output.parents:
            parser.error(f"Use an outputs/ directory; selected assets and source are protected: {protected}")
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((DATA / "diagram_spec.json").read_text())
    provenance = json.loads((DATA / "provenance.json").read_text())
    for filename, expected in provenance["render_inputs"].items():
        if sha256(DATA / filename) != expected:
            raise ValueError(f"Diagram source hash differs: {filename}")
    plt.rcParams.update({"font.family": source["font_family"], "mathtext.fontset": source["math_fontset"],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": "paper-diagrams-v1"})
    generated = {}
    names = NAMES if args.diagram == "all" else (args.diagram,)
    for name in names:
        figure = (regime_diagram if name == "regime_model_diagram" else bandit_diagram)(source[name], source["dpi"])
        for extension in ("png", "pdf", "svg"):
            path = output / f"{name}.{extension}"
            metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else ({"Date": None} if extension == "svg" else {})
            figure.savefig(path, dpi=source["dpi"], facecolor="white", metadata=metadata)
            generated[path.name] = sha256(path)
        plt.close(figure)
        print(f"Rendered {name}.{{png,pdf,svg}} from vector source")
    manifest = dict(renderer_sha256=sha256(__file__), source_sha256=sha256(DATA / "diagram_spec.json"),
                    provenance_sha256=sha256(DATA / "provenance.json"), matplotlib_version=matplotlib.__version__,
                    input_rasters=[], pixel_identity_with_previous_exports_expected=False, generated=generated)
    (output / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
