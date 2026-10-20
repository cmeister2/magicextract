#!/usr/bin/env python3
"""Render the PR's extraction-latency chart from the archived benchmark summary.

Requires Matplotlib (tested with 3.11.2). Run from any directory:
    python3 benchmarks/2026-10-20/render-performance-chart.py

Each panel compares the same eligible scenes. The panels use different cohorts
and must not be read as successive optimizations of one shared workload.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SUMMARY = Path(__file__).with_name("summary-final.json")
OUTPUT = Path(__file__).with_name("images") / "latency-comparison.png"

INK = "#162938"
MUTED = "#536677"
BEFORE = "#74869b"
AFTER = "#087f70"
WARNING = "#a45112"
BACKGROUND = "#f5f8fa"
GRID = "#dce4e9"


def comparison(summary, description):
    """Select an exact, explicitly paired comparison from the saved summary."""
    return next(
        item for item in summary["comparisons"]
        if item["description"] == description
    )


def accuracy_label(accuracy):
    """Show the number of scenes meeting the recorded correctness thresholds."""
    return f'{accuracy["passed"]}/{accuracy["scenes"]} scenes pass'


def render_panel(ax, item, title, subtitle, before_label, after_label):
    """Draw median bars and separately marked p95 tails without error-bar claims."""
    metrics = item["extract"]
    ax.set_facecolor("white")
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.6, 1.85)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="x", colors=MUTED, length=0, pad=8, labelsize=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(-0.215, 1.18, title, transform=ax.transAxes, fontsize=18,
            fontweight="bold", color=INK, va="bottom")
    ax.text(-0.215, 1.04, subtitle, transform=ax.transAxes, fontsize=13,
            color=MUTED, va="bottom")
    ax.text(1, 1.17, f'{metrics["median_speedup"]:.2f}× faster median',
            transform=ax.transAxes, fontsize=15, fontweight="bold", color=AFTER,
            ha="right", va="bottom")

    rows = [
        (1.05, "before", before_label, BEFORE),
        (0.00, "after", after_label, AFTER),
    ]
    for y, key, label, color in rows:
        timing = metrics[f"{key}_ms"]
        accuracy = item[f"{key}_accuracy"]
        median, p95 = timing["median"], timing["p95"]
        ax.barh(y, median, height=0.34, color=color, zorder=3)
        ax.plot([median, p95], [y, y], color=color, linewidth=2, zorder=2)
        ax.plot(p95, y, marker="o", markersize=7, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=2, zorder=4)
        ax.text(median - 1.1, y, f"{median:.1f}", color="white", va="center",
                ha="right", fontsize=13, fontweight="bold")
        ax.text(p95 + 1.2, y, f"{p95:.1f}", color=color, va="center",
                fontsize=13, fontweight="bold")
        # Labels use the axes' x coordinate and data y coordinate.
        ax.text(-0.025, y + 0.03, label,
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=15, fontweight="bold", color=INK)
        ax.text(-0.025, y - 0.12, accuracy_label(accuracy),
                transform=ax.get_yaxis_transform(), ha="right", va="top",
                fontsize=12,
                color=WARNING if accuracy["passed"] != accuracy["scenes"] else MUTED)

    ax.set_xlabel("Extraction latency (ms) · lower is faster", color=MUTED,
                  fontsize=12, labelpad=9)


def main():
    """Read the archived measurements and write the reproducible chart."""
    summary = json.loads(SUMMARY.read_text())
    baseline = summary["baseline_comparison"]
    angle = comparison(
        summary, "full-640 -> full-640-angle, identical eligible scenes"
    )
    roi = comparison(
        summary, "full-640 -> full-640-roi, identical eligible scenes"
    )
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(14, 11.7), dpi=120, facecolor=BACKGROUND)
    fig.text(0.05, 0.957, "Faster card extraction, measured first",
             fontsize=27, fontweight="bold", color=INK)
    fig.text(0.05, 0.927, "Full extraction • matched scene pairs in every panel",
             fontsize=16, color=MUTED)

    fig.legend(
        handles=[
            Patch(color=AFTER, label="Filled bar: median"),
            Line2D([], [], color=AFTER, marker="o", markerfacecolor="white",
                   markeredgewidth=2, linewidth=2, label="Open circle: p95"),
        ],
        loc="upper left", bbox_to_anchor=(0.043, 0.906),
        frameon=False, ncol=2, fontsize=13, labelcolor=MUTED,
        columnspacing=2.2,
    )

    panels = [
        (0.655, baseline, "Full pipeline optimization",
         "68 regression scenes · same inputs, same correctness thresholds",
         "Original Full", "Optimized Full"),
        (0.400, angle, "Optional orientation constraint",
         "70 eligible scenes · optimized Full at 640 px · upright ±15°",
         "Unrestricted", "Upright ±15°"),
        (0.145, roi, "Optional central detection region",
         "107 eligible scenes · optimized Full at 640 px · central 80% width × 80% height",
         "Whole frame", "Central region"),
    ]
    for bottom, item, title, subtitle, before_label, after_label in panels:
        ax = fig.add_axes([0.23, bottom, 0.72, 0.155])
        render_panel(ax, item, title, subtitle, before_label, after_label)

    fig.text(0.05, 0.066,
             "AMD Ryzen 7 PRO 7840U · release build · 640 × 520 fixtures · 3 timed samples per scene",
             fontsize=12, color=MUTED)
    fig.text(0.05, 0.043,
             "Timing includes detection + RGB crop allocation; excludes decoding, capture and display.",
             fontsize=12, color=MUTED)
    fig.text(0.05, 0.020,
             "Each panel has its own matched cohort. Android handset runtime is not yet measured.",
             fontsize=12, color=INK, fontweight="bold")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=120, facecolor=BACKGROUND,
                metadata={"Title": "MagicExtract matched extraction latency"})
    plt.close(fig)
    print(f"Wrote {OUTPUT} (1680 × 1404)")


if __name__ == "__main__":
    main()
