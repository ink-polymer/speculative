"""Generate vector figures for the ICLR manuscript.

The script contains no result measurements.  The experiment matrix deliberately
renders empty cells until the audited formal run is complete.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle


OUT = Path(__file__).resolve().parent
BLUE = "#2962A3"
ORANGE = "#E07A2D"
GREEN = "#238B57"
PURPLE = "#7353BA"
GRAY = "#5F6B76"
LIGHT = "#EEF3F7"
DARK = "#1D2630"


def setup(width: float, height: float):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, xy, wh, text, color, *, subtitle=None, lw=1.2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor="white", edgecolor=color, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * (0.60 if subtitle else 0.5), text,
            ha="center", va="center", color=DARK, weight="bold", fontsize=8)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.28, subtitle,
                ha="center", va="center", color=GRAY, fontsize=6.6)
    return patch


def arrow(ax, start, end, color=GRAY, *, style="-|>", lw=1.25, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=9,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
    ))


def system_overview():
    fig, ax = setup(7.15, 2.65)
    ax.text(0.02, 0.92, "Common proposal and target computation", color=GRAY,
            weight="bold", fontsize=8)
    box(ax, (0.02, 0.56), (0.18, 0.22), "Verified history", BLUE,
        subtitle="target KV + anchor")
    box(ax, (0.25, 0.56), (0.18, 0.22), "DFlash draft", ORANGE,
        subtitle="one forward, L slots")
    box(ax, (0.48, 0.56), (0.18, 0.22), "Nested DDTree", GREEN,
        subtitle="best-first prefixes")
    box(ax, (0.72, 0.56), (0.25, 0.22), "Ancestor-masked Target", BLUE,
        subtitle="one row per tree node")
    arrow(ax, (0.20, 0.67), (0.25, 0.67))
    arrow(ax, (0.43, 0.67), (0.48, 0.67))
    arrow(ax, (0.66, 0.67), (0.72, 0.67))

    ax.plot([0.60, 0.60], [0.53, 0.43], color=GRAY, lw=1)
    arrow(ax, (0.60, 0.43), (0.31, 0.31), color=GREEN)
    arrow(ax, (0.83, 0.53), (0.81, 0.31), color=PURPLE)

    box(ax, (0.14, 0.07), (0.34, 0.22), "T = 0: AdaptiveTree", GREEN,
        subtitle="budget-aware cost and greedy commit")
    box(ax, (0.62, 0.07), (0.35, 0.22), "T > 0: exact TBV", PURPLE,
        subtitle="persistent scan on reached rows")
    ax.text(0.50, 0.17, "protocol\nseparation", ha="center", va="center",
            fontsize=6.8, color=GRAY)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "system_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def tree_block_walk():
    fig, ax = setup(7.0, 2.65)
    nodes = {
        "r": (0.10, 0.52),
        "a": (0.31, 0.73), "b": (0.31, 0.31),
        "c": (0.53, 0.82), "d": (0.53, 0.62),
        "e": (0.53, 0.35), "f": (0.53, 0.15),
        "g": (0.73, 0.70), "h": (0.73, 0.50),
    }
    edges = [("r", "a", "the"), ("r", "b", "a"),
             ("a", "c", "model"), ("a", "d", "tree"),
             ("b", "e", "small"), ("b", "f", "fast"),
             ("d", "g", "can"), ("d", "h", "will")]
    accepted = {("r", "a"), ("a", "d")}
    for u, v, label in edges:
        col = BLUE if (u, v) in accepted else "#AAB4BD"
        arrow(ax, nodes[u], nodes[v], color=col, lw=2.0 if (u, v) in accepted else 1.0)
        mx = (nodes[u][0] + nodes[v][0]) / 2
        my = (nodes[u][1] + nodes[v][1]) / 2
        ax.text(mx, my + 0.035, label, ha="center", va="center", color=col,
                fontsize=7, weight="bold" if (u, v) in accepted else "normal")
    for name, (x, y) in nodes.items():
        fc = BLUE if name in {"r", "a", "d"} else "white"
        ax.add_patch(Circle((x, y), 0.026, facecolor=fc,
                            edgecolor=BLUE if fc == BLUE else GRAY, linewidth=1.1))
    arrow(ax, nodes["d"], (0.78, 0.31), color=ORANGE, lw=2.2)
    ax.text(0.66, 0.42, "first exit: decodes", color=ORANGE,
            ha="center", va="bottom", weight="bold")
    ax.add_patch(Circle((0.80, 0.30), 0.028, facecolor=ORANGE,
                        edgecolor=ORANGE, linewidth=1.1))
    ax.text(0.08, 0.90, "sample p(root)", color=BLUE, fontsize=7)
    ax.text(0.28, 0.90, "sample p(the)", color=BLUE, fontsize=7)
    ax.text(0.50, 0.90, "sample p(the tree)", color=BLUE, fontsize=7)
    box(ax, (0.72, 0.05), (0.25, 0.14), "emitted block", PURPLE,
        subtitle="the | tree | decodes")
    ax.text(0.02, 0.08, "Blue: accepted tree edges", color=BLUE, weight="bold")
    ax.text(0.02, 0.015, "Orange: target bonus token", color=ORANGE, weight="bold")
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "tree_block_walk.pdf", bbox_inches="tight")
    plt.close(fig)


def kernel_comparison():
    fig, ax = setup(7.0, 2.9)
    ax.text(0.04, 0.91, "Official DDTree verifier", color=GREEN, weight="bold", fontsize=9)
    ax.text(0.54, 0.91, "Persistent tree-block verifier", color=PURPLE, weight="bold", fontsize=9)
    # left lane
    box(ax, (0.04, 0.67), (0.38, 0.13), "All FP64 probability rows", BLUE)
    arrow(ax, (0.23, 0.67), (0.23, 0.57))
    box(ax, (0.04, 0.42), (0.38, 0.13), "Batched categorical on every row", GREEN)
    arrow(ax, (0.23, 0.42), (0.23, 0.32))
    box(ax, (0.04, 0.17), (0.38, 0.13), "Transfer row samples; follow path", GREEN)
    # right lane
    box(ax, (0.54, 0.67), (0.41, 0.13), "Same FP64 probability rows", BLUE)
    arrow(ax, (0.745, 0.67), (0.745, 0.57))
    box(ax, (0.54, 0.42), (0.41, 0.13), "One CUDA block: scan reached row", PURPLE)
    arrow(ax, (0.745, 0.42), (0.745, 0.32))
    box(ax, (0.54, 0.17), (0.41, 0.13), "Device traversal; transfer packed path", PURPLE)
    ax.text(0.49, 0.49, "vs", ha="center", va="center", color=GRAY, weight="bold")
    ax.text(0.50, 0.05, "Identical tree and target law; only post-forward execution differs",
            ha="center", va="center", color=DARK, fontsize=7.5, weight="bold")
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "kernel_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def experiment_matrix():
    fig, ax = setup(7.1, 3.2)
    methods = ["Target", "DFlash", "DDTree", "AdaptiveTree", "TBV"]
    rows = ["4B / T=0", "8B / T=0", "4B / T=1", "8B / T=1"]
    enabled = {
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 4),
        (3, 0), (3, 1), (3, 2), (3, 4),
    }
    x0, y0, cw, ch = 0.23, 0.76, 0.14, 0.16
    for j, name in enumerate(methods):
        ax.text(x0 + (j + 0.5) * cw, 0.88, name, ha="center", va="center",
                color=DARK, weight="bold", fontsize=7.4)
    for i, name in enumerate(rows):
        y = y0 - (i + 1) * ch
        ax.text(0.20, y + ch / 2, name, ha="right", va="center",
                color=DARK, weight="bold", fontsize=7.5)
        for j in range(len(methods)):
            x = x0 + j * cw
            on = (i, j) in enabled
            edge = [BLUE, ORANGE, GREEN, GREEN, PURPLE][j] if on else "#D5DBE0"
            fill = "white" if on else "#F4F6F8"
            ax.add_patch(Rectangle((x, y), cw - 0.012, ch - 0.012,
                                   facecolor=fill, edgecolor=edge, linewidth=1.2 if on else 0.7))
            ax.text(x + (cw - 0.012) / 2, y + (ch - 0.012) / 2,
                    "" if on else "n/a", ha="center", va="center",
                    color="#AAB2B9", fontsize=6.4)
            if on:
                ax.plot([x + 0.03, x + cw - 0.04], [y + 0.045, y + 0.045],
                        color="#A6AFB8", lw=0.8)
    ax.text(0.50, 0.07, "Blank underline = reserved result field after audit",
            ha="center", va="center", color=GRAY, fontsize=7.2)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "experiment_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    system_overview()
    tree_block_walk()
    kernel_comparison()
    experiment_matrix()
