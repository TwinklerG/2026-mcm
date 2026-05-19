"""
Task 1 evaluation metrics: figures for the paper.

- Elimination accuracy by season (stacked bar).
- Rank correlation (Kendall τ) by season.
- Posterior consistency / ESS: only aggregates in JSON — no per-season data;
  single bar or two bars are meaningless; report in table/text, no figure.
- Uncertainty: CI width by industry (distribution) + high-uncertainty cases by season (change).

Style aligned with draft/wmh/data/plot_eliminations.py: paper context, white style,
Google Material–style palette, alpha on bars to reduce saturation.
Saves figures to draft/wmh/task1/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
DATA_DIR = SCRIPT_DIR.parent / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Alpha for bars (match eliminations chart: softer, less saturated)
BAR_ALPHA = 0.85

# Style: paper, white, no top/right spines (like plot_eliminations)
def _setup_style() -> None:
    sns.set_theme(
        context="paper",
        style="white",
        font="sans-serif",
        font_scale=1.05,
        rc={
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        },
    )
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"


# Softer palette; blue = dark blue, red = slightly darker
COLOR_HIGH = "#66BB6A"      # green (Material Green 400, slightly darker)
COLOR_LOW = "#e53935"       # red (Material Red 600, slightly darker)
COLOR_BOTTOM2 = "#1565C0"   # dark blue (Material Blue 800)
REF_LINE_COLOR = "#757575"
RULE_LINE_COLOR = "#A1887F"  # brown/grey
ACC_THRESHOLD = 1.0


def load_results() -> dict:
    """Load results.json from outputs."""
    with (OUTPUTS_DIR / "results.json").open(encoding="utf-8") as f:
        return json.load(f)


def load_bottom2() -> dict:
    """Load bottom2_accuracy.json."""
    with (OUTPUTS_DIR / "bottom2_accuracy.json").open(encoding="utf-8") as f:
        return json.load(f)


def load_finale() -> dict:
    """Load finale_ranking.json."""
    with (OUTPUTS_DIR / "finale_ranking.json").open(encoding="utf-8") as f:
        return json.load(f)


def load_detailed_uncertainty() -> dict:
    """Load detailed_uncertainty.json."""
    with (OUTPUTS_DIR / "detailed_uncertainty.json").open(encoding="utf-8") as f:
        return json.load(f)


def plot_elimination_accuracy_by_season() -> None:
    """
    Elimination accuracy by season (stacked): strict accuracy (green/red) +
    bottom-2 recall only (blue). Style: paper/white, alpha on bars, white edges.
    """
    _setup_style()
    data = load_bottom2()
    per = data["per_season"]
    seasons = sorted([int(k) for k in per.keys()])
    acc = [per[str(s)]["strict_accuracy"] for s in seasons]
    b2_recall = [per[str(s)]["bottom2_recall"] for s in seasons]
    b2_only = [max(0, b2 - a) for a, b2 in zip(acc, b2_recall)]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(seasons))
    bar_width = 0.6
    colors_acc = [COLOR_HIGH if a >= ACC_THRESHOLD else COLOR_LOW for a in acc]
    ax.bar(
        x,
        acc,
        width=bar_width,
        color=colors_acc,
        edgecolor="white",
        linewidth=0.5,
        alpha=BAR_ALPHA,
    )
    ax.bar(
        x,
        b2_only,
        width=bar_width,
        bottom=acc,
        color=COLOR_BOTTOM2,
        edgecolor="white",
        linewidth=0.5,
        alpha=BAR_ALPHA,
    )
    ax.axhline(1.0, color=REF_LINE_COLOR, linestyle="--", linewidth=0.9, zorder=0)
    s28_idx = next((i for i, s in enumerate(seasons) if s == 28), None)
    if s28_idx is not None:
        ax.axvline(
            s28_idx - 0.5,
            color=RULE_LINE_COLOR,
            linestyle="--",
            linewidth=2.2,
            zorder=0,
        )
    ax.set_xticks(x[::2])
    ax.set_xticklabels([seasons[i] for i in range(0, len(seasons), 2)], fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=13)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.set_xlabel("Season", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
    ax.set_title("Elimination accuracy by season", fontsize=16, fontweight="bold", loc="center")
    ax.legend(
        handles=[
            Patch(facecolor=COLOR_HIGH, edgecolor="white", alpha=BAR_ALPHA, label="Acc = 100%"),
            Patch(facecolor=COLOR_LOW, edgecolor="white", alpha=BAR_ALPHA, label="Acc < 100%"),
            Patch(facecolor=COLOR_BOTTOM2, edgecolor="white", alpha=BAR_ALPHA, label="Bottom-2"),
        ],
        loc="lower left",
        frameon=True,
        fontsize=12,
    )
    ax.set_ylim(0, 1.08)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "elimination_accuracy_by_season.png")
    plt.close(fig)


def plot_rank_correlation_by_season() -> None:
    """
    Metric 2: Rank correlation (Kendall τ) by season.
    Finale predicted vs actual ranking. Same style: alpha, white edge.
    """
    _setup_style()
    data = load_finale()
    taus = data["kendall_tau_values"]
    seasons = [x["season"] for x in taus]
    tau = [x["tau"] for x in taus]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        seasons,
        tau,
        color=COLOR_BOTTOM2,
        edgecolor="white",
        linewidth=0.5,
        alpha=BAR_ALPHA,
    )
    ax.axhline(1.0, color=REF_LINE_COLOR, linestyle="--", linewidth=1)
    ax.set_xlabel("Season", fontsize=14, fontweight="bold")
    ax.set_ylabel("Kendall τ", fontsize=14, fontweight="bold")
    ax.set_ylim(0.7, 1.05)
    ax.set_title(
        "Finale rank correlation (predicted vs actual) by season",
        fontsize=16,
        fontweight="bold",
        loc="center",
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rank_correlation_by_season.png")
    plt.close(fig)


def plot_posterior_consistency_summary() -> None:
    """
    Posterior consistency: only aggregate in results.json (no per-season/episode).
    Skip figure — single bar is meaningless; report metric in table/text.
    """
    pass


def plot_uncertainty_credible_intervals() -> None:
    """
    Uncertainty: (A) CI width by industry (distribution across groups),
    (B) High-uncertainty cases by season. Same style: alpha, white edge.
    """
    _setup_style()
    data = load_detailed_uncertainty()
    by_ind = data.get("by_industry", {})
    controversial = data.get("controversial_contestants", [])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.32)
    ax1, ax2 = axes

    if by_ind:
        names = list(by_ind.keys())
        means = [by_ind[n]["mean_ci_width"] for n in names]
        stds = [by_ind[n]["std_ci_width"] for n in names]
        order = sorted(range(len(names)), key=lambda i: means[i])
        names = [names[i] for i in order]
        means = [means[i] for i in order]
        stds = [stds[i] for i in order]
        y_pos = range(len(names))
        ax1.barh(
            y_pos,
            means,
            xerr=stds,
            color=COLOR_BOTTOM2,
            edgecolor="white",
            linewidth=0.5,
            capsize=2,
            alpha=BAR_ALPHA,
        )
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel("Mean 95% CI width (± std)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Industry", fontsize=12, fontweight="bold")
        ax1.set_title("Posterior uncertainty by industry", fontsize=14, fontweight="bold")
        ax1.set_axisbelow(True)
        ax1.invert_yaxis()

    if controversial:
        seasons = [c["season"] for c in controversial]
        widths = [c["ci_width"] for c in controversial]
        ax2.scatter(
            seasons,
            widths,
            color=COLOR_LOW,
            s=28,
            alpha=BAR_ALPHA,
            edgecolors="white",
            linewidths=0.5,
        )
        ax2.axhline(
            data["overall"]["mean_ci_width"],
            color=REF_LINE_COLOR,
            linestyle="--",
            linewidth=1,
            label="Overall mean",
        )
        ax2.set_xlabel("Season", fontsize=12, fontweight="bold")
        ax2.set_ylabel("95% CI width", fontsize=12, fontweight="bold")
        ax2.set_title("High-uncertainty cases by season", fontsize=14, fontweight="bold")
        ax2.legend(loc="upper right", frameon=True, framealpha=0.95)
        ax2.set_axisbelow(True)

    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "uncertainty_credible_intervals.png")
    plt.close(fig)


def main() -> None:
    """Generate metric figures (no single-bar / two-bar aggregates)."""
    plot_elimination_accuracy_by_season()
    plot_rank_correlation_by_season()
    plot_posterior_consistency_summary()  # no-op: no figure for aggregate-only metric
    plot_uncertainty_credible_intervals()
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
