"""
Eliminations-per-week figure for MCM 2026 Problem C (DWTS).

Weekly elimination count is not fixed (1, 2, or 3 per week).
Heatmap: season × week, color = number eliminated (0/1/2/3).
Uses Google Material green–yellow sequential palette and seaborn heatmap best practices.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.colors import ListedColormap

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Google Material green → yellow sequential (0=neutral, 1–3=light green → dark green → yellow)
# Material: Green 50 #E8F5E9, 100 #C8E6C9, 300 #81C784, 800 #2E7D32; Yellow 400 #FBC02D
CMAP_VALS = [
    "#fafafa",
    "#C8E6C9",
    "#81C784",
    "#FBC02D",
]  # neutral, light green, green, yellow


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
        },
    )
    plt.rcParams["axes.titleweight"] = "normal"
    plt.rcParams["axes.titlelocation"] = "left"


def fig_eliminations_heatmap() -> None:
    """Heatmap: season (rows) × week (columns), Google green–yellow, annot + linewidths."""
    _setup_style()
    pivot = pl.read_csv(OUTPUTS_DIR / "eliminations_per_week.csv")
    week_cols = [c for c in pivot.columns if c != "season"]
    arr = pivot.select(week_cols).to_numpy().astype(int)
    seasons = pivot["season"].to_list()
    xticklabels = [f"W{w}" for w in week_cols]
    yticklabels = [str(s) for s in seasons]

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = ListedColormap(CMAP_VALS)
    sns.heatmap(
        arr,
        ax=ax,
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
        annot=True,
        fmt="d",
        annot_kws={"size": 8, "weight": "normal"},
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Eliminations", "shrink": 0.6, "ticks": [0, 1, 2, 3]},
        xticklabels=xticklabels,
        yticklabels=yticklabels,
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Season")
    ax.set_title("Eliminations per week by season")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "08_eliminations_per_week_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: 08_eliminations_per_week_heatmap.png")


def fig_eliminations_distribution() -> None:
    """Bar chart: number of (season, week) cells with 1, 2, or 3 eliminations (green–yellow)."""
    _setup_style()
    with open(OUTPUTS_DIR / "eliminations_per_week.json", encoding="utf-8") as f:
        data = json.load(f)
    dist = data["distribution"]
    labels = [str(d["n_eliminated"]) for d in dist]
    counts = [d["count"] for d in dist]
    colors = [CMAP_VALS[i] for i in range(1, len(counts) + 1)]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(labels))
    ax.bar(x, counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab} per week" for lab in labels])
    ax.set_ylabel("Number of (season, week) cells")
    ax.set_title("Distribution of eliminations per week")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "09_eliminations_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Saved: 09_eliminations_distribution.png")


def fig_eliminations_by_week() -> None:
    """Alternative: stacked bar by week — for each week, how many seasons had 1 vs 2 vs 3 eliminations."""
    _setup_style()
    pivot = pl.read_csv(OUTPUTS_DIR / "eliminations_per_week.csv")
    week_cols = [c for c in pivot.columns if c != "season"]
    weeks = [int(w) for w in week_cols]
    counts_1 = [pivot.filter(pl.col(w) == 1).height for w in week_cols]
    counts_2 = [pivot.filter(pl.col(w) == 2).height for w in week_cols]
    counts_3 = [pivot.filter(pl.col(w) == 3).height for w in week_cols]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(weeks))
    w_bar = 0.6
    ax.bar(
        x,
        counts_1,
        width=w_bar,
        label="1 elimination",
        color=CMAP_VALS[1],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x,
        counts_2,
        width=w_bar,
        bottom=counts_1,
        label="2 eliminations",
        color=CMAP_VALS[2],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x,
        counts_3,
        width=w_bar,
        bottom=np.array(counts_1) + np.array(counts_2),
        label="3 eliminations",
        color=CMAP_VALS[3],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"W{w}" for w in weeks], fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=13)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.set_xlabel("Week", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of seasons", fontsize=14, fontweight="bold")
    ax.set_title(
        "Eliminations per week: distribution by week",
        fontsize=16,
        fontweight="bold",
        loc="center",
    )
    ax.legend(loc="upper right", frameon=True, fontsize=12)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "10_eliminations_by_week_stacked.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: 10_eliminations_by_week_stacked.png")


def main() -> None:
    fig_eliminations_heatmap()
    fig_eliminations_distribution()
    fig_eliminations_by_week()
    print("Done.")


if __name__ == "__main__":
    main()
