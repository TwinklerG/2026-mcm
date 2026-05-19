"""
Draw one correlation matrix heatmap using all numeric variables from preprocessed data.
Merges week_summary, contestants, season_stats, partner_stats and selects numeric columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PROCESSED = SCRIPT_DIR / "processed"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def _setup_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="sans-serif",
        font_scale=1.05,
        rc={
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.4,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        },
    )
    plt.rcParams["axes.titleweight"] = "normal"
    plt.rcParams["axes.titlelocation"] = "left"


def main() -> None:
    _setup_style()

    week = pl.read_csv(DATA_PROCESSED / "week_summary.csv")
    contestants = pl.read_csv(DATA_PROCESSED / "contestants.csv")
    season = pl.read_csv(DATA_PROCESSED / "season_stats.csv")
    partner = pl.read_csv(DATA_PROCESSED / "partner_stats.csv")

    # Merge: week_summary + contestants (ballroom_partner, age) + partner_stats + season_stats
    cont_sub = contestants.select(
        ["contestant_id", "ballroom_partner", "celebrity_age_during_season"]
    )
    merged = week.join(cont_sub, on="contestant_id", how="left")
    merged = merged.join(partner, on="ballroom_partner", how="left")

    season_num = season.select(
        [
            "contestant_id",
            "weeks_competed",
            "season_total_score",
            "season_avg_weekly_score",
            "season_avg_judge_score",
            "score_volatility",
            "avg_rank",
            "weeks_ranked_first",
            "placement",
        ]
    )
    merged = merged.join(season_num, on="contestant_id", how="left")

    # Keep only numeric columns (Int, Float), drop ids and names
    exclude = {"contestant_id", "celebrity_name", "ballroom_partner", "season", "week"}
    num_cols = [
        c
        for c in merged.columns
        if c not in exclude
        and merged[c].dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.Float32, pl.Float64)
    ]
    sub = merged.select(num_cols).drop_nulls()
    if sub.height < 10:
        print("Too few rows after drop_nulls, skipping.")
        return

    cols = sub.columns
    data = np.column_stack([sub[c].to_numpy() for c in cols])
    corr_arr, _ = stats.spearmanr(data)
    if np.isscalar(corr_arr):
        corr_arr = np.array([[1, corr_arr], [corr_arr, 1]])

    n = len(cols)
    figsize = (max(10, n * 0.55), max(8, n * 0.5))
    fig, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(250, 15, s=75, l=50, as_cmap=True)
    mask = np.triu(np.ones_like(corr_arr, dtype=bool), k=1)
    sns.heatmap(
        corr_arr,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        annot=True,
        fmt=".2f",
        annot_kws={"size": max(6, 12 - n // 4)},
        ax=ax,
        cbar_kws={"label": "Spearman correlation", "shrink": 0.7},
    )
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(cols, rotation=0, fontsize=10)
    ax.set_title(
        "All preprocessed variables: Spearman correlation matrix",
        fontsize=16,
        fontweight="bold",
        loc="center",
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label("Spearman correlation", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    out = FIGURES_DIR / "06_correlation_matrix_all.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
