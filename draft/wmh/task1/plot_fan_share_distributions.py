"""
Task 1 predicted fan-share distributions for the paper.

1. One episode: bar chart of posterior mean fan share per contestant (one week).
2. One season: time series of posterior mean fan share vs week per contestant.
3. Predicted fan vote share distribution (Task 1): for each of 8 seasons, histogram of
   (contestant, week) posterior mean fan shares — 8 separate figures. This is the
   distribution of Task 1 predicted fan votes per season.

Style: paper/white, alpha, same palette as plot_metrics / plot_eliminations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

BAR_ALPHA = 0.85
COLOR_SURVIVED = "#81C784"
COLOR_ELIMINATED = "#e57373"
REF_LINE_COLOR = "#757575"


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


def _load_fan_shares() -> pl.DataFrame:
    return pl.read_csv(OUTPUTS_DIR / "fan_shares.csv")


def plot_fan_share_one_episode() -> None:
    """Bar chart: posterior mean fan share for one episode (e.g. S1 W6 finale)."""
    _setup_style()
    df = _load_fan_shares()
    # Season 1 Week 6: finale, 2 contestants (John O'Hurley, Kelly Monaco)
    ep = df.filter((pl.col("season") == 1) & (pl.col("week") == 6))
    if ep.height == 0:
        # Fallback: any episode with 3–6 contestants
        agg = (
            df.group_by(["season", "week"])
            .agg(pl.len().alias("n"))
            .filter(pl.col("n").is_between(3, 6))
            .head(1)
        )
        if agg.height == 0:
            return
        s, w = agg["season"][0], agg["week"][0]
        ep = df.filter((pl.col("season") == s) & (pl.col("week") == w))
    labels = [c.replace("_S1", "").replace("_S2", "") for c in ep["contestant_id"].to_list()]
    shares = ep["estimated_fan_share"].to_numpy()
    eliminated = ep["is_eliminated"].to_list()
    colors = [COLOR_ELIMINATED if e else COLOR_SURVIVED for e in eliminated]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = range(len(labels))
    ax.bar(
        x,
        shares,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        alpha=BAR_ALPHA,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Estimated fan share", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("Posterior mean fan share (Season 1, Week 6 finale)", fontsize=13, fontweight="bold")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fan_share_one_episode.png")
    plt.close(fig)
    print("Saved: fan_share_one_episode.png")


def plot_fan_share_one_season() -> None:
    """Time series: posterior mean fan share vs week for one season (e.g. S1)."""
    _setup_style()
    df = _load_fan_shares()
    s1 = df.filter(pl.col("season") == 1).sort(["week", "contestant_id"])
    if s1.height == 0:
        return
    contestants = s1["contestant_id"].unique().to_list()
    # Shorten labels
    short = [c.replace("_S1", "").replace("_S2", "") for c in contestants]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (cid, name) in enumerate(zip(contestants, short)):
        sub = s1.filter(pl.col("contestant_id") == cid)
        weeks = sub["week"].to_numpy()
        shares = sub["estimated_fan_share"].to_numpy()
        elim = sub["is_eliminated"].to_list()
        color = COLOR_ELIMINATED if any(e for e in elim) else COLOR_SURVIVED
        ax.plot(weeks, shares, "o-", label=name, color=color, alpha=BAR_ALPHA, linewidth=1.5, markersize=5)
    ax.set_xlabel("Week", fontsize=12, fontweight="bold")
    ax.set_ylabel("Estimated fan share", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 0.65)
    ax.set_title("Posterior mean fan share by week (Season 1)", fontsize=13, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fan_share_one_season.png")
    plt.close(fig)
    print("Saved: fan_share_one_season.png")


# 8 seasons for per-season histogram figures; same figsize and xlim, per-season ylim so S1 not too low
HISTOGRAM_SEASONS = (1, 5, 10, 15, 20, 25, 30, 34)
HIST_XMAX = 0.5  # do not extend to 1.0 so right side is not empty
HIST_FIGSIZE = (5, 3.5)
# More bins so more bars; min 15 bars, max 40; small-n seasons (e.g. S1) get ~2 points per bin
HIST_NBINS_MIN = 15
HIST_NBINS_MAX = 40


def plot_fan_share_histogram_per_season() -> None:
    """One histogram per season; same figsize and xlim, per-season ylim so each panel fills height."""
    _setup_style()
    df = _load_fan_shares()
    for season in HISTOGRAM_SEASONS:
        sub = df.filter(pl.col("season") == season)
        if sub.height == 0:
            continue
        shares = sub["estimated_fan_share"].to_numpy()
        n_bins = min(HIST_NBINS_MAX, max(HIST_NBINS_MIN, sub.height // 2))
        counts, bin_edges = np.histogram(shares, bins=n_bins, range=(0, HIST_XMAX))
        y_max_season = counts.max()
        y_lim = (0, y_max_season * 1.08)

        fig, ax = plt.subplots(figsize=HIST_FIGSIZE)
        ax.hist(
            shares,
            bins=n_bins,
            range=(0, HIST_XMAX),
            color="#81C784",
            edgecolor="white",
            alpha=BAR_ALPHA,
            linewidth=0.3,
        )
        ax.axvline(shares.mean(), color=REF_LINE_COLOR, linestyle="--", linewidth=1.2, label=f"Mean: {shares.mean():.3f}")
        ax.set_xlabel("Estimated fan share", fontsize=17, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=17, fontweight="bold")
        ax.set_title(f"Season {season}", fontsize=19, fontweight="bold")
        ax.set_xlim(0, HIST_XMAX)
        ax.set_ylim(y_lim)
        ax.legend(loc="upper right", fontsize=13)
        ax.tick_params(axis="both", labelsize=14)
        sns.despine(ax=ax)
        fig.tight_layout()
        out_name = f"fan_share_histogram_s{season:02d}.png"
        fig.savefig(FIGURES_DIR / out_name)
        plt.close(fig)
        print(f"Saved: {out_name}")


def main() -> None:
    plot_fan_share_one_episode()
    plot_fan_share_one_season()
    plot_fan_share_histogram_per_season()
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
