"""Visualization utilities for task2 (ranks / controversy).

Generates and saves:
- scatter of `fan_percent` vs `judge_percent` with regression
- time-series for controversy cases (fan vs judge over weeks)
- per-week Spearman rank correlation (agreement) plot

Saves PNG files into `DATA_RES / figures` with high DPI.
"""

from __future__ import annotations

import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from consts import DATA_RES

FIG_DIR = DATA_RES / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def scatter_fan_vs_judge(df: pl.DataFrame, filename: str = "fan_vs_judge.png") -> Path:
    """Scatter plot of fan_percent vs judge_percent with regression line.

    Returns saved file path.
    """
    pdf = df.select(["fan_percent", "judge_percent"]).drop_nulls()
    if pdf.height == 0:
        raise ValueError("no data for scatter plot")

    pd = pdf.to_pandas()

    plt.figure(figsize=(6, 6), dpi=300)
    sns.regplot(x="fan_percent", y="judge_percent", data=pd, scatter_kws={"s": 10, "alpha": 0.6})
    plt.xlabel("Fan percent")
    plt.ylabel("Judge percent")
    plt.title("Fan percent vs Judge percent")
    out = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def time_series_cases(df: pl.DataFrame, cases: t.List[t.Tuple[str, int, str]]) -> t.List[Path]:
    """Plot fan_percent and judge_percent over weeks for each controversy case separately.

    `cases` is list of (name, season, reason).
    Returns list of saved file paths (one per case).
    """
    saved_files = []

    for case in cases:
        name, season, reason = case
        cid = f"{name}_S{season}"
        dfc = df.filter(pl.col("contestant_id") == cid).sort("week")
        if dfc.height == 0:
            print(f"[WARNING] No data for {cid}")
            continue

        pd = dfc.select(["week", "fan_percent", "judge_percent"]).to_pandas()

        # separate figure per case
        plt.figure(figsize=(8, 5), dpi=300)
        plt.plot(pd["week"], pd["fan_percent"], marker="o", linewidth=2, markersize=8, label="fan_percent")
        plt.plot(pd["week"], pd["judge_percent"], marker="s", linewidth=2, markersize=8, label="judge_percent")
        plt.xlabel("Week")
        plt.ylabel("Percent")
        plt.title(f"{name} (Season {season}) — {reason}")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # filename based on case
        filename = f"case_{name.lower().replace(' ', '_')}_s{season}.png"
        out = FIG_DIR / filename
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        saved_files.append(out)
        print(f"Saved: {out}")

    return saved_files


def weekly_spearman(df: pl.DataFrame, filename: str = "weekly_spearman.png") -> Path:
    """Compute per-(season,week) Spearman correlation between fan_rank and judge_rank, plot as histogram and time series.
    """
    # compute spearman per season-week
    groups = df.group_by(["season", "week"]).agg(
        [pl.col("fan_rank"), pl.col("judge_rank")]
    )

    seasons = []
    weeks = []
    rhos = []
    from scipy.stats import spearmanr

    for row in groups.iter_rows(named=True):
        fr = np.array(row["fan_rank"])
        jr = np.array(row["judge_rank"])
        if len(fr) < 2:
            continue
        # 检查是否为常数（标准差为0）
        if np.std(fr) == 0 or np.std(jr) == 0:
            continue
        rho, p = spearmanr(fr, jr)
        seasons.append(row["season"])
        weeks.append(row["week"])
        rhos.append(float(rho) if not np.isnan(rho) else 0.0)

    if not rhos:
        raise ValueError("no groups for spearman calculation")

    # histogram + time series
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=300)
    axes[0].hist(rhos, bins=25, color="#3b85c3", edgecolor="k")
    axes[0].set_title("Distribution of Spearman rho (fan_rank vs judge_rank) per week")
    axes[0].set_xlabel("Spearman rho")

    # time series: plot mean rho per week index across seasons
    import pandas as pd
    df_rho = pd.DataFrame({"season": seasons, "week": weeks, "rho": rhos})
    # aggregate by week number (across seasons)
    weekly = df_rho.groupby("week").rho.mean().reset_index()
    axes[1].plot(weekly["week"], weekly["rho"], marker="o")
    axes[1].set_xlabel("week")
    axes[1].set_ylabel("mean Spearman rho")
    axes[1].set_title("Mean Spearman rho by week (across seasons)")

    out = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def main():
    # load ranks
    ranks = pl.read_csv(DATA_RES / "ranks.csv")

    # scatter
    try:
        p1 = scatter_fan_vs_judge(ranks, "fan_vs_judge.png")
        print(f"[✓] Scatter plot: {p1}")
    except Exception as e:
        print(f"[✗] scatter failed: {e}")

    # controversy timeseries (separate files per case)
    cases = [
        ("Jerry Rice", 2, "Runner-up but low judge scores in week 5"),
        ("Billy Ray Cyrus", 4, "5th place but low judge scores in week 6"),
        ("Bristol Palin", 11, "Third place but low judge scores in 12 weeks"),
        ("Bobby Bones", 27, "Champion but consistently low judge scores"),
    ]

    try:
        saved = time_series_cases(ranks, cases)
        print(f"[✓] Time-series plots: {len(saved)} file(s) saved")
    except Exception as e:
        print(f"[✗] timeseries failed: {e}")

    # weekly spearman
    try:
        p3 = weekly_spearman(ranks, "weekly_spearman.png")
        print(f"[✓] Weekly Spearman: {p3}")
    except Exception as e:
        print(f"[✗] weekly spearman failed: {e}")


if __name__ == "__main__":
    main()
