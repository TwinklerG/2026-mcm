"""
Standalone script: distribution of judge share vs fan share in one chosen week.

- Judge share = judge score as percentage within that week (sum = 1).
- Fan share = estimated fraction of fan votes in that week (sum = 1).
- Two figures: bar chart of shares per contestant in the same week.

Output: figures/judge_share_distribution.png, figures/fan_share_distribution.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

COLOR_JUDGE = "#1565C0"
COLOR_FAN = "#C62828"

# Pick one week for the distribution (season, week)
SEASON, WEEK = 10, 5


def main() -> None:
    panel_path = DATA_DIR / "panel_data.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel data not found: {panel_path}")

    df = pd.read_csv(panel_path)
    season_use, week_use = SEASON, WEEK
    one = df[(df["season"] == season_use) & (df["week"] == week_use)].copy()
    if one.empty:
        grp = df.groupby(["season", "week"]).size().reset_index(name="n")
        row = grp.iloc[grp["n"].argmax()]
        season_use, week_use = int(row["season"]), int(row["week"])
        one = df[(df["season"] == season_use) & (df["week"] == week_use)].copy()
    one = one.sort_values("contestant_id").reset_index(drop=True)

    tot_judge = one["judge_score"].sum()
    one["judge_share"] = (one["judge_score"] / tot_judge).clip(0, 1)
    if "estimated_fan_share" in one.columns:
        one["fan_share"] = one["estimated_fan_share"].clip(0, 1)
    else:
        one["fan_share"] = (np.exp(one["log_fan_share"]) - 1e-6).clip(0, 1)

    n = len(one)
    x = np.arange(n)
    labels = [str(c)[:14] for c in one["contestant_id"]]

    # Figure 1: Judge share in this week (per contestant)
    fig1, ax1 = plt.subplots(figsize=(max(6, n * 0.5), 4))
    ax1.bar(x, one["judge_share"], color=COLOR_JUDGE, alpha=0.88, edgecolor="#37474F", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax1.set_xlabel("Contestant", fontsize=14)
    ax1.set_ylabel("Judge Share", fontsize=14)
    ax1.set_title(f"Judge Share by Contestant (Season {season_use}, Week {week_use})", fontsize=15, fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.grid(True, axis="y", alpha=0.4)
    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / "judge_share_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {FIGURES_DIR / 'judge_share_distribution.png'}")

    # Figure 2: Fan share in this week (per contestant)
    fig2, ax2 = plt.subplots(figsize=(max(6, n * 0.5), 4))
    ax2.bar(x, one["fan_share"], color=COLOR_FAN, alpha=0.88, edgecolor="#37474F", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax2.set_xlabel("Contestant", fontsize=14)
    ax2.set_ylabel("Fan Share", fontsize=14)
    ax2.set_title(f"Fan Share by Contestant (Season {season_use}, Week {week_use})", fontsize=15, fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis="y", alpha=0.4)
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "fan_share_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {FIGURES_DIR / 'fan_share_distribution.png'}")


if __name__ == "__main__":
    main()
