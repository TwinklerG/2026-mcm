"""Enhanced visualization for controversy cases with multiple ranking methods.

For each controversy case, creates a line plot with:
- X-axis: week (within season)
- Y-axis: rank (1 = best)
- Multiple lines: original rank, rank_rank, percent_rank, rev_rank_rank, rev_percent_rank
- Secondary info: number of contestants remaining in that week

Saves high-DPI PNG per case.
"""

from __future__ import annotations

import typing as t
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from consts import DATA_RES

# MCM-style (publication) defaults: serif fonts, larger labels, colorblind palette
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
    }
)
sns.set_style("white")
PALETTE = sns.color_palette("colorblind")

FIG_DIR = DATA_RES / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_case_rankings(
    df: pl.DataFrame,
    contestant_id: str,
    season: int,
    reason: str,
) -> Path:
    """Plot evolution of a contestant's rankings across weeks.

    Lines plotted (if available):
    - Original judge rank (judge_rank)
    - Combined methods: rank_rank, percent_rank, rev_rank_rank, rev_percent_rank

    Also show number of contestants remaining each week.
    """
    # Filter to this contestant
    dfc = df.filter(
        (pl.col("contestant_id") == contestant_id) & (pl.col("season") == season)
    ).sort("week")

    if dfc.height == 0:
        print(f"[!] No data for {contestant_id} (S{season})")
        return None

    # Convert to pandas for easier plotting
    pdf = dfc.to_pandas().sort_values("week")

    # Identify available rank columns (exclude 'rev_' ranks per request)
    rank_cols = []
    for col in ["judge_rank", "fan_rank", "rank_rank", "percent_rank", "week_left"]:
        if col in pdf.columns and pdf[col].notna().any():
            rank_cols.append(col)

    if not rank_cols:
        print(f"[!] No rank columns found for {contestant_id}")
        return None

    # Apply seaborn paper-style theme for scientific look
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        },
    )

    # Create figure with primary (ranking) and secondary (count) y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot rankings on primary y-axis with clear markers and thicker lines
    palette = sns.color_palette("tab10", n_colors=len(rank_cols))
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, col in enumerate(rank_cols):
        if pdf[col].notna().any():
            ax1.plot(
                pdf["week"],
                pdf[col],
                marker=markers[i % len(markers)],
                label=col.replace("_", " "),
                color=palette[i],
                linewidth=2 if col in ("rank_rank", "percent_rank") else 0.5,
                markersize=6,
                markeredgewidth=0.8,
                markeredgecolor="white",
            )

    ax1.set_xlabel("Week", fontsize=12)
    ax1.set_ylabel("Ranking (lower is better)", fontsize=12)
    ax1.invert_yaxis()  # Invert y-axis so rank 1 is at top
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="upper left", fontsize=9, frameon=True)

    # --- 新增：填充不可能到达的区域 ---
    if "week_left" in pdf.columns:
        # x轴：周数
        x = pdf["week"]
        # y1: 剩余人数 (排名不可能比这个数字更大)
        # y2: 坐标轴底部 (在反转轴中，这通常是一个很大的排名数字)
        y_limit = pdf["week_left"]

        # 获取当前y轴的显示范围，确保阴影填满到底部
        current_ylim = ax1.get_ylim()
        bottom = max(current_ylim)

        ax1.fill_between(
            x,
            y_limit,
            bottom,
            color="gray",
            alpha=0.2,
            hatch="//",
            label="Impossible Rank Range",
        )
    # ------------------------------

    # Secondary y-axis for number of contestants (if available)
    if any(c in pdf.columns for c in ["judge_rank", "fan_rank"]):
        ax2 = ax1.twinx()
        # Estimate contestants by max rank across available rank cols per row
        max_rank = int(np.nanmax(pdf[rank_cols].values)) if len(rank_cols) > 0 else None
        if max_rank is not None and max_rank > 0:
            ax2.set_ylim(max_rank + 0.5, 0.5)  # reverse scale for contestants
            ax2.set_ylabel("Estimated contestants remaining", fontsize=11, color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")

    name = contestant_id.split("_S")[0]
    # Mark elimination week (if contestant leaves before season finale)
    try:
        season_max_week = int(
            df.filter(pl.col("season") == season)
            .select(pl.col("week"))
            .to_series()
            .max()
        )
    except Exception:
        season_max_week = int(pdf["week"].max())

    contestant_last_week = int(pdf["week"].max())
    if contestant_last_week < season_max_week:
        elim_week = contestant_last_week
        # vertical line
        ax1.axvline(elim_week, color="red", linestyle="--", linewidth=1)
        # mark each available rank at elimination week
        for col in rank_cols:
            if (
                col in pdf.columns
                and not pd.isna(pdf.loc[pdf["week"] == elim_week, col]).all()
            ):
                yvals = pdf.loc[pdf["week"] == elim_week, col].values
                if len(yvals) > 0:
                    y = float(yvals[0])
                    ax1.scatter(
                        [elim_week], [y], color="red", marker="X", s=80, zorder=5
                    )
                    ax1.annotate(
                        "Eliminated",
                        xy=(elim_week, y),
                        xytext=(elim_week + 0.3, y + 1),
                        color="red",
                        fontsize=10,
                        arrowprops={
                            "arrowstyle": "->",
                            "color": "red",
                            "linewidth": 0.8,
                        },
                    )
                    break
    else:
        # finale / survived to end
        # ax1.annotate(
        #     "Finale",
        #     xy=(contestant_last_week, pdf[rank_cols[0]].iloc[-1]),
        #     xytext=(contestant_last_week - 1, pdf[rank_cols[0]].iloc[-1] - 1),
        #     color="black",
        #     fontsize=10,
        # )
        pass

    for row in pdf.itertuples():
        if row.rank_rank_is_eliminated:
            ax1.annotate(
                "Eliminated",
                xy=(row.week, row.rank_rank),
                xytext=(row.week + 0.3, row.rank_rank + 1),
                color="red",
                fontsize=10,
                arrowprops={
                    "arrowstyle": "->",
                    "color": "red",
                    "linewidth": 0.8,
                },
            )

        if row.percent_rank_is_eliminated:
            ax1.annotate(
                "Eliminated",
                xy=(row.week, row.percent_rank),
                xytext=(row.week + 0.3, row.percent_rank + 1),
                color="red",
                fontsize=10,
                arrowprops={
                    "arrowstyle": "->",
                    "color": "red",
                    "linewidth": 0.8,
                },
            )

    plt.title(f"{name} (Season {season}) - {reason}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Save
    filename = f"case_{name.lower().replace(' ', '_')}_s{season}_rankings.png"
    out = FIG_DIR / filename
    # Save high-quality PNG for publication (PDF output disabled)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    return out


def plot_all_controversy_cases(
    df: pl.DataFrame,
    cases: t.List[t.Tuple[str, int, str]],
) -> t.List[Path]:
    """Generate ranking evolution plots for all controversy cases.

    Args:
        df: DataFrame with rankings data
        cases: List of (name, season, reason) tuples

    Returns:
        List of saved file paths
    """
    saved = []
    for name, season, reason in cases:
        cid = f"{name}_S{season}"
        try:
            path = plot_case_rankings(df, cid, season, reason)
            if path:
                saved.append(path)
                print(f"[✓] Saved: {path}")
        except Exception as e:
            print(f"[✗] Failed for {cid}: {e}")

    return saved


def main():
    # Load data (criteria or ranks)
    try:
        df = pl.read_csv(DATA_RES / "criteria.csv")
        print("[✓] Loaded criteria.csv")
    except FileNotFoundError:
        df = pl.read_csv(DATA_RES / "ranks.csv")
        print("[✓] Loaded ranks.csv")

    # Define controversy cases
    cases = [
        ("Jerry Rice", 2, "Runner-up but low judge scores in week 5"),
        ("Billy Ray Cyrus", 4, "5th place but low judge scores in week 6"),
        ("Bristol Palin", 11, "Third place but low judge scores in 12 weeks"),
        ("Bobby Bones", 27, "Champion but consistently low judge scores"),
    ]

    # Generate plots
    print("\n[*] Generating ranking evolution plots...")
    results = plot_all_controversy_cases(df, cases)

    print(f"\n[✓] Generated {len(results)} plots in {FIG_DIR}")


if __name__ == "__main__":
    main()
