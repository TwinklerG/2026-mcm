"""
Task 2: Line charts for four controversial contestants from ranks.csv.

Plots rank_rank (Rank method) and percent_rank (Percentage method) over weeks.
Gray area = elimination zone (ranks that get eliminated each week, from Task 1 left_contestants); only rank axis, no second y-axis.
Output: figures/controversial_*_rank_over_weeks.png (one per contestant),
        figures/controversial_four_panel.png (2x2 combined),
        figures/controversial_*_rank_over_weeks.svg (simplified SVG, one per contestant).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
RANKS_PATH = SCRIPT_DIR / "ranks.csv"
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

CONTROVERSIAL: list[tuple[str, int]] = [
    ("Jerry Rice", 2),
    ("Billy Ray Cyrus", 4),
    ("Bristol Palin", 11),
    ("Bobby Bones", 27),
]


def contestant_id(name: str, season: int) -> str:
    return f"{name}_S{season}"


def plot_one(
    df: pl.DataFrame,
    name: str,
    season: int,
    out_path: Path,
) -> None:
    """Plot rank_rank, percent_rank; gray = elimination zone (ranks eliminated each week) on rank axis only."""
    cid = contestant_id(name, season)
    sub = df.filter(pl.col("contestant_id") == cid).sort("week")
    if sub.height == 0:
        return
    weeks = sub["week"].to_list()
    rank_rank = sub["rank_rank"].to_list()
    percent_rank = sub["percent_rank"].to_list()
    left_contestants = sub["left_contestants"].to_list()
    n_weeks = len(weeks)
    week_elim = [
        left_contestants[i] - left_contestants[i + 1]
        for i in range(n_weeks - 1)
    ]
    week_elim.append(0)
    elim_cutoff = [
        max(1, n - k + 1) for n, k in zip(left_contestants, week_elim)
    ]
    y_bottom = max(left_contestants) + 1
    # Truncate at touch: do not draw past the week when rank enters elimination zone
    t_rank_out = next((t for t in range(n_weeks) if rank_rank[t] >= elim_cutoff[t]), n_weeks)
    t_pct_out = next((t for t in range(n_weeks) if percent_rank[t] >= elim_cutoff[t]), n_weeks)
    weeks_rank = weeks[: t_rank_out + 1]
    rank_rank_trunc = rank_rank[: t_rank_out + 1]
    weeks_pct = weeks[: t_pct_out + 1]
    percent_rank_trunc = percent_rank[: t_pct_out + 1]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.fill_between(
        weeks, elim_cutoff, y_bottom, color="gray", alpha=0.35, label="Elimination zone (touch = out)"
    )
    ax.plot(weeks, elim_cutoff, color="gray", linewidth=1.5, linestyle="-", label="Elimination cutoff")
    ax.plot(
        weeks_rank, rank_rank_trunc, marker="o", linewidth=2, markersize=6, label="Rank method"
    )
    ax.plot(
        weeks_pct,
        percent_rank_trunc,
        color="red",
        marker="s",
        linewidth=2,
        markersize=6,
        label="Percentage method",
    )
    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("Rank (1 = best)", fontsize=12)
    ax.set_title(f"{name} (Season {season})", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.invert_yaxis()
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: "" if x == 0 else str(int(x))))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_one_svg(
    df: pl.DataFrame,
    name: str,
    season: int,
    out_path: Path,
) -> None:
    """Simplified single-contestant plot for PPT: same data as plot_one, minimal decor, SVG output."""
    cid = contestant_id(name, season)
    sub = df.filter(pl.col("contestant_id") == cid).sort("week")
    if sub.height == 0:
        return
    weeks = sub["week"].to_list()
    rank_rank = sub["rank_rank"].to_list()
    percent_rank = sub["percent_rank"].to_list()
    left_contestants = sub["left_contestants"].to_list()
    n_weeks = len(weeks)
    week_elim = [
        left_contestants[i] - left_contestants[i + 1]
        for i in range(n_weeks - 1)
    ]
    week_elim.append(0)
    elim_cutoff = [
        max(1, n - k + 1) for n, k in zip(left_contestants, week_elim)
    ]
    y_bottom = max(left_contestants) + 1
    t_rank_out = next((t for t in range(n_weeks) if rank_rank[t] >= elim_cutoff[t]), n_weeks)
    t_pct_out = next((t for t in range(n_weeks) if percent_rank[t] >= elim_cutoff[t]), n_weeks)
    weeks_rank = weeks[: t_rank_out + 1]
    rank_rank_trunc = rank_rank[: t_rank_out + 1]
    weeks_pct = weeks[: t_pct_out + 1]
    percent_rank_trunc = percent_rank[: t_pct_out + 1]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.fill_between(weeks, elim_cutoff, y_bottom, color="gray", alpha=0.3)
    ax.plot(weeks, elim_cutoff, color="gray", linewidth=1.2, linestyle="-")
    ax.plot(weeks_rank, rank_rank_trunc, color="#2563EB", marker="o", linewidth=2, markersize=5)
    ax.plot(weeks_pct, percent_rank_trunc, color="#DC2626", marker="s", linewidth=2, markersize=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_ylim(bottom=0)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    n_w = len(weeks)
    ax.set_xlim(0.5, n_w + 0.5)
    # X-axis arrow (right end of bottom axis), in axes fraction
    ax.annotate(
        "", xy=(1.02, 0), xytext=(0.98, 0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=2, color="#333333"),
        annotation_clip=False,
    )
    # Y-axis arrow (top end of left axis), in axes fraction
    ax.annotate(
        "", xy=(0, 1.02), xytext=(0, 0.98),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=2, color="#333333"),
        annotation_clip=False,
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)


def plot_four_panel(df: pl.DataFrame, out_path: Path) -> None:
    """Plot 2x2 panel: one subplot per contestant; gray = elimination zone (rank axis only)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    for idx, (name, season) in enumerate(CONTROVERSIAL):
        ax = axes[idx // 2, idx % 2]
        cid = contestant_id(name, season)
        sub = df.filter(pl.col("contestant_id") == cid).sort("week")
        if sub.height == 0:
            ax.set_title(f"{name} (S{season}) — no data")
            continue
        weeks = sub["week"].to_list()
        left_contestants = sub["left_contestants"].to_list()
        n_weeks = len(weeks)
        week_elim = [
            left_contestants[i] - left_contestants[i + 1]
            for i in range(n_weeks - 1)
        ]
        week_elim.append(0)
        elim_cutoff = [
            max(1, n - k + 1) for n, k in zip(left_contestants, week_elim)
        ]
        y_bottom = max(left_contestants) + 1
        rr = sub["rank_rank"].to_list()
        pr = sub["percent_rank"].to_list()
        # Truncate at touch: do not draw past the week when rank enters elimination zone
        t_rank_out = next((t for t in range(n_weeks) if rr[t] >= elim_cutoff[t]), n_weeks)
        t_pct_out = next((t for t in range(n_weeks) if pr[t] >= elim_cutoff[t]), n_weeks)
        weeks_rank = weeks[: t_rank_out + 1]
        weeks_pct = weeks[: t_pct_out + 1]
        ax.fill_between(
            weeks, elim_cutoff, y_bottom, color="gray", alpha=0.35, label="Elim. zone (touch = out)"
        )
        ax.plot(weeks, elim_cutoff, color="gray", linewidth=1.2, linestyle="-", label="Cutoff")
        ax.plot(
            weeks_rank,
            rr[: t_rank_out + 1],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label="Rank",
        )
        ax.plot(
            weeks_pct,
            pr[: t_pct_out + 1],
            color="red",
            marker="s",
            linewidth=1.5,
            markersize=4,
            label="Pct",
        )
        ax.set_xlabel("Week", fontsize=11)
        ax.set_ylabel("Rank (1 = best)", fontsize=11)
        ax.set_title(f"{name} (Season {season})", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.invert_yaxis()
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: "" if x == 0 else str(int(x))))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pl.read_csv(RANKS_PATH)
    for name, season in CONTROVERSIAL:
        slug = name.lower().replace(" ", "_")
        out_path = FIG_DIR / f"controversial_{slug}_rank_over_weeks.png"
        plot_one(df, name, season, out_path)
        print("Saved:", out_path)
    for name, season in CONTROVERSIAL:
        slug = name.lower().replace(" ", "_")
        out_svg = FIG_DIR / f"controversial_{slug}_rank_over_weeks.svg"
        plot_one_svg(df, name, season, out_svg)
        print("Saved:", out_svg)
    plot_four_panel(df, FIG_DIR / "controversial_four_panel.png")
    print("Saved:", FIG_DIR / "controversial_four_panel.png")


if __name__ == "__main__":
    main()
