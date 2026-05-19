"""
Data Preprocessing Figures for MCM 2026 Problem C (DWTS).

Generates publication-quality figures for Section 3: Data Preprocessing.
Uses seaborn with colorblind-safe, print-friendly palettes (Okabe–Ito / Nature-style).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from scipy import stats

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_RAW = SCRIPT_DIR / "raw"
DATA_PROCESSED = SCRIPT_DIR / "processed"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Publication palette: Okabe–Ito (colorblind-safe, Nature/Science style)
# Order: orange, sky blue, bluish green, yellow, blue, vermillion, reddish purple
PALETTE_OKABE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]
PALETTE_POS_NEG = {"positive": "#D55E00", "negative": "#0072B2"}
PALETTE_VOTING = {
    "rank_v1": "#009E73",
    "percentage": "#0072B2",
    "rank_v2": "#CC79A7",
}
# Correlation heatmap: purple (negative) -> teal (zero) -> yellow (positive), no red/blue
CMAP_CORRELATION = LinearSegmentedColormap.from_list(
    "correlation_purple_teal_yellow",
    ["#4a148c", "#1a237e", "#006064", "#00897b", "#7cb342", "#fdd835"],
    N=256,
)


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


def fig_score_boxplot() -> None:
    """3.1: Score distribution by week (0 = eliminated, bonus > 10)."""
    _setup_style()
    df = pl.read_csv(DATA_PROCESSED / "scores_long.csv")
    df = df.filter(pl.col("score").is_not_null())

    plot_df = pd.DataFrame(
        {"week": df["week"].to_list(), "score": df["score"].to_list()}
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(
        data=plot_df,
        x="week",
        y="score",
        color=PALETTE_OKABE[1],
        saturation=0.85,
        width=0.6,
        linewidth=1.0,
        fliersize=2.5,
        ax=ax,
    )
    ax.axhline(y=0, color=".35", linestyle="--", linewidth=1)
    ax.axhline(
        y=10, color=PALETTE_OKABE[5], linestyle="--", linewidth=1, label="Bonus > 10"
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Judge score")
    ax.set_title("Score distribution by week (0 = eliminated)")
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor=".4")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_score_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: 01_score_boxplot.png")


def fig_missing_heatmap() -> None:
    """3.1: Missing value pattern (judges vary by season; zeros after elimination)."""
    _setup_style()
    df = pl.read_csv(
        DATA_RAW / "2026_MCM_Problem_C_Data.csv",
        null_values=["N/A", "NA", ""],
        infer_schema_length=10000,
    )
    score_cols = [c for c in df.columns if "judge" in c and "score" in c]
    sub = df.select(score_cols)
    mask_df = sub.select([pl.col(c).is_null() for c in score_cols])
    arr = mask_df.to_numpy().astype(float)

    fig, ax = plt.subplots(figsize=(12, 14))
    cmap = LinearSegmentedColormap.from_list("missing", ["#f5f5f5", "#525252"], N=2)
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    xtick_indices = [0, 11, 22, 33, len(score_cols) - 1]
    ax.set_xticks(xtick_indices)
    ax.set_xticklabels(["W1", "W4", "W7", "W10", "W11"])
    ax.set_xlabel("Week (score columns)")
    ax.set_ylabel("Contestant (row index)")
    ax.set_title("Missing values (dark = missing)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=25)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(["Present", "Missing"])
    sns.despine(ax=ax, left=True, bottom=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_missing_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: 02_missing_heatmap.png")


def fig_wide_to_long_schematic() -> None:
    """3.2: Wide-to-long transformation schematic."""
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.axis("off")

    wide_cols = ["week1_j1", "week1_j2", "week1_j3", "week2_j1", "..."]
    for i, c in enumerate(wide_cols):
        rect = Rectangle(
            (1, 7 - i * 1.2),
            6,
            0.8,
            facecolor=PALETTE_OKABE[1],
            alpha=0.75,
            edgecolor=".3",
            linewidth=0.8,
        )
        ax1.add_patch(rect)
        ax1.text(1.2, 7.4 - i * 1.2, c, va="center", fontsize=10)
    ax1.text(
        1, 8.5, "One row per contestant × many score columns", fontsize=9, color=".25"
    )

    long_cols = ["contestant_id", "season", "week", "judge", "score"]
    for i, c in enumerate(long_cols):
        rect = Rectangle(
            (1, 7 - i * 1.2),
            6,
            0.8,
            facecolor=PALETTE_OKABE[2],
            alpha=0.75,
            edgecolor=".3",
            linewidth=0.8,
        )
        ax2.add_patch(rect)
        ax2.text(1.2, 7.4 - i * 1.2, c, va="center", fontsize=10)
    ax2.text(1, 8.5, "One row per (contestant × week × judge)", fontsize=9, color=".25")

    ax1.set_title("Before: wide format", fontsize=11, pad=8)
    ax2.set_title("After: long format", fontsize=11, pad=8)
    fig.suptitle("Data transformation: wide → long", fontsize=12, y=1.02, x=0.5)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "03_wide_to_long_schematic.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Saved: 03_wide_to_long_schematic.png")


def fig_industry_distribution() -> None:
    """3.3: Celebrity industry distribution."""
    _setup_style()
    df = pl.read_csv(DATA_PROCESSED / "contestants.csv")
    counts = (
        df.group_by("celebrity_industry")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    plot_df = pd.DataFrame(
        {
            "celebrity_industry": counts["celebrity_industry"].to_list(),
            "count": counts["count"].to_list(),
        }
    )

    n_bars = len(plot_df)
    colors = sns.cubehelix_palette(
        n_bars, start=0.35, rot=-0.4, dark=0.25, light=0.85, reverse=True
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=plot_df,
        y="celebrity_industry",
        x="count",
        hue="celebrity_industry",
        palette=colors,
        saturation=0.9,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.set_title("Celebrity industry distribution")
    for i, v in enumerate(plot_df["count"]):
        ax.text(v + 0.8, i, str(v), va="center", fontsize=9, color=".3")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "04_industry_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Saved: 04_industry_distribution.png")


def fig_voting_method_by_season() -> None:
    """3.3: Voting method by season."""
    _setup_style()
    df = pl.read_csv(DATA_PROCESSED / "season_stats.csv")
    method_per_season = (
        df.group_by("season").first().select(["season", "voting_method"]).sort("season")
    )
    seasons = method_per_season["season"].to_list()
    methods = method_per_season["voting_method"].to_list()

    fig, ax = plt.subplots(figsize=(12, 2.8))
    for i, m in enumerate(methods):
        color = PALETTE_VOTING.get(m, ".6")
        ax.barh(
            0, 0.85, left=i, height=0.55, color=color, edgecolor=".35", linewidth=0.6
        )
    ax.set_xlim(-0.5, len(seasons) - 0.5)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("Season")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(seasons, rotation=45, ha="right")
    ax.set_title("Voting method by season")
    legend_handles = [
        Patch(
            facecolor=PALETTE_VOTING["rank_v1"], edgecolor=".35", label="Rank (S1–2)"
        ),
        Patch(
            facecolor=PALETTE_VOTING["percentage"],
            edgecolor=".35",
            label="Percentage (S3–27)",
        ),
        Patch(
            facecolor=PALETTE_VOTING["rank_v2"],
            edgecolor=".35",
            label="Rank + bottom-2 (S28+)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor=".4",
    )
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "05_voting_method_by_season.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Saved: 05_voting_method_by_season.png")


def fig_correlation_matrix() -> None:
    """3.4: Correlation matrix of week_summary variables."""
    _setup_style()
    df = pl.read_csv(DATA_PROCESSED / "week_summary.csv")
    num_cols = ["total_score", "avg_score", "judge_count", "week_rank"]
    for c in [
        "prev_week_score",
        "score_change",
        "cumulative_avg",
        "deviation_from_avg",
    ]:
        if c in df.columns:
            num_cols.append(c)
    sub = df.select([c for c in num_cols if c in df.columns]).drop_nulls()
    cols = sub.columns
    data = np.column_stack([sub[c].to_numpy() for c in cols])
    corr_arr, _ = stats.spearmanr(data)
    if np.isscalar(corr_arr):
        corr_arr = np.array([[1, corr_arr], [corr_arr, 1]])

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    mask = np.triu(np.ones_like(corr_arr, dtype=bool), k=1)
    sns.heatmap(
        corr_arr,
        mask=mask,
        cmap=CMAP_CORRELATION,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        annot=False,
        ax=ax,
        cbar_kws={"label": "Spearman correlation", "shrink": 0.7},
    )
    ax.set_xticklabels(cols, rotation=90, ha="center", fontsize=13)
    ax.set_yticklabels(cols, rotation=0, fontsize=13)
    ax.set_title(
        "Week summary variables: correlation matrix",
        fontsize=18,
        fontweight="bold",
        loc="center",
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label("Spearman correlation", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: 06_correlation_matrix.png")


def fig_season_correlation_matrix() -> None:
    """3.4: Correlation matrix of season_summary variables."""
    _setup_style()
    df = pl.read_csv(DATA_PROCESSED / "season_stats.csv")
    num_cols = [
        "weeks_competed",
        "season_total_score",
        "season_avg_weekly_score",
        "season_avg_judge_score",
        "score_volatility",
        "avg_rank",
        "weeks_ranked_first",
        "placement",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    sub = df.select(num_cols).drop_nulls()
    cols = sub.columns
    data = np.column_stack([sub[c].to_numpy() for c in cols])
    corr_arr, _ = stats.spearmanr(data)
    if np.isscalar(corr_arr):
        corr_arr = np.array([[1, corr_arr], [corr_arr, 1]])

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    mask = np.triu(np.ones_like(corr_arr, dtype=bool), k=1)
    sns.heatmap(
        corr_arr,
        mask=mask,
        cmap=CMAP_CORRELATION,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        annot=False,
        ax=ax,
        cbar_kws={"label": "Spearman correlation", "shrink": 0.7},
    )
    ax.set_xticklabels(cols, rotation=90, ha="center", fontsize=13)
    ax.set_yticklabels(cols, rotation=0, fontsize=13)
    ax.set_title(
        "Season summary variables: correlation matrix",
        fontsize=18,
        fontweight="bold",
        loc="center",
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label("Spearman correlation", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "11_season_summary_correlation_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: 11_season_summary_correlation_matrix.png")


def fig_partner_correlation_matrix() -> None:
    """3.4: Correlation matrix of partner-level variables."""
    _setup_style()
    df = pl.read_csv(DATA_PROCESSED / "partner_stats.csv")
    num_cols = [
        "total_partners",
        "avg_placement",
        "best_placement",
        "championships",
        "top3_finishes",
        "avg_judge_score",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    sub = df.select(num_cols).drop_nulls()
    cols = sub.columns
    data = np.column_stack([sub[c].to_numpy() for c in cols])
    corr_arr, _ = stats.spearmanr(data)
    if np.isscalar(corr_arr):
        corr_arr = np.array([[1, corr_arr], [corr_arr, 1]])

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    mask = np.triu(np.ones_like(corr_arr, dtype=bool), k=1)
    sns.heatmap(
        corr_arr,
        mask=mask,
        cmap=CMAP_CORRELATION,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        annot=False,
        ax=ax,
        cbar_kws={"label": "Spearman correlation", "shrink": 0.7},
    )
    ax.set_xticklabels(cols, rotation=90, ha="center", fontsize=13)
    ax.set_yticklabels(cols, rotation=0, fontsize=13)
    ax.set_title(
        "Partner summary variables: correlation matrix",
        fontsize=18,
        fontweight="bold",
        loc="center",
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label("Spearman correlation", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "12_partner_summary_correlation_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: 12_partner_summary_correlation_matrix.png")


def fig_correlation_matrices_three_levels() -> None:
    """3.4: Three correlation matrices side by side (Weekly | Season | Partner)."""
    _setup_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    def _draw_one(ax: plt.Axes, corr_arr: np.ndarray, cols: list[str], title: str) -> None:
        mask = np.triu(np.ones_like(corr_arr, dtype=bool), k=1)
        sns.heatmap(
            corr_arr,
            mask=mask,
            cmap=CMAP_CORRELATION,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            annot=False,
            ax=ax,
            cbar=False,
        )
        ax.set_xticklabels(cols, rotation=90, ha="center", fontsize=10)
        ax.set_yticklabels(cols, rotation=0, fontsize=10)
        ax.set_title(
            title,
            fontsize=18,
            fontweight="bold",
            loc="center",
        )

    # Weekly
    df_w = pl.read_csv(DATA_PROCESSED / "week_summary.csv")
    num_w = ["total_score", "avg_score", "judge_count", "week_rank"]
    for c in ["prev_week_score", "score_change", "cumulative_avg", "deviation_from_avg"]:
        if c in df_w.columns:
            num_w.append(c)
    num_w = [c for c in num_w if c in df_w.columns]
    sub_w = df_w.select(num_w).drop_nulls()
    cols_w = sub_w.columns
    data_w = np.column_stack([sub_w[c].to_numpy() for c in cols_w])
    corr_w, _ = stats.spearmanr(data_w)
    if np.isscalar(corr_w):
        corr_w = np.array([[1, corr_w], [corr_w, 1]])
    _draw_one(ax1, corr_w, cols_w, "Weekly")

    # Season
    df_s = pl.read_csv(DATA_PROCESSED / "season_stats.csv")
    num_s = [
        "weeks_competed",
        "season_total_score",
        "season_avg_weekly_score",
        "season_avg_judge_score",
        "score_volatility",
        "avg_rank",
        "weeks_ranked_first",
        "placement",
    ]
    num_s = [c for c in num_s if c in df_s.columns]
    sub_s = df_s.select(num_s).drop_nulls()
    cols_s = sub_s.columns
    data_s = np.column_stack([sub_s[c].to_numpy() for c in cols_s])
    corr_s, _ = stats.spearmanr(data_s)
    if np.isscalar(corr_s):
        corr_s = np.array([[1, corr_s], [corr_s, 1]])
    _draw_one(ax2, corr_s, cols_s, "Season")

    # Partner
    df_p = pl.read_csv(DATA_PROCESSED / "partner_stats.csv")
    num_p = [
        "total_partners",
        "avg_placement",
        "best_placement",
        "championships",
        "top3_finishes",
        "avg_judge_score",
    ]
    num_p = [c for c in num_p if c in df_p.columns]
    sub_p = df_p.select(num_p).drop_nulls()
    cols_p = sub_p.columns
    data_p = np.column_stack([sub_p[c].to_numpy() for c in cols_p])
    corr_p, _ = stats.spearmanr(data_p)
    if np.isscalar(corr_p):
        corr_p = np.array([[1, corr_p], [corr_p, 1]])
    _draw_one(ax3, corr_p, cols_p, "Partner")

    # Shared colorbar on the far right (avoid overlapping the Partner panel)
    fig.subplots_adjust(right=0.88, wspace=0.4)
    im = ax1.collections[0]
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Spearman correlation", fontsize=12)
    fig.savefig(
        FIGURES_DIR / "13_correlation_matrices_three_levels.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: 13_correlation_matrices_three_levels.png")


def fig_variable_importance() -> None:
    """3.4: Variable correlation with elimination outcome."""
    _setup_style()
    week_df = pl.read_csv(DATA_PROCESSED / "week_summary.csv")
    cont_df = pl.read_csv(DATA_PROCESSED / "contestants.csv")
    cont_df = cont_df.with_columns(
        pl.col("results")
        .str.extract(r"Eliminated Week (\d+)", 1)
        .cast(pl.Int32)
        .alias("elim_week")
    )
    merged = week_df.join(
        cont_df.select(["contestant_id", "elim_week"]),
        on="contestant_id",
        how="left",
    )
    merged = merged.with_columns(
        pl.when(pl.col("elim_week").is_null())
        .then(pl.lit(False))
        .otherwise(pl.col("week") == pl.col("elim_week"))
        .alias("eliminated")
    )
    num_cols = [
        "total_score",
        "avg_score",
        "week_rank",
        "score_change",
        "deviation_from_avg",
    ]
    num_cols = [c for c in num_cols if c in merged.columns]
    sub = merged.select(num_cols + ["eliminated"]).drop_nulls()

    correlations: list[tuple[str, float]] = []
    elim_arr = sub["eliminated"].cast(pl.Int32).to_numpy()
    for c in num_cols:
        x = sub[c].to_numpy()
        r, _ = stats.spearmanr(x, elim_arr)
        correlations.append((c, float(r) if not np.isnan(r) else 0.0))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    labels = [x[0] for x in correlations]
    vals = [x[1] for x in correlations]
    colors = [
        PALETTE_POS_NEG["positive"] if v > 0 else PALETTE_POS_NEG["negative"]
        for v in vals
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, vals, color=colors, edgecolor=".35", linewidth=0.6)
    ax.axvline(x=0, color=".3", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Spearman correlation with elimination")
    ax.set_title("Variable importance: correlation with elimination outcome")
    ax.invert_yaxis()
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "07_variable_importance.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Saved: 07_variable_importance.png")


def main() -> None:
    """Generate all preprocessing figures."""
    print(f"Output directory: {FIGURES_DIR}")
    fig_score_boxplot()
    fig_missing_heatmap()
    fig_wide_to_long_schematic()
    fig_industry_distribution()
    fig_voting_method_by_season()
    fig_correlation_matrix()
    fig_season_correlation_matrix()
    fig_partner_correlation_matrix()
    fig_correlation_matrices_three_levels()
    fig_variable_importance()
    print("Done.")


if __name__ == "__main__":
    main()
