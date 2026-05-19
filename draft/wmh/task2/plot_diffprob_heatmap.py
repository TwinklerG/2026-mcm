"""
Task 2: Heatmap of Rank vs Percentage disagreement (DiffProb) by season and week.

Circle-matrix style: each cell is a circle; color = agree (green) / no elimination (grey) /
disagree (blue). White background, light grid, legend on the right.
Output: figures/diffprob_heatmap.png, figures/diffprob_heatmap.pdf, figures/diffprob_heatmap.svg
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Circle

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

DIFFPROB_CSV = OUTPUTS_DIR / "diffprob_by_week.csv"

# Circle-matrix colors (green = agree, grey = no elimination, blue = disagree)
COLOR_AGREE = "#2e7d32"
COLOR_NO_ELIM = "#bdbdbd"
COLOR_DISAGREE = "#1565c0"


def load_diffprob_matrix() -> tuple[np.ndarray, list[int], list[int]]:
    """
    Load diffprob_by_week.csv and build season x week matrix.

    Returns
    -------
    mat : np.ndarray
        Shape (n_seasons, n_weeks). Values: 0 = agree, 1 = disagree, np.nan = no elimination.
    seasons : list[int]
        Season indices (1..34).
    weeks : list[int]
        Week indices (1..11).
    """
    df = pl.read_csv(DIFFPROB_CSV)
    seasons = sorted(df["season"].unique().to_list())
    weeks = sorted(df["week"].unique().to_list())
    n_seasons = len(seasons)
    n_weeks = len(weeks)
    season_to_row = {s: i for i, s in enumerate(seasons)}
    week_to_col = {w: i for i, w in enumerate(weeks)}
    mat = np.full((n_seasons, n_weeks), np.nan)
    for row in df.iter_rows(named=True):
        r = season_to_row[row["season"]]
        c = week_to_col[row["week"]]
        mat[r, c] = float(row["diffprob_t"])
    return mat, seasons, weeks


def _value_to_color(val: float) -> str:
    if np.isnan(val) or val == 0.5:
        return COLOR_NO_ELIM
    if val <= 0:
        return COLOR_AGREE
    return COLOR_DISAGREE


def plot_heatmap(
    mat: np.ndarray,
    seasons: list[int],
    weeks: list[int],
    out_path: Path,
) -> None:
    """Draw circle-matrix heatmap (horizontal: Season on x, Week on y)."""
    n_seasons, n_weeks = mat.shape
    mat_t = mat.T
    n_rows, n_cols = n_weeks, n_seasons
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    display = np.where(np.isnan(mat_t), 0.5, mat_t)
    radius = 0.38
    for r in range(n_rows):
        for c in range(n_cols):
            color = _value_to_color(display[r, c])
            circle = Circle(
                (c, r), radius, facecolor=color, edgecolor="white", linewidth=0.8
            )
            ax.add_patch(circle)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(seasons, fontsize=11, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(weeks, fontsize=11)
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Week", fontsize=12)
    ax.set_title("Rank vs Percentage", fontsize=16, fontweight="bold")
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(
        True, which="minor", axis="both", color="#e0e0e0", linestyle="-", linewidth=0.6
    )
    ax.tick_params(which="minor", size=0)
    from matplotlib.legend_handler import HandlerPatch

    def _legend_circle(
        legend, orig_handle, xdescent, ydescent, width, height, fontsize
    ):
        return Circle(
            (width / 2 - xdescent, height / 2 - ydescent), min(width, height) / 3
        )

    legend_handles = [
        Circle((0, 0), 0.5, facecolor=COLOR_AGREE, edgecolor="white", label="Agree"),
        Circle(
            (0, 0),
            0.5,
            facecolor=COLOR_NO_ELIM,
            edgecolor="white",
            label="No elimination",
        ),
        Circle(
            (0, 0), 0.5, facecolor=COLOR_DISAGREE, edgecolor="white", label="Disagree"
        ),
    ]
    ax.legend(
        legend_handles,
        ["Agree", "No elimination", "Disagree"],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        handler_map={Circle: HandlerPatch(patch_func=_legend_circle)},
        frameon=True,
        edgecolor="#e0e0e0",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def main() -> None:
    mat, seasons, weeks = load_diffprob_matrix()
    for ext in ("png", "pdf", "svg"):
        plot_heatmap(mat, seasons, weeks, FIG_DIR / f"diffprob_heatmap.{ext}")
    print("Output:", FIG_DIR / "diffprob_heatmap.png (pdf, svg)")


if __name__ == "__main__":
    main()
