"""
Task 1 MCMC figures: strategy schematic and diagnostics.

1. MCMC strategy schematic: flowchart (propose → renormalize → accept/reject → adaptive).
2. MCMC diagnostics: acceptance rate and ESS (paper style, same as plot_metrics).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

BAR_ALPHA = 0.85
COLOR_BOX = "#E3F2FD"
COLOR_ARROW = "#37474F"
COLOR_SUCCESS = "#81C784"
COLOR_NEUTRAL = "#90A4AE"


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


def plot_mcmc_strategy_schematic() -> None:
    """Flowchart: current state → propose → renormalize → accept/reject → adaptive."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    box_w, box_h = 5.5, 1.0
    x_center = 5
    ys = [12, 10, 8, 5.5, 3.5, 1.5]

    boxes_text = [
        (r"State $\mathbf{f}^{(k)}$ (current fan shares)"),
        (r"Propose: $\log \mathbf{f}^* = \log \mathbf{f}^{(k)} + \varepsilon$," + "\n" + r"$\varepsilon \sim \mathcal{N}(0, \sigma_{\mathrm{prop}}^2)$"),
        (r"Renormalize: $\sum_i f_i^* = 1$ (simplex)"),
        (r"Acceptance: $\alpha = \min(1, P(\mathbf{f}^*) / P(\mathbf{f}^{(k)}))$"),
        (r"Accept with prob. $\alpha$: $\mathbf{f}^{(k+1)} = \mathbf{f}^*$," + "\n" + r"else $\mathbf{f}^{(k+1)} = \mathbf{f}^{(k)}$"),
        (r"Adaptive: update $\sigma_{\mathrm{prop}}$ to target ~0.35"),
    ]

    for i, (y, text) in enumerate(zip(ys, boxes_text)):
        box = FancyBboxPatch(
            (x_center - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02",
            facecolor=COLOR_BOX,
            edgecolor=COLOR_ARROW,
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(
            x_center,
            y,
            text,
            ha="center",
            va="center",
            fontsize=9,
            wrap=True,
        )
        if i < len(ys) - 1:
            ax.annotate(
                "",
                xy=(x_center, ys[i + 1] + box_h / 2 + 0.15),
                xytext=(x_center, y - box_h / 2 - 0.15),
                arrowprops=dict(arrowstyle="->", color=COLOR_ARROW, lw=1.5),
            )

    ax.set_title("MCMC sampling strategy", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mcmc_strategy_schematic.png")
    plt.close(fig)
    print("Saved: mcmc_strategy_schematic.png")


def plot_mcmc_diagnostics() -> None:
    """Diagnostics: acceptance rate and ESS (paper style, alpha, white edge)."""
    _setup_style()
    with (OUTPUTS_DIR / "results.json").open(encoding="utf-8") as f:
        results = json.load(f)
    diag = results.get("diagnostics", {})
    if not diag:
        print("No diagnostics in results.json, skip mcmc_diagnostics.png")
        return

    mean_acc = diag["mean_acceptance_rate"]
    mean_ess = diag["mean_ess"]
    min_ess = diag["min_ess"]
    target_acc = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    ax1, ax2 = axes

    x1 = ["Mean\nacceptance", "Target\n(0.35)"]
    vals1 = [mean_acc, target_acc]
    colors1 = [COLOR_SUCCESS if abs(mean_acc - target_acc) < 0.1 else COLOR_NEUTRAL, "#E0E0E0"]
    bars1 = ax1.bar(
        x1,
        vals1,
        color=colors1,
        edgecolor="white",
        linewidth=0.5,
        alpha=BAR_ALPHA,
    )
    ax1.axhline(target_acc, color="#757575", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Rate", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 0.55)
    ax1.set_title("Acceptance rate", fontsize=13, fontweight="bold")
    for b, v in zip(bars1, vals1):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    sns.despine(ax=ax1)

    x2 = ["Mean ESS", "Min ESS"]
    vals2 = [mean_ess, min_ess]
    colors2 = [COLOR_SUCCESS, COLOR_SUCCESS if min_ess >= 100 else COLOR_NEUTRAL]
    bars2 = ax2.bar(
        x2,
        vals2,
        color=colors2,
        edgecolor="white",
        linewidth=0.5,
        alpha=BAR_ALPHA,
    )
    ax2.axhline(100, color="#757575", linestyle="--", linewidth=0.8, label="Threshold 100")
    ax2.set_ylabel("ESS", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(vals2) * 1.15)
    ax2.set_title("Effective sample size", fontsize=13, fontweight="bold")
    for b, v in zip(bars2, vals2):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, f"{v:.0f}", ha="center", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    sns.despine(ax=ax2)

    fig.suptitle("MCMC diagnostics", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mcmc_diagnostics.png")
    plt.close(fig)
    print("Saved: mcmc_diagnostics.png")


def main() -> None:
    plot_mcmc_strategy_schematic()
    plot_mcmc_diagnostics()
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
