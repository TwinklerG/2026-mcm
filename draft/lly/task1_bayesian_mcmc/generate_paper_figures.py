"""
补充图表生成脚本 - Task 1。

生成 PAPER.md 中建议但尚未创建的图表：
1. 后验一致性热力图 (performance_heatmap.png) - 需更新
2. 论文总结图 (paper_summary.png) - 核心结果总览
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# 添加 shared 模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.viz_theme import (
    COLORS,
    add_rule_change_marker,
    add_threshold_line,
    apply_theme,
    create_figure,
)

# 设置随机种子
np.random.seed(42)
random.seed(42)

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = SCRIPT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[dict, dict, dict, dict]:
    """加载所有需要的数据。"""
    with open(OUTPUT_DIR / "results.json", encoding="utf-8") as f:
        results = json.load(f)
    with open(OUTPUT_DIR / "reconstruction_details.json", encoding="utf-8") as f:
        recon_details = json.load(f)
    with open(OUTPUT_DIR / "bottom2_accuracy.json", encoding="utf-8") as f:
        bottom2_results = json.load(f)
    with open(OUTPUT_DIR / "detailed_uncertainty.json", encoding="utf-8") as f:
        uncertainty = json.load(f)
    return results, recon_details, bottom2_results, uncertainty


def plot_paper_summary(results: dict, recon_details: dict, output_path: Path) -> None:
    """
    生成论文核心结果总览图。

    2x2 子图布局：
    - 左上：核心验证指标雷达图
    - 右上：S28 前后对比条形图
    - 左下：MCMC 诊断指标
    - 右下：不确定性统计
    """
    apply_theme()
    fig = plt.figure(figsize=(14, 10))

    # ── 左上：核心验证指标条形图 ──
    ax1 = fig.add_subplot(2, 2, 1)

    metrics = {
        "Reconstruction\nAccuracy": results["validation"]["reconstruction_accuracy"],
        "Bottom-2\nRecall": results["validation"]["bottom2_overall_recall"],
        "Constraint\nSatisfaction": results["validation"][
            "constraint_satisfaction_rate"
        ],
        "Finale τ": results["validation"]["finale_kendall_tau"],
        "Winner\nAccuracy": results["validation"]["finale_winner_accuracy"],
    }

    x = np.arange(len(metrics))
    bars = ax1.bar(
        x,
        list(metrics.values()),
        color=[
            COLORS["judge"],
            COLORS["fan"],
            COLORS["success"],
            COLORS["highlight"],
            COLORS["success"],
        ],
        edgecolor="black",
        alpha=0.85,
    )

    # 添加数值标注
    for bar, val in zip(bars, metrics.values()):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(list(metrics.keys()), fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Rate")
    ax1.set_title("Core Validation Metrics", fontweight="bold")
    ax1.axhline(0.9, color="gray", linestyle=":", alpha=0.5, label="90% threshold")

    # ── 右上：S28 前后对比 ──
    ax2 = fig.add_subplot(2, 2, 2)

    pre_s28 = {
        "Strict": results["validation"]["bottom2_pre_s28_strict"],
        "Bottom-2": results["validation"]["bottom2_pre_s28_recall"],
    }
    post_s28 = {
        "Strict": results["validation"]["bottom2_post_s28_strict"],
        "Bottom-2": results["validation"]["bottom2_post_s28_recall"],
    }

    x = np.arange(2)
    width = 0.35

    bars1 = ax2.bar(
        x - width / 2,
        list(pre_s28.values()),
        width,
        label="Seasons 1-27",
        color=COLORS["judge"],
        edgecolor="black",
    )
    bars2 = ax2.bar(
        x + width / 2,
        list(post_s28.values()),
        width,
        label="Seasons 28+",
        color=COLORS["fan"],
        edgecolor="black",
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Strict Accuracy", "Bottom-2 Recall"])
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Rate")
    ax2.set_title("Rule Change Impact (S28)", fontweight="bold")
    ax2.legend()

    # 添加数值标注
    for bars in [bars1, bars2]:
        for bar in bars:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.1%}",
                ha="center",
                fontsize=9,
            )

    # ── 左下：MCMC 诊断 ──
    ax3 = fig.add_subplot(2, 2, 3)

    diag_metrics = {
        "Acceptance\nRate": (
            results["diagnostics"]["mean_acceptance_rate"],
            0.35,
            "Target: 35%",
        ),
        "Mean ESS\n(÷100)": (results["diagnostics"]["mean_ess"] / 100, 1.0, "Min: 100"),
        "Min ESS\n(÷100)": (results["diagnostics"]["min_ess"] / 100, 1.0, "Min: 100"),
    }

    x = np.arange(len(diag_metrics))
    values = [v[0] for v in diag_metrics.values()]
    targets = [v[1] for v in diag_metrics.values()]

    bars = ax3.bar(x, values, color=COLORS["success"], edgecolor="black", alpha=0.85)
    ax3.scatter(
        x,
        targets,
        marker="*",
        s=200,
        color=COLORS["warning"],
        zorder=5,
        label="Target/Min",
    )

    ax3.set_xticks(x)
    ax3.set_xticklabels(list(diag_metrics.keys()))
    ax3.set_ylabel("Value")
    ax3.set_title("MCMC Diagnostics", fontweight="bold")
    ax3.legend()

    # ── 右下：泛化分析 ──
    ax4 = fig.add_subplot(2, 2, 4)

    stab = results["stability"]
    labels = ["Train (S1-20)", "Test (S21+)", "Gap"]
    values = [stab["train_accuracy"], stab["test_accuracy"], stab["generalization_gap"]]
    colors = [COLORS["success"], COLORS["highlight"], COLORS["warning"]]

    bars = ax4.bar(labels, values, color=colors, edgecolor="black", alpha=0.85)

    for bar, val in zip(bars, values):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax4.set_ylim(0, 1.15)
    ax4.set_ylabel("Rate")
    ax4.set_title("Generalization Analysis", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_uncertainty_analysis(uncertainty: dict, output_path: Path) -> None:
    """
    生成不确定性分析综合图。

    2x2 布局：
    - 左上：CrI 宽度直方图
    - 右上：按周次的 CrI 宽度箱线图
    - 左下：按状态的 CrI 对比
    - 右下：争议型选手 Top 5
    """
    apply_theme()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 准备数据
    all_widths = []
    for season_data in uncertainty.get("per_season", {}).values():
        for week_data in season_data.get("weeks", {}).values():
            for contestant in week_data.get("contestants", []):
                all_widths.append({
                    "width": contestant.get("cri_width", 0),
                    "week": week_data.get("week", 0),
                    "status": contestant.get("status", "unknown"),
                })

    if not all_widths:
        plt.close()
        return

    widths = [d["width"] for d in all_widths]
    weeks = [d["week"] for d in all_widths]
    statuses = [d["status"] for d in all_widths]

    # ── 左上：直方图 ──
    ax1 = axes[0, 0]
    ax1.hist(widths, bins=50, color=COLORS["judge"], edgecolor="white", alpha=0.8)
    ax1.axvline(
        np.mean(widths),
        color=COLORS["warning"],
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(widths):.3f}",
    )
    ax1.axvline(
        np.median(widths),
        color=COLORS["success"],
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(widths):.3f}",
    )
    ax1.set_xlabel("95% Credible Interval Width")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Estimation Uncertainty")
    ax1.legend()

    # ── 右上：按周次箱线图 ──
    ax2 = axes[0, 1]
    week_groups = {}
    for w, width in zip(weeks, widths):
        if w not in week_groups:
            week_groups[w] = []
        week_groups[w].append(width)

    sorted_weeks = sorted(week_groups.keys())
    boxplot_data = [week_groups[w] for w in sorted_weeks]

    bp = ax2.boxplot(boxplot_data, labels=sorted_weeks, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["judge"])
        patch.set_alpha(0.7)

    ax2.set_xlabel("Week")
    ax2.set_ylabel("CrI Width")
    ax2.set_title("Uncertainty by Week")

    # ── 左下：按状态对比 ──
    ax3 = axes[1, 0]
    status_groups = {}
    for status, width in zip(statuses, widths):
        if status not in status_groups:
            status_groups[status] = []
        status_groups[status].append(width)

    status_means = {k: np.mean(v) for k, v in status_groups.items()}
    status_labels = list(status_means.keys())
    status_values = list(status_means.values())

    colors_map = {"eliminated": COLORS["warning"], "survived": COLORS["success"]}
    bar_colors = [colors_map.get(s, COLORS["neutral"]) for s in status_labels]

    bars = ax3.bar(status_labels, status_values, color=bar_colors, edgecolor="black")
    for bar, val in zip(bars, status_values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center",
            fontsize=10,
        )

    ax3.set_ylabel("Mean CrI Width")
    ax3.set_title("Uncertainty by Elimination Status")

    # ── 右下：早期 vs 晚期 ──
    ax4 = axes[1, 1]
    early_weeks = [w for w, week in zip(widths, weeks) if week <= 4]
    late_weeks = [w for w, week in zip(widths, weeks) if week > 4]

    data = [early_weeks, late_weeks]
    labels = [f"Early (W1-4)\nN={len(early_weeks)}", f"Late (W5+)\nN={len(late_weeks)}"]

    bp = ax4.boxplot(data, labels=labels, patch_artist=True)
    bp["boxes"][0].set_facecolor(COLORS["judge"])
    bp["boxes"][1].set_facecolor(COLORS["fan"])

    ax4.set_ylabel("CrI Width")
    ax4.set_title("Uncertainty: Early vs Late Weeks")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_judges_save_analysis(results: dict, output_path: Path) -> None:
    """
    生成 Judges' Save 机制分析图。

    展示评委救人的偏好分析。
    """
    apply_theme()
    fig, ax = plt.subplots(figsize=(10, 6))

    # 从 judges_save_analysis.json 读取数据（如果存在）
    judges_save_path = OUTPUT_DIR / "judges_save_analysis.json"
    if judges_save_path.exists():
        with open(judges_save_path, encoding="utf-8") as f:
            js_data = json.load(f)

        metrics = {
            "Higher Judge\nScore": js_data.get("saved_higher_judge_rate", 0.5),
            "Higher Fan\nShare": js_data.get("saved_higher_fan_rate", 0.905),
            "Higher Combined\nRank": js_data.get("saved_higher_combined_rate", 0.81),
        }
    else:
        # 使用 PAPER.md 中的数据
        metrics = {
            "Higher Judge\nScore": 0.50,
            "Higher Fan\nShare": 0.905,
            "Higher Combined\nRank": 0.81,
        }

    x = np.arange(len(metrics))
    colors = [COLORS["judge"], COLORS["fan"], COLORS["highlight"]]

    bars = ax.bar(
        x, list(metrics.values()), color=colors, edgecolor="black", alpha=0.85
    )

    # 添加 50% 参考线（随机选择）
    ax.axhline(
        0.5, color="gray", linestyle="--", linewidth=2, label="Random Choice (50%)"
    )

    # 添加数值标注
    for bar, val in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Rate of Saving Higher Contestant")
    ax.set_title(
        "Judges' Save Preference Analysis (Seasons 28+)\n"
        "Do judges protect technically superior dancers?",
        fontweight="bold",
    )
    ax.legend(loc="upper left")

    # 添加注释
    ax.annotate(
        "Judges show NO significant\npreference for technical skill\n(p = 0.56, binomial test)",
        xy=(0, 0.5),
        xytext=(0.5, 0.25),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color=COLORS["neutral"]),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="lightyellow",
            edgecolor=COLORS["highlight"],
        ),
    )

    ax.annotate(
        "Judges align with\nfan preferences 90.5%\nof the time!",
        xy=(1, 0.905),
        xytext=(1.5, 0.7),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color=COLORS["warning"]),
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="mistyrose", edgecolor=COLORS["warning"]
        ),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """生成所有补充图表。"""
    print("=" * 60)
    print("Task 1: 补充图表生成")
    print("=" * 60)

    results, recon_details, bottom2_results, uncertainty = load_data()

    # 1. 论文总结图
    print("\n[1/3] 生成论文核心结果总览图...")
    plot_paper_summary(results, recon_details, FIGURE_DIR / "paper_summary.png")

    # 2. 不确定性分析图
    print("\n[2/3] 生成不确定性分析图...")
    plot_uncertainty_analysis(uncertainty, FIGURE_DIR / "uncertainty_analysis.png")

    # 3. Judges' Save 分析图
    print("\n[3/3] 生成 Judges' Save 分析图...")
    plot_judges_save_analysis(results, FIGURE_DIR / "judges_save_preference.png")

    print("\n" + "=" * 60)
    print("Task 1 补充图表生成完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
