"""
跨问题综合分析图表生成脚本。

生成可能用于论文多个部分的综合图表：
1. 三个问题的核心发现总览
2. 数据集概览
3. 方法论流程图
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 添加 shared 模块路径
sys.path.insert(0, str(Path(__file__).parent))

from shared.viz_theme import (
    COLORS,
    SYSTEM_COLORS,
    apply_theme,
)

# 设置随机种子
np.random.seed(42)
random.seed(42)

SCRIPT_DIR = Path(__file__).parent
FIGURE_DIR = SCRIPT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def load_all_results() -> dict:
    """加载三个任务的所有结果。"""
    data = {}

    # Task 1
    with open(
        SCRIPT_DIR / "task1_bayesian_mcmc/outputs/results.json", encoding="utf-8"
    ) as f:
        data["task1"] = json.load(f)

    # Task 3
    with open(
        SCRIPT_DIR / "task3_mixed_effects/outputs/conclusions.json", encoding="utf-8"
    ) as f:
        data["task3"] = json.load(f)

    # Task 4
    with open(
        SCRIPT_DIR / "task4_fair_system/outputs/system_comparison.json",
        encoding="utf-8",
    ) as f:
        data["task4"] = json.load(f)

    return data


def plot_three_problems_overview(data: dict, output_path: Path) -> None:
    """
    生成三个问题核心发现的综合总览图。

    横向 1x3 布局，每个问题一个面板。
    """
    apply_theme()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # ══════════════════════════════════════════════════════════════════════════
    # Task 1: 贝叶斯粉丝投票估计
    # ══════════════════════════════════════════════════════════════════════════
    ax1 = axes[0]
    ax1.set_title(
        "Task 1: Fan Vote Estimation\n(Bayesian MCMC)", fontsize=12, fontweight="bold"
    )

    t1 = data["task1"]
    metrics = {
        "Reconstruction\nAccuracy": t1["validation"]["reconstruction_accuracy"],
        "Bottom-2\nRecall": t1["validation"]["bottom2_overall_recall"],
        "Finale\nKendall's τ": t1["validation"]["finale_kendall_tau"],
        "Winner\nAccuracy": t1["validation"]["finale_winner_accuracy"],
    }

    x = np.arange(len(metrics))
    colors = [COLORS["judge"], COLORS["fan"], COLORS["highlight"], COLORS["success"]]
    bars = ax1.bar(
        x, list(metrics.values()), color=colors, edgecolor="black", alpha=0.85
    )

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
    ax1.set_xticklabels(list(metrics.keys()), fontsize=8)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Rate")
    ax1.axhline(0.9, color="gray", linestyle=":", alpha=0.5)

    # 核心发现文字
    ax1.text(
        0.5,
        0.03,
        "• Judges' Save: 90.5% align with fans\n• S28+ structural drift detected",
        transform=ax1.transAxes,
        fontsize=8,
        va="bottom",
        ha="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Task 3: 混合效应分析
    # ══════════════════════════════════════════════════════════════════════════
    ax2 = axes[1]
    ax2.set_title(
        "Task 3: Impact Analysis\n(Mixed-Effects Model)", fontsize=12, fontweight="bold"
    )

    t3 = data["task3"]

    # 运动员悖论
    athlete = t3["athlete_paradox"]
    labels = ["Judge β\n(Athlete)", "Fan β\n(Athlete)"]
    values = [athlete["judge_coefficient"], athlete["fan_coefficient"]]
    colors = [COLORS["judge"], COLORS["fan"]]

    bars = ax2.bar(labels, values, color=colors, edgecolor="black", alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)

    for bar, val in zip(bars, values):
        sign = "+" if val > 0 else ""
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.01 if val > 0 else -0.02),
            f"{sign}{val:.3f}",
            ha="center",
            va="bottom" if val > 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    ax2.set_ylabel("Coefficient")
    ax2.set_ylim(-0.2, 0.12)

    # ICC 对比（小图）
    ax2_inset = ax2.inset_axes([0.6, 0.6, 0.35, 0.35])
    icc_labels = ["Judge", "Fan"]
    icc_values = [
        t3["variance_explained"]["judge_icc"],
        t3["variance_explained"]["fan_icc"],
    ]
    ax2_inset.bar(
        icc_labels,
        icc_values,
        color=[COLORS["judge"], COLORS["fan"]],
        edgecolor="black",
        alpha=0.7,
    )
    ax2_inset.set_ylabel("ICC", fontsize=8)
    ax2_inset.set_title("Pro Dancer\nVariance", fontsize=8)
    ax2_inset.tick_params(labelsize=7)

    # 核心发现文字
    ax2.text(
        0.5,
        0.03,
        "• Athlete Paradox: Opposite directions!\n• ICC: 2.3× higher for fan voting",
        transform=ax2.transAxes,
        fontsize=8,
        va="bottom",
        ha="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Task 4: PTFS 系统
    # ══════════════════════════════════════════════════════════════════════════
    ax3 = axes[2]
    ax3.set_title(
        "Task 4: Fair System Design\n(Progressive Technical Fairness)",
        fontsize=12,
        fontweight="bold",
    )

    t4 = data["task4"]
    systems = {s["name"]: s for s in t4["systems"]}

    labels = ["Rank", "Percentage", "PTFS"]
    tau_values = [
        systems["Rank-based"]["kendall_tau"],
        systems["Percentage-based"]["kendall_tau"],
        systems["PTFS"]["kendall_tau"],
    ]

    colors = [
        SYSTEM_COLORS["Rank-based"],
        SYSTEM_COLORS["Percentage-based"],
        SYSTEM_COLORS["PTFS"],
    ]
    bars = ax3.bar(labels, tau_values, color=colors, edgecolor="black", alpha=0.85)

    for bar, val in zip(bars, tau_values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax3.set_ylabel("Kendall's τ")
    ax3.set_ylim(0.6, 0.8)

    # 统计显著性标注
    ax3.annotate(
        "p = 0.031 *",
        xy=(2, systems["PTFS"]["kendall_tau"]),
        xytext=(1.5, 0.77),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=SYSTEM_COLORS["PTFS"]),
    )

    # 核心发现文字
    ax3.text(
        0.5,
        0.03,
        "• PTFS: 45% → 80% judge weight\n• +0.022 τ improvement (significant)",
        transform=ax3.transAxes,
        fontsize=8,
        va="bottom",
        ha="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_methodology_flow(output_path: Path) -> None:
    """
    生成方法论流程图。

    展示三个问题之间的数据流和依赖关系。
    """
    apply_theme()
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # 标题
    ax.text(
        7,
        7.5,
        "Methodology Overview: Data Flow Across Tasks",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    # ═══════════════════════════════════════════════════════════════════
    # 数据源 (左侧)
    # ═══════════════════════════════════════════════════════════════════
    ax.add_patch(
        plt.Rectangle((0.5, 3), 2.5, 2, facecolor="#E8E8E8", edgecolor="black", lw=2)
    )
    ax.text(
        1.75,
        4,
        "Raw Data\n\n• Judge Scores\n• Elimination Results\n• Celebrity Info",
        ha="center",
        va="center",
        fontsize=9,
    )

    # ═══════════════════════════════════════════════════════════════════
    # Task 1 (中左)
    # ═══════════════════════════════════════════════════════════════════
    ax.add_patch(
        plt.Rectangle(
            (4, 4.5),
            2.5,
            2.5,
            facecolor=COLORS["judge"],
            edgecolor="black",
            lw=2,
            alpha=0.3,
        )
    )
    ax.text(
        5.25,
        6.5,
        "Task 1",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["judge"],
    )
    ax.text(
        5.25,
        5.5,
        "Bayesian MCMC\nFan Vote Estimation\n\nOutput: f̂ᵢₜ",
        ha="center",
        va="center",
        fontsize=9,
    )

    # 箭头：Data → Task 1
    ax.annotate(
        "",
        xy=(4, 5.75),
        xytext=(3, 4.5),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )

    # ═══════════════════════════════════════════════════════════════════
    # Task 3 (中右)
    # ═══════════════════════════════════════════════════════════════════
    ax.add_patch(
        plt.Rectangle(
            (7.5, 4.5),
            2.5,
            2.5,
            facecolor=COLORS["fan"],
            edgecolor="black",
            lw=2,
            alpha=0.3,
        )
    )
    ax.text(
        8.75,
        6.5,
        "Task 3",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["fan"],
    )
    ax.text(
        8.75,
        5.5,
        "Mixed-Effects\nImpact Analysis\n\nOutput: β, ICC",
        ha="center",
        va="center",
        fontsize=9,
    )

    # 箭头：Task 1 → Task 3
    ax.annotate(
        "f̂ᵢₜ",
        xy=(7.5, 5.75),
        xytext=(6.5, 5.75),
        arrowprops=dict(arrowstyle="->", color=COLORS["judge"], lw=2),
        fontsize=10,
        fontweight="bold",
        color=COLORS["judge"],
    )

    # ═══════════════════════════════════════════════════════════════════
    # Task 4 (右侧)
    # ═══════════════════════════════════════════════════════════════════
    ax.add_patch(
        plt.Rectangle(
            (11, 4.5),
            2.5,
            2.5,
            facecolor=SYSTEM_COLORS["PTFS"],
            edgecolor="black",
            lw=2,
            alpha=0.3,
        )
    )
    ax.text(
        12.25,
        6.5,
        "Task 4",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=SYSTEM_COLORS["PTFS"],
    )
    ax.text(
        12.25,
        5.5,
        "PTFS Design\nFair System\n\nOutput: wⱼ(t)",
        ha="center",
        va="center",
        fontsize=9,
    )

    # 箭头：Task 1 → Task 4
    ax.annotate(
        "f̂ᵢₜ",
        xy=(11, 5.25),
        xytext=(6.5, 5.25),
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["judge"],
            lw=2,
            connectionstyle="arc3,rad=-0.2",
        ),
        fontsize=10,
        fontweight="bold",
        color=COLORS["judge"],
    )

    # 箭头：Task 3 → Task 4 (诊断信息)
    ax.annotate(
        "Diagnostic\nInsights",
        xy=(11, 5.75),
        xytext=(10, 5.75),
        arrowprops=dict(arrowstyle="->", color=COLORS["fan"], lw=1.5),
        fontsize=8,
        color=COLORS["fan"],
    )

    # ═══════════════════════════════════════════════════════════════════
    # 输出 (底部)
    # ═══════════════════════════════════════════════════════════════════
    ax.add_patch(
        plt.Rectangle((4, 1), 6, 2, facecolor="#F5F5F5", edgecolor="black", lw=2)
    )
    ax.text(7, 2.5, "Paper Deliverables", ha="center", fontsize=11, fontweight="bold")
    ax.text(
        7,
        1.5,
        "• Fan Share Estimates (Task 1)\n• Impact Analysis (Task 3)\n• PTFS Recommendation (Task 4)",
        ha="center",
        va="center",
        fontsize=9,
    )

    # 箭头：Tasks → Output
    ax.annotate(
        "",
        xy=(5.25, 3),
        xytext=(5.25, 4.5),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )
    ax.annotate(
        "",
        xy=(8.75, 3),
        xytext=(8.75, 4.5),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_dataset_overview(output_path: Path) -> None:
    """
    生成数据集概览图。
    """
    apply_theme()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # ═══════════════════════════════════════════════════════════════════
    # 数据规模
    # ═══════════════════════════════════════════════════════════════════
    ax1 = axes[0]
    ax1.set_title("Dataset Scale", fontweight="bold")

    stats = {
        "Seasons": 34,
        "Celebrities": 411,
        "Pro Dancers": 60,
        "Contestant-Weeks": 2738,
        "Eliminations": 261,
    }

    y = np.arange(len(stats))
    bars = ax1.barh(
        y, list(stats.values()), color=COLORS["judge"], edgecolor="black", alpha=0.85
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(list(stats.keys()))
    ax1.set_xlabel("Count")

    for bar, val in zip(bars, stats.values()):
        ax1.text(
            bar.get_width() + 20,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            fontsize=10,
        )

    ax1.set_xlim(0, 3000)

    # ═══════════════════════════════════════════════════════════════════
    # 行业分布
    # ═══════════════════════════════════════════════════════════════════
    ax2 = axes[1]
    ax2.set_title("Industry Distribution", fontweight="bold")

    industries = {
        "Actor": 875,
        "Athlete": 642,
        "TV": 585,
        "Singer": 400,
        "Entertainment": 89,
        "Model": 76,
        "Other": 71,
    }

    colors = [
        COLORS["judge"],
        COLORS["warning"],
        "#9B59B6",
        COLORS["highlight"],
        COLORS["fan"],
        "#1ABC9C",
        COLORS["neutral"],
    ]

    wedges, texts, autotexts = ax2.pie(
        list(industries.values()),
        labels=list(industries.keys()),
        autopct="%1.0f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.75,
    )

    for autotext in autotexts:
        autotext.set_fontsize(8)

    # ═══════════════════════════════════════════════════════════════════
    # 规则时间线
    # ═══════════════════════════════════════════════════════════════════
    ax3 = axes[2]
    ax3.set_title("Voting Rule Timeline", fontweight="bold")

    # 时间线
    ax3.plot([1, 34], [0.5, 0.5], color="black", lw=2)

    # 赛季标记
    seasons = [1, 3, 27, 28, 34]
    labels = [
        "S1\nRank",
        "S3\nPercentage",
        "S27\nBobby Bones\nControversy",
        "S28\nJudges' Save",
        "S34\nCurrent",
    ]
    colors = [
        SYSTEM_COLORS["Rank-based"],
        SYSTEM_COLORS["Percentage-based"],
        COLORS["warning"],
        COLORS["highlight"],
        COLORS["neutral"],
    ]

    for i, (s, label, color) in enumerate(zip(seasons, labels, colors)):
        ax3.plot(s, 0.5, "o", markersize=15, color=color, markeredgecolor="black")
        ax3.text(
            s,
            0.7 if i % 2 == 0 else 0.3,
            label,
            ha="center",
            va="bottom" if i % 2 == 0 else "top",
            fontsize=8,
        )

    ax3.set_xlim(0, 35)
    ax3.set_ylim(0, 1)
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """生成所有综合图表。"""
    print("=" * 60)
    print("跨问题综合图表生成")
    print("=" * 60)

    data = load_all_results()

    # 注意：three_problems_overview 和 methodology_flow 已被移除
    # 因为 overview 有重叠问题，methodology_flow 应该用专业软件绘制

    # 1. 数据集概览（保留，可用于论文）
    print("\n[1/1] 生成数据集概览图...")
    plot_dataset_overview(FIGURE_DIR / "dataset_overview.png")

    print("\n" + "=" * 60)
    print("综合图表生成完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
