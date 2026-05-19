"""
补充图表生成脚本 - Task 3。

生成 PAPER.md 中建议但可能需要更新的图表：
1. 综合分析图 (paper_summary.png) - 核心发现总览
2. 运动员悖论可视化 (athlete_paradox.png)
3. γ 敏感性分析图 (gamma_sensitivity.png)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 添加 shared 模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.viz_theme import (
    COLORS,
    INDUSTRY_COLORS,
    PRO_CATEGORY_COLORS,
    apply_theme,
)

# 设置随机种子
np.random.seed(42)
random.seed(42)

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = SCRIPT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def load_data() -> dict:
    """加载所有需要的数据。"""
    data = {}

    with open(OUTPUT_DIR / "conclusions.json", encoding="utf-8") as f:
        data["conclusions"] = json.load(f)

    with open(OUTPUT_DIR / "variance_decomposition.json", encoding="utf-8") as f:
        data["variance"] = json.load(f)

    with open(OUTPUT_DIR / "fixed_effects_analysis.json", encoding="utf-8") as f:
        data["fixed_effects"] = json.load(f)

    with open(OUTPUT_DIR / "gamma_sensitivity.json", encoding="utf-8") as f:
        data["gamma"] = json.load(f)

    # 加载 IDI 数据
    idi_path = OUTPUT_DIR / "impact_divergence_index.csv"
    if idi_path.exists():
        data["idi"] = pd.read_csv(idi_path)

    # 加载职业舞伴数据
    pro_path = OUTPUT_DIR / "pro_rankings.csv"
    if pro_path.exists():
        data["pro_rankings"] = pd.read_csv(pro_path)

    return data


def plot_paper_summary(data: dict, output_path: Path) -> None:
    """
    生成论文核心发现总览图。

    2x2 子图布局展示关键发现。
    """
    apply_theme()
    fig = plt.figure(figsize=(14, 10))

    conclusions = data["conclusions"]
    variance = data["variance"]

    # ── 左上：运动员悖论 ──
    ax1 = fig.add_subplot(2, 2, 1)

    athlete = conclusions["athlete_paradox"]
    labels = ["Judge Score\nCoefficient", "Fan Share\nCoefficient"]
    values = [athlete["judge_coefficient"], athlete["fan_coefficient"]]
    colors = [COLORS["judge"] if v < 0 else COLORS["warning"] for v in values]
    colors = [COLORS["judge"], COLORS["fan"]]  # 使用标准颜色

    bars = ax1.bar(labels, values, color=colors, edgecolor="black", alpha=0.85)
    ax1.axhline(0, color="black", linewidth=0.8)

    for bar, val in zip(bars, values):
        sign = "+" if val > 0 else ""
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.01 if val > 0 else -0.02),
            f"{sign}{val:.3f}",
            ha="center",
            va="bottom" if val > 0 else "top",
            fontsize=11,
            fontweight="bold",
        )

    ax1.set_ylabel("Coefficient (relative to Actor)")
    ax1.set_title(
        "The Athlete Paradox\n(Opposite Effects on Two Systems)", fontweight="bold"
    )
    ax1.set_ylim(-0.2, 0.1)

    # 添加注释
    ax1.annotate(
        "Judges penalize\nathletes",
        xy=(0, athlete["judge_coefficient"]),
        xytext=(0.3, -0.05),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=COLORS["judge"]),
    )
    ax1.annotate(
        "Fans reward\nathletes!",
        xy=(1, athlete["fan_coefficient"]),
        xytext=(0.7, 0.08),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=COLORS["fan"]),
    )

    # ── 右上：ICC 对比 ──
    ax2 = fig.add_subplot(2, 2, 2)

    icc_judge = variance["judge"]["icc"]
    icc_fan = variance["fan"]["icc"]

    labels = ["Judge Score\nModel", "Fan Share\nModel"]
    values = [icc_judge, icc_fan]

    bars = ax2.bar(
        labels,
        values,
        color=[COLORS["judge"], COLORS["fan"]],
        edgecolor="black",
        alpha=0.85,
    )

    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    ax2.set_ylabel("Intraclass Correlation (ICC)")
    ax2.set_title(
        "Pro Dancer Variance Contribution\n(2.3× higher for Fan Voting)",
        fontweight="bold",
    )
    ax2.set_ylim(0, 0.3)

    # 添加比例箭头
    ax2.annotate(
        "",
        xy=(1, icc_fan),
        xytext=(0, icc_judge),
        arrowprops=dict(arrowstyle="<->", color=COLORS["highlight"], lw=2),
    )
    ax2.text(
        0.5,
        0.15,
        "2.3×",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=COLORS["highlight"],
    )

    # ── 左下：γ 敏感性 ──
    ax3 = fig.add_subplot(2, 2, 3)

    gamma_data = data["gamma"]
    models = ["Original\n(with week)", "Without\nweek", "With\ninteraction"]
    gammas = [
        gamma_data.get("original", {}).get("gamma", 0.004),
        gamma_data.get("without_week", {}).get("gamma", 0.161),
        gamma_data.get("with_interaction", {}).get("gamma", -0.040),
    ]
    significant = [
        not gamma_data.get("original", {}).get("significant", False),
        gamma_data.get("without_week", {}).get("significant", True),
        gamma_data.get("with_interaction", {}).get("significant", True),
    ]

    colors = [COLORS["neutral"] if not s else COLORS["success"] for s in significant]

    bars = ax3.bar(models, gammas, color=colors, edgecolor="black", alpha=0.85)
    ax3.axhline(0, color="black", linewidth=0.8)

    for bar, val, sig in zip(bars, gammas, significant):
        star = "***" if sig else "ns"
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.01 if val >= 0 else -0.02),
            f"{val:.3f}\n{star}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=10,
        )

    ax3.set_ylabel("γ (Performance Conversion)")
    ax3.set_title(
        "γ Sensitivity Analysis\n(Multicollinearity Confirmed)", fontweight="bold"
    )

    # ── 右下：IDI 关键发现 ──
    ax4 = fig.add_subplot(2, 2, 4)

    if "idi" in data and len(data["idi"]) > 0:
        idi_df = data["idi"].sort_values("IDI", ascending=False)
        top_vars = idi_df.head(5)

        y_pos = np.arange(len(top_vars))
        bars = ax4.barh(
            y_pos,
            top_vars["IDI"].values,
            color=[
                COLORS["warning"]
                if v > 3
                else COLORS["highlight"]
                if v > 1.96
                else COLORS["neutral"]
                for v in top_vars["IDI"].values
            ],
            edgecolor="black",
            alpha=0.85,
        )

        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(top_vars["variable"].values)
        ax4.axvline(1.96, color="gray", linestyle="--", label="p=0.05 threshold")
        ax4.axvline(
            3.0, color=COLORS["warning"], linestyle="--", label="Substantive (IDI=3)"
        )
        ax4.set_xlabel("Impact Divergence Index")
        ax4.set_title("Top Divergent Effects\n(Judge vs Fan)", fontweight="bold")
        ax4.legend(loc="lower right", fontsize=8)
    else:
        # 备用数据
        vars_names = ["Week", "Age", "TV", "Athlete", "Other"]
        idi_values = [27.26, 17.19, 6.79, 3.26, 3.29]

        y_pos = np.arange(len(vars_names))
        bars = ax4.barh(
            y_pos,
            idi_values,
            color=[
                COLORS["warning"] if v > 3 else COLORS["highlight"] for v in idi_values
            ],
            edgecolor="black",
            alpha=0.85,
        )

        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(vars_names)
        ax4.axvline(1.96, color="gray", linestyle="--", label="p=0.05")
        ax4.axvline(3.0, color=COLORS["warning"], linestyle="--", label="IDI=3")
        ax4.set_xlabel("Impact Divergence Index")
        ax4.set_title("Top Divergent Effects", fontweight="bold")
        ax4.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_athlete_paradox_detail(data: dict, output_path: Path) -> None:
    """
    生成运动员悖论详细分析图。
    """
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    conclusions = data["conclusions"]
    athlete = conclusions["athlete_paradox"]

    # ── 左图：系数对比 ──
    ax1 = axes[0]

    industries = ["Actor\n(baseline)", "Athlete", "Singer", "TV", "Model", "Other"]
    # 使用固定效应数据
    judge_coefs = [0, -0.129, -0.021, -0.419, -0.535, -0.689]
    fan_coefs = [0, 0.052, -0.019, -0.033, -0.217, -0.246]

    x = np.arange(len(industries))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        judge_coefs,
        width,
        label="Judge Score",
        color=COLORS["judge"],
        edgecolor="black",
        alpha=0.85,
    )
    bars2 = ax1.bar(
        x + width / 2,
        fan_coefs,
        width,
        label="Fan Share",
        color=COLORS["fan"],
        edgecolor="black",
        alpha=0.85,
    )

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(industries, fontsize=9)
    ax1.set_ylabel("Coefficient (relative to Actor)")
    ax1.set_title("Industry Effects on Judge Scores vs Fan Shares", fontweight="bold")
    ax1.legend()

    # 高亮运动员
    ax1.annotate(
        "OPPOSITE\nDIRECTIONS!",
        xy=(1, 0),
        xytext=(1.5, 0.3),
        fontsize=10,
        fontweight="bold",
        color=COLORS["warning"],
        arrowprops=dict(arrowstyle="->", color=COLORS["warning"], lw=2),
    )

    # ── 右图：方向图示 ──
    ax2 = axes[1]
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)

    # 评委视角
    ax2.annotate(
        "",
        xy=(-0.8, 0.3),
        xytext=(0, 0.3),
        arrowprops=dict(arrowstyle="->", color=COLORS["judge"], lw=3),
    )
    ax2.text(
        -0.4,
        0.5,
        "JUDGES",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["judge"],
    )
    ax2.text(-0.4, 0.15, "Athletes score LOWER\n(β = −0.129)", ha="center", fontsize=10)

    # 粉丝视角
    ax2.annotate(
        "",
        xy=(0.8, -0.3),
        xytext=(0, -0.3),
        arrowprops=dict(arrowstyle="->", color=COLORS["fan"], lw=3),
    )
    ax2.text(
        0.4,
        -0.1,
        "FANS",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["fan"],
    )
    ax2.text(0.4, -0.45, "Athletes vote HIGHER\n(β = +0.052)", ha="center", fontsize=10)

    # 中心标签
    ax2.text(
        0,
        0,
        "ATHLETE",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(
            boxstyle="circle,pad=0.3", facecolor="lightyellow", edgecolor="black"
        ),
    )

    ax2.set_title(
        "The Athlete Paradox:\nOpposing Evaluation Directions", fontweight="bold"
    )
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_answer_to_problem3(data: dict, output_path: Path) -> None:
    """
    生成问题三核心答案可视化图。

    "Do they impact judges' scores and fan votes in the same way?"
    Answer: NO.
    """
    apply_theme()
    fig = plt.figure(figsize=(12, 8))

    # 标题
    fig.suptitle(
        "Answer to Problem 3: Do factors impact judges and fans THE SAME WAY?",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 大字 NO
    ax_main = fig.add_axes([0.35, 0.55, 0.3, 0.35])
    ax_main.text(
        0.5,
        0.5,
        "NO",
        ha="center",
        va="center",
        fontsize=72,
        fontweight="bold",
        color=COLORS["warning"],
    )
    ax_main.axis("off")

    # 四个证据框
    evidence = [
        ("Athlete Paradox", "Judge: β = −0.13\nFan: β = +0.05\n(Opposite directions!)"),
        ("Age Penalty", "Judge: 18× stronger\nthan Fan\n(IDI = 17.19)"),
        ("Pro Dancer ICC", "Fan: 21.8%\nJudge: 9.5%\n(2.3× difference)"),
        ("γ Conversion", "γ ≈ 0 (ns)\nbut significant\nwhen week removed"),
    ]

    positions = [(0.08, 0.08), (0.33, 0.08), (0.58, 0.08), (0.83, 0.08)]
    colors = [COLORS["warning"], COLORS["highlight"], COLORS["fan"], COLORS["neutral"]]

    for (title, text), (x, y), color in zip(evidence, positions, colors):
        ax = fig.add_axes([x, y, 0.18, 0.35])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # 标题
        ax.text(0.5, 0.9, title, ha="center", va="top", fontsize=11, fontweight="bold")

        # 内容
        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2),
        )

        ax.axis("off")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """生成所有补充图表。"""
    print("=" * 60)
    print("Task 3: 补充图表生成")
    print("=" * 60)

    data = load_data()

    # 1. 论文总结图
    print("\n[1/2] 生成论文核心发现总览图...")
    plot_paper_summary(data, FIGURE_DIR / "paper_summary.png")

    # 2. 运动员悖论详细图
    print("\n[2/2] 生成运动员悖论详细分析图...")
    plot_athlete_paradox_detail(data, FIGURE_DIR / "athlete_paradox_detail.png")

    # 注意：answer_to_problem3 已被移除，使用 professional_figures 中的 key_findings 替代

    print("\n" + "=" * 60)
    print("Task 3 补充图表生成完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
