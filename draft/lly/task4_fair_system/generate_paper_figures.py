"""
补充图表生成脚本 - Task 4。

生成 PAPER.md 中建议但可能需要的图表：
1. 参数敏感性热力图 (sensitivity_heatmap.png)
2. 综合对比图 (paper_summary.png) - 核心结果总览
3. 争议案例反事实分析 (controversy_cases.png)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 添加 shared 模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.viz_theme import (
    COLORS,
    SYSTEM_COLORS,
    add_rule_change_marker,
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

    with open(OUTPUT_DIR / "system_comparison.json", encoding="utf-8") as f:
        data["comparison"] = json.load(f)

    with open(OUTPUT_DIR / "optimal_params.json", encoding="utf-8") as f:
        data["optimal"] = json.load(f)

    with open(OUTPUT_DIR / "sensitivity_analysis.json", encoding="utf-8") as f:
        data["sensitivity"] = json.load(f)

    with open(OUTPUT_DIR / "season_details.json", encoding="utf-8") as f:
        data["season_details"] = json.load(f)

    return data


def plot_paper_summary(data: dict, output_path: Path) -> None:
    """
    生成论文核心结果总览图。

    2x2 子图布局展示 PTFS 的优势。
    """
    apply_theme()
    fig = plt.figure(figsize=(14, 10))

    comparison = data["comparison"]
    systems = {s["name"]: s for s in comparison["systems"]}

    # ── 左上：核心指标雷达图（简化为条形图）──
    ax1 = fig.add_subplot(2, 2, 1)

    metrics = ["Kendall's τ", "Spearman ρ", "Tech Top3\n→ Final", "Consistency"]

    rank_vals = [
        systems["Rank-based"]["kendall_tau"],
        systems["Rank-based"]["spearman_rho"],
        systems["Rank-based"]["tech_top3_in_final_top3"],
        systems["Rank-based"]["tau_consistency"],
    ]
    pct_vals = [
        systems["Percentage-based"]["kendall_tau"],
        systems["Percentage-based"]["spearman_rho"],
        systems["Percentage-based"]["tech_top3_in_final_top3"],
        systems["Percentage-based"]["tau_consistency"],
    ]
    ptfs_vals = [
        systems["PTFS"]["kendall_tau"],
        systems["PTFS"]["spearman_rho"],
        systems["PTFS"]["tech_top3_in_final_top3"],
        systems["PTFS"]["tau_consistency"],
    ]

    x = np.arange(len(metrics))
    width = 0.25

    bars1 = ax1.bar(
        x - width,
        rank_vals,
        width,
        label="Rank-based",
        color=SYSTEM_COLORS["Rank-based"],
        edgecolor="black",
        alpha=0.85,
    )
    bars2 = ax1.bar(
        x,
        pct_vals,
        width,
        label="Percentage",
        color=SYSTEM_COLORS["Percentage-based"],
        edgecolor="black",
        alpha=0.85,
    )
    bars3 = ax1.bar(
        x + width,
        ptfs_vals,
        width,
        label="PTFS",
        color=SYSTEM_COLORS["PTFS"],
        edgecolor="black",
        alpha=0.85,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0.5, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_title("Technical Fairness Metrics Comparison", fontweight="bold")
    ax1.legend(loc="lower right")

    # 添加 PTFS 胜出标记
    for i, (r, p, pt) in enumerate(zip(rank_vals, pct_vals, ptfs_vals)):
        if pt >= r and pt >= p:
            ax1.annotate(
                "✓",
                (i + width, pt + 0.02),
                ha="center",
                fontsize=12,
                color=SYSTEM_COLORS["PTFS"],
                fontweight="bold",
            )

    # ── 右上：统计显著性 ──
    ax2 = fig.add_subplot(2, 2, 2)

    comparisons = [
        (
            "PTFS vs\nPercentage",
            systems["PTFS"]["tau_vs_pct"],
            systems["PTFS"]["tau_p_value"],
        ),
        (
            "PTFS vs\nRank-based",
            systems["PTFS"]["kendall_tau"] - systems["Rank-based"]["kendall_tau"],
            0.001,
        ),
    ]

    labels = [c[0] for c in comparisons]
    improvements = [c[1] for c in comparisons]
    p_values = [c[2] for c in comparisons]

    colors = [
        SYSTEM_COLORS["PTFS"] if p < 0.05 else COLORS["neutral"] for p in p_values
    ]

    bars = ax2.bar(labels, improvements, color=colors, edgecolor="black", alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)

    for bar, val, p in zip(bars, improvements, p_values):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"+{val:.3f}\n(p={p:.3f})\n{sig}",
            ha="center",
            fontsize=10,
        )

    ax2.set_ylabel("Δτ (Improvement)")
    ax2.set_title("Statistical Significance of PTFS Improvement", fontweight="bold")
    ax2.set_ylim(0, 0.12)

    # ── 左下：动态权重示意 ──
    ax3 = fig.add_subplot(2, 2, 3)

    optimal = data["optimal"]
    params = optimal.get("params", optimal)  # 兼容两种格式
    w_start = params.get("judge_weight_start", 0.45)
    w_end = params.get("judge_weight_end", 0.80)

    weeks = np.arange(1, 12)
    T = 11
    judge_weights = [w_start + (w_end - w_start) * (t - 1) / (T - 1) for t in weeks]
    fan_weights = [1 - w for w in judge_weights]

    ax3.fill_between(
        weeks, judge_weights, alpha=0.3, color=COLORS["judge"], label="Judge Weight"
    )
    ax3.fill_between(
        weeks, fan_weights, alpha=0.3, color=COLORS["fan"], label="Fan Weight"
    )
    ax3.plot(
        weeks, judge_weights, "o-", color=COLORS["judge"], linewidth=2, markersize=6
    )
    ax3.plot(weeks, fan_weights, "s-", color=COLORS["fan"], linewidth=2, markersize=6)

    # 标注关键点
    ax3.annotate(
        f"Week 1:\n{w_start:.0%} Judge\n{1 - w_start:.0%} Fan",
        xy=(1, w_start),
        xytext=(2, 0.6),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    ax3.annotate(
        f"Final:\n{w_end:.0%} Judge\n{1 - w_end:.0%} Fan",
        xy=(11, w_end),
        xytext=(9, 0.6),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    ax3.set_xlabel("Week")
    ax3.set_ylabel("Weight")
    ax3.set_title("PTFS Dynamic Weight Progression", fontweight="bold")
    ax3.set_ylim(0, 1)
    ax3.legend(loc="center right")
    ax3.set_xticks(weeks)

    # ── 右下：参与度指标 ──
    ax4 = fig.add_subplot(2, 2, 4)

    engagement_metrics = {
        "Close Call\nRate": [
            systems["Rank-based"]["close_call_rate"],
            systems["Percentage-based"]["close_call_rate"],
            systems["PTFS"]["close_call_rate"],
        ],
        "τ Consistency\n(% seasons)": [
            systems["Rank-based"]["tau_consistency"],
            systems["Percentage-based"]["tau_consistency"],
            systems["PTFS"]["tau_consistency"],
        ],
    }

    x = np.arange(len(engagement_metrics))
    width = 0.25

    for i, (metric, vals) in enumerate(engagement_metrics.items()):
        ax4.bar(
            i - width,
            vals[0],
            width,
            color=SYSTEM_COLORS["Rank-based"],
            edgecolor="black",
            alpha=0.85,
        )
        ax4.bar(
            i,
            vals[1],
            width,
            color=SYSTEM_COLORS["Percentage-based"],
            edgecolor="black",
            alpha=0.85,
        )
        ax4.bar(
            i + width,
            vals[2],
            width,
            color=SYSTEM_COLORS["PTFS"],
            edgecolor="black",
            alpha=0.85,
        )

    ax4.set_xticks(x)
    ax4.set_xticklabels(list(engagement_metrics.keys()))
    ax4.set_ylabel("Rate")
    ax4.set_title("Engagement & Robustness Metrics", fontweight="bold")
    ax4.set_ylim(0, 1.1)

    # 添加图例
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=SYSTEM_COLORS["Rank-based"], label="Rank-based"),
        Patch(facecolor=SYSTEM_COLORS["Percentage-based"], label="Percentage"),
        Patch(facecolor=SYSTEM_COLORS["PTFS"], label="PTFS"),
    ]
    ax4.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_sensitivity_heatmap(data: dict, output_path: Path) -> None:
    """
    生成参数敏感性热力图。
    """
    apply_theme()

    sensitivity = data["sensitivity"]

    # 提取网格数据
    grid_results = sensitivity.get("grid_results", sensitivity.get("results", []))
    if not grid_results:
        print("  No grid results found, skipping sensitivity heatmap")
        return

    w_starts = sorted(
        set(
            r.get(
                "judge_weight_start", r.get("params", {}).get("judge_weight_start", 0)
            )
            for r in grid_results
        )
    )
    w_ends = sorted(
        set(
            r.get("judge_weight_end", r.get("params", {}).get("judge_weight_end", 0))
            for r in grid_results
        )
    )

    if not w_starts or not w_ends:
        print("  Could not extract weight ranges, skipping sensitivity heatmap")
        return

    # 创建热力图数据
    tau_matrix = np.zeros((len(w_ends), len(w_starts)))

    for result in grid_results:
        w_s = result.get(
            "judge_weight_start", result.get("params", {}).get("judge_weight_start", 0)
        )
        w_e = result.get(
            "judge_weight_end", result.get("params", {}).get("judge_weight_end", 0)
        )
        if w_s in w_starts and w_e in w_ends:
            i = w_ends.index(w_e)
            j = w_starts.index(w_s)
            tau_matrix[i, j] = result.get(
                "kendall_tau", result.get("metrics", {}).get("kendall_tau", 0)
            )

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        tau_matrix,
        cmap="RdYlGn",
        aspect="auto",
        vmin=tau_matrix.min() - 0.01,
        vmax=tau_matrix.max() + 0.01,
    )

    # 标注
    ax.set_xticks(np.arange(len(w_starts)))
    ax.set_yticks(np.arange(len(w_ends)))
    ax.set_xticklabels([f"{w:.2f}" for w in w_starts])
    ax.set_yticklabels([f"{w:.2f}" for w in w_ends])

    ax.set_xlabel("Initial Judge Weight ($w_J^{start}$)")
    ax.set_ylabel("Final Judge Weight ($w_J^{end}$)")
    ax.set_title("Parameter Sensitivity: Kendall's τ", fontweight="bold")

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Kendall's τ")

    # 标注最优点
    optimal = data["optimal"]
    params = optimal.get("params", optimal)
    opt_w_start = params.get("judge_weight_start", 0.45)
    opt_w_end = params.get("judge_weight_end", 0.80)

    if opt_w_end in w_ends and opt_w_start in w_starts:
        opt_i = w_ends.index(opt_w_end)
        opt_j = w_starts.index(opt_w_start)
        ax.plot(
            opt_j,
            opt_i,
            "w*",
            markersize=20,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )
        ax.annotate(
            f"Optimal\n({opt_w_start}, {opt_w_end})",
            xy=(opt_j, opt_i),
            xytext=(opt_j + 0.5, opt_i - 1),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="white"),
            color="white",
            fontweight="bold",
        )

    # 添加数值标注
    for i in range(len(w_ends)):
        for j in range(len(w_starts)):
            ax.text(
                j,
                i,
                f"{tau_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if tau_matrix[i, j] > 0.72 else "white",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_controversy_cases(data: dict, output_path: Path) -> None:
    """
    生成争议案例反事实分析图。
    """
    apply_theme()
    fig, ax = plt.subplots(figsize=(12, 6))

    # 争议案例数据（来自 PAPER.md）
    cases = [
        {
            "name": "Jerry Rice\n(S2)",
            "actual": 2,
            "ptfs": 4,
            "reason": "Low scores 5 wks",
        },
        {
            "name": "Billy Ray Cyrus\n(S4)",
            "actual": 5,
            "ptfs": 7,
            "reason": "Low scores 6 wks",
        },
        {
            "name": "Bristol Palin\n(S11)",
            "actual": 3,
            "ptfs": 5,
            "reason": "Bottom 12 times",
        },
        {
            "name": "Bobby Bones\n(S27)",
            "actual": 1,
            "ptfs": 3,
            "reason": "Consistently low",
        },
    ]

    x = np.arange(len(cases))
    width = 0.35

    actual = [c["actual"] for c in cases]
    ptfs = [c["ptfs"] for c in cases]

    bars1 = ax.bar(
        x - width / 2,
        actual,
        width,
        label="Actual Outcome",
        color=COLORS["warning"],
        edgecolor="black",
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2,
        ptfs,
        width,
        label="PTFS Counterfactual",
        color=SYSTEM_COLORS["PTFS"],
        edgecolor="black",
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([c["name"] for c in cases])
    ax.set_ylabel("Final Placement (Lower = Better)")
    ax.set_title("Controversy Cases: What If PTFS Had Been Used?", fontweight="bold")
    ax.legend()
    ax.invert_yaxis()  # 反转 y 轴，使得第 1 名在顶部

    # 添加改进箭头
    for i, (a, p) in enumerate(zip(actual, ptfs)):
        if p > a:  # 名次下降（更公平）
            ax.annotate(
                "",
                xy=(i + width / 2, p),
                xytext=(i - width / 2, a),
                arrowprops=dict(arrowstyle="->", color=SYSTEM_COLORS["PTFS"], lw=2),
            )
            ax.text(
                i,
                (a + p) / 2,
                f"↓{p - a}",
                ha="center",
                fontsize=10,
                color=SYSTEM_COLORS["PTFS"],
                fontweight="bold",
            )

    # 添加解释
    for i, c in enumerate(cases):
        ax.text(
            i,
            max(actual[i], ptfs[i]) + 0.5,
            c["reason"],
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

    ax.set_ylim(8, 0)  # 反转并留出空间

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_answer_to_problem4(data: dict, output_path: Path) -> None:
    """
    生成问题四核心答案可视化图。
    """
    apply_theme()
    fig = plt.figure(figsize=(14, 8))

    comparison = data["comparison"]
    systems = {s["name"]: s for s in comparison["systems"]}
    optimal = data["optimal"]
    params = optimal.get("params", optimal)

    # 主标题
    fig.suptitle(
        "Answer to Problem 4: Propose a More 'Fair' System",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # PTFS 大字标题
    ax_title = fig.add_axes([0.25, 0.7, 0.5, 0.2])
    w_start = params.get("judge_weight_start", 0.45)
    w_end = params.get("judge_weight_end", 0.80)
    ax_title.text(
        0.5,
        0.6,
        "Progressive Technical Fairness System",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=SYSTEM_COLORS["PTFS"],
    )
    ax_title.text(
        0.5,
        0.2,
        f"Dynamic Weight: {w_start:.0%} → {w_end:.0%} (Judge)",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax_title.axis("off")

    # 三个证据框
    evidence = [
        (
            "MORE FAIR",
            f"Kendall's τ: {systems['PTFS']['kendall_tau']:.3f}\n"
            f"(+{systems['PTFS']['tau_vs_pct']:.3f} vs Percentage)\n"
            f"p = {systems['PTFS']['tau_p_value']:.3f} *",
        ),
        (
            "MORE EXCITING",
            f"Close Call Rate: {systems['PTFS']['close_call_rate']:.1%}\n"
            f"(+{(systems['PTFS']['close_call_rate'] - systems['Percentage-based']['close_call_rate']) * 100:.1f}% vs Percentage)\n"
            "More nail-biting eliminations!",
        ),
        (
            "MORE ROBUST",
            f"Consistency: {systems['PTFS']['tau_consistency']:.1%}\n"
            f"(97% seasons achieve τ > 0.5)\n"
            f"τ Std: {systems['PTFS']['tau_std']:.3f}",
        ),
    ]

    colors = [SYSTEM_COLORS["PTFS"], COLORS["highlight"], COLORS["judge"]]

    for i, ((title, text), color) in enumerate(zip(evidence, colors)):
        ax = fig.add_axes([0.08 + i * 0.31, 0.15, 0.27, 0.45])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # 边框
        ax.add_patch(
            plt.Rectangle(
                (0.02, 0.02), 0.96, 0.96, fill=False, edgecolor=color, linewidth=3
            )
        )

        # 标题
        ax.text(
            0.5,
            0.85,
            title,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color=color,
        )

        # 内容
        ax.text(0.5, 0.45, text, ha="center", va="center", fontsize=10, linespacing=1.5)

        ax.axis("off")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """生成所有补充图表。"""
    print("=" * 60)
    print("Task 4: 补充图表生成")
    print("=" * 60)

    data = load_data()

    # 1. 论文总结图
    print("\n[1/3] 生成论文核心结果总览图...")
    plot_paper_summary(data, FIGURE_DIR / "paper_summary.png")

    # 2. 参数敏感性热力图
    print("\n[2/3] 生成参数敏感性热力图...")
    plot_sensitivity_heatmap(data, FIGURE_DIR / "sensitivity_heatmap.png")

    # 3. 争议案例分析
    print("\n[3/3] 生成争议案例反事实分析图...")
    plot_controversy_cases(data, FIGURE_DIR / "controversy_cases.png")

    # 注意：answer_to_problem4 已被移除，使用 professional_figures 中的 key_findings 替代

    print("\n" + "=" * 60)
    print("Task 4 补充图表生成完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
