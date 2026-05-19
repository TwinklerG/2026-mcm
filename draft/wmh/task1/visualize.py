"""
DWTS 粉丝投票估计的可视化。

生成论文级图表：
1. MCMC 诊断（迭代、自相关、ESS）
2. 按赛季/周的粉丝份额分布
3. 不确定性可视化（可信区间）
4. 异常检测案例
5. 验证结果总结
6. [NEW] 堆叠柱状图：Strict Accuracy vs Bottom-2 Recall
7. [NEW] CI 宽度分布图（箱线图/直方图）
8. [NEW] 决赛排名准确率
9. [NEW] 结构漂移分析（S28 规则变化）
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# 设置随机种子以确保可重复性
np.random.seed(42)
random.seed(42)

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = SCRIPT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# 论文发表的样式设置
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# 颜色方案
COLORS = {
    "strict": "#2E86AB",  # 深蓝 - Strict Accuracy
    "bottom2": "#A23B72",  # 紫红 - Bottom-2 Only
    "overall": "#F18F01",  # 橙色 - Overall
    "pre_s28": "#C73E1D",  # 红色 - S1-S27
    "post_s28": "#3B1F2B",  # 深紫 - S28+
    "success": "#2ECC71",  # 绿色
    "warning": "#E74C3C",  # 红色
    "neutral": "#95A5A6",  # 灰色
}


def plot_reconstruction_accuracy_by_season(
    results: dict,
    recon_details: dict,
    output_path: Path | None = None,
):
    """绘制每个赛季的重构准确率（简单版本）。"""
    per_season = recon_details.get("per_season", {})
    if not per_season:
        print("没有每赛季数据可用。")
        return

    seasons = sorted([int(s) for s in per_season.keys()])
    accuracies = [per_season[str(s)]["accuracy"] for s in seasons]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(seasons, accuracies, color=COLORS["strict"], edgecolor="navy", alpha=0.8)

    # 突出显示总体准确率
    overall = results["validation"]["reconstruction_accuracy"]
    ax.axhline(
        overall,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Overall: {overall:.1%}",
    )

    # 标记 S28 规则变化
    ax.axvline(
        27.5, color="orange", linestyle=":", linewidth=2, label="Rule Change (S28)"
    )

    ax.set_xlabel("Season")
    ax.set_ylabel("Reconstruction Accuracy")
    ax.set_title("Reconstruction Accuracy by Season (Strict)")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(seasons[::2])
    ax.legend()

    # 标注低准确率赛季
    for s, acc in zip(seasons, accuracies):
        if acc < 0.7:
            ax.annotate(
                f"S{s}\n{acc:.0%}",
                (s, acc + 0.05),
                ha="center",
                fontsize=8,
                color="red",
            )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_stacked_accuracy_by_season(
    bottom2_results: dict,
    output_path: Path | None = None,
):
    """绘制堆叠柱状图：Strict Accuracy vs Bottom-2 Only。"""
    per_season = bottom2_results.get("per_season", {})
    if not per_season:
        print("没有 Bottom-2 数据可用。")
        return

    # 键可能是字符串或整数
    seasons = sorted([int(s) for s in per_season.keys()])
    strict_acc = []
    bottom2_only = []

    for s in seasons:
        # 尝试字符串键和整数键
        data = per_season.get(str(s)) or per_season.get(s)
        if data is None:
            strict_acc.append(0)
            bottom2_only.append(0)
            continue

        total = data["total"]
        if total > 0:
            strict = data["strict_correct"] / total
            bottom2 = data["bottom2_correct"] / total
            # Bottom-2 Only = Bottom-2 Recall - Strict Accuracy
            strict_acc.append(strict)
            bottom2_only.append(max(0, bottom2 - strict))
        else:
            strict_acc.append(0)
            bottom2_only.append(0)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.array(seasons)
    width = 0.8

    # 堆叠柱状图
    ax.bar(
        x,
        strict_acc,
        width,
        label="Strict Accuracy (Pred = Eliminated)",
        color=COLORS["strict"],
        edgecolor="black",
        alpha=0.9,
    )
    ax.bar(
        x,
        bottom2_only,
        width,
        bottom=strict_acc,
        label="Bottom-2 Only (Pred in Bottom 2, but not #1)",
        color=COLORS["bottom2"],
        edgecolor="black",
        alpha=0.7,
    )

    # 标记 S28 规则变化
    ax.axvline(
        27.5,
        color="orange",
        linestyle="--",
        linewidth=2.5,
        label="Judges' Save Rule (S28+)",
    )

    # 添加注释框
    ax.annotate(
        "Rule Change:\nJudges' Save\nintroduced",
        xy=(28, 0.95),
        xytext=(31, 0.85),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="orange"),
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="orange"
        ),
    )

    # 整体统计
    overall = bottom2_results["overall"]
    pre_s28 = bottom2_results["pre_s28"]
    post_s28 = bottom2_results["post_s28"]

    # 添加文本框显示统计数据
    stats_text = (
        f"Overall: Strict={overall['strict_accuracy']:.1%}, B2-Recall={overall['bottom2_recall']:.1%}\n"
        f"S1-S27:  Strict={pre_s28['strict_accuracy']:.1%}, B2-Recall={pre_s28['bottom2_recall']:.1%}\n"
        f"S28+:    Strict={post_s28['strict_accuracy']:.1%}, B2-Recall={post_s28['bottom2_recall']:.1%}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        fontfamily="monospace",
    )

    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        "Model Performance: Strict Accuracy vs Bottom-2 Recall by Season", fontsize=14
    )
    ax.set_ylim(0, 1.15)
    ax.set_xticks(seasons[::2])
    ax.legend(loc="upper right")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_structural_drift_analysis(
    bottom2_results: dict,
    stability_results: dict | None = None,
    output_path: Path | None = None,
):
    """结构漂移分析图：展示 S28 规则变化的影响。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. 规则变化前后的对比
    ax = axes[0]
    pre_s28 = bottom2_results["pre_s28"]
    post_s28 = bottom2_results["post_s28"]

    metrics = ["Strict\nAccuracy", "Bottom-2\nRecall"]
    pre_values = [pre_s28["strict_accuracy"], pre_s28["bottom2_recall"]]
    post_values = [post_s28["strict_accuracy"], post_s28["bottom2_recall"]]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        pre_values,
        width,
        label="S1-S27 (Percentage Method)",
        color=COLORS["pre_s28"],
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        post_values,
        width,
        label="S28+ (Rank + Judges' Save)",
        color=COLORS["post_s28"],
        alpha=0.8,
    )

    ax.set_ylabel("Accuracy")
    ax.set_title("Impact of Rule Change")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()

    # 添加数值标签
    for bar, val in zip(bars1, pre_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=9,
        )
    for bar, val in zip(bars2, post_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=9,
        )

    # 2. 结构漂移指标
    ax = axes[1]
    drift_metrics = {
        "Strict Accuracy\nDrop": pre_s28["strict_accuracy"]
        - post_s28["strict_accuracy"],
        "Bottom-2 Recall\nDrop": pre_s28["bottom2_recall"] - post_s28["bottom2_recall"],
    }

    if stability_results:
        drift_metrics["Generalization\nGap"] = stability_results.get(
            "generalization_gap", 0
        )

    colors = [
        COLORS["warning"] if v > 0.05 else COLORS["success"]
        for v in drift_metrics.values()
    ]
    bars = ax.bar(
        drift_metrics.keys(), drift_metrics.values(), color=colors, edgecolor="black"
    )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(
        0.1, color="red", linestyle="--", alpha=0.5, label="Significant Drift Threshold"
    )
    ax.set_ylabel("Performance Drop")
    ax.set_title("Structural Drift Indicators")
    ax.legend()

    for bar, val in zip(bars, drift_metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.1%}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_fan_share_distribution(
    df_fan_shares: pl.DataFrame,
    output_path: Path | None = None,
):
    """绘制估计粉丝份额的分布。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 左侧：直方图
    shares = df_fan_shares["estimated_fan_share"].to_numpy()
    axes[0].hist(shares, bins=50, color="steelblue", edgecolor="navy", alpha=0.7)
    axes[0].set_xlabel("Estimated Fan Share")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Fan Share Estimates")
    axes[0].axvline(
        np.median(shares),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(shares):.3f}",
    )
    axes[0].legend()

    # 右侧：按淘汰状态
    eliminated = df_fan_shares.filter(pl.col("is_eliminated"))[
        "estimated_fan_share"
    ].to_numpy()
    survived = df_fan_shares.filter(~pl.col("is_eliminated"))[
        "estimated_fan_share"
    ].to_numpy()

    axes[1].boxplot(
        [eliminated, survived],
        tick_labels=["Eliminated", "Survived"],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
    )
    axes[1].set_ylabel("Estimated Fan Share")
    axes[1].set_title("Fan Shares: Eliminated vs Survived")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_validation_summary(
    results: dict,
    output_path: Path | None = None,
):
    """创建验证指标的总结可视化。"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. Main Metrics Bar Chart
    metrics = {
        "Reconstruction\nAccuracy": results["validation"]["reconstruction_accuracy"],
        "Posterior\nConsistency": results["validation"]["posterior_consistency_mean"],
        "Constraint\nSatisfaction": results["validation"][
            "constraint_satisfaction_rate"
        ],
    }

    colors = ["steelblue", "seagreen", "coral"]
    bars = axes[0].bar(
        metrics.keys(), metrics.values(), color=colors, edgecolor="black"
    )
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Validation Metrics")

    for bar, val in zip(bars, metrics.values()):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=10,
        )

    # 2. MCMC Diagnostics
    diag = results["diagnostics"]
    diag_metrics = {
        "Mean ESS": diag["mean_ess"],
        "Min ESS": diag["min_ess"],
    }

    axes[1].bar(
        diag_metrics.keys(),
        diag_metrics.values(),
        color="mediumpurple",
        edgecolor="black",
    )
    axes[1].axhline(100, color="red", linestyle="--", label="ESS=100 threshold")
    axes[1].set_ylabel("Effective Sample Size")
    axes[1].set_title("MCMC Diagnostics")
    axes[1].legend()

    # 3. Stability/Generalization (if available)
    if "stability" in results:
        stab = results["stability"]
        stab_data = {
            "Train\nAccuracy": stab["train_accuracy"],
            "Test\nAccuracy": stab["test_accuracy"],
        }
        bars = axes[2].bar(
            stab_data.keys(),
            stab_data.values(),
            color=["lightgreen", "lightcoral"],
            edgecolor="black",
        )
        axes[2].set_ylim(0, 1.1)
        axes[2].set_ylabel("Accuracy")
        axes[2].set_title(
            f"Cross-Season Generalization\n(Gap: {stab['generalization_gap']:.1%})"
        )

        for bar, val in zip(bars, stab_data.values()):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.1%}",
                ha="center",
                fontsize=10,
            )
    else:
        axes[2].text(
            0.5,
            0.5,
            "Stability analysis\nnot available",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_title("Cross-Season Generalization")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_credible_interval_distribution(
    results: dict,
    output_path: Path | None = None,
):
    """绘制可信区间宽度的分布（简单版本）。"""
    fig, ax = plt.subplots(figsize=(6, 4))

    # 我们没有保存单个 CRI 值，所以使用总结数据
    mean_cri = results["uncertainty"]["mean_cri_width"]
    median_cri = results["uncertainty"]["median_cri_width"]

    ax.bar(
        ["Mean", "Median"],
        [mean_cri, median_cri],
        color=[COLORS["strict"], COLORS["bottom2"]],
        edgecolor="black",
    )
    ax.set_ylabel("95% Credible Interval Width")
    ax.set_title("Posterior Uncertainty")

    for i, val in enumerate([mean_cri, median_cri]):
        ax.text(i, val + 0.001, f"{val:.4f}", ha="center")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_detailed_ci_distribution(
    detailed_uncertainty: dict,
    output_path: Path | None = None,
):
    """
    详细的 CI 宽度分布图：箱线图和直方图。

    回答问题："is that certainty always the same for each contestant/week?"
    """
    raw_data = detailed_uncertainty.get("raw_data", [])
    if not raw_data:
        print("没有详细的不确定性数据可用。")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. CI 宽度直方图
    ax = axes[0, 0]
    ci_widths = [d["ci_width"] for d in raw_data]
    ax.hist(ci_widths, bins=50, color=COLORS["strict"], edgecolor="black", alpha=0.7)
    ax.axvline(
        np.mean(ci_widths),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(ci_widths):.4f}",
    )
    ax.axvline(
        np.median(ci_widths),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(ci_widths):.4f}",
    )
    ax.set_xlabel("95% Credible Interval Width")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Posterior Uncertainty")
    ax.legend()

    # 2. 按淘汰状态的箱线图
    ax = axes[0, 1]
    eliminated_ci = [d["ci_width"] for d in raw_data if d["is_eliminated"]]
    survived_ci = [d["ci_width"] for d in raw_data if not d["is_eliminated"]]

    bp = ax.boxplot(
        [eliminated_ci, survived_ci],
        tick_labels=["Eliminated", "Survived"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor(COLORS["warning"])
    bp["boxes"][1].set_facecolor(COLORS["success"])
    ax.set_ylabel("95% CI Width")
    ax.set_title("Uncertainty: Eliminated vs Survived")

    # 添加统计文本
    ax.text(
        0.95,
        0.95,
        f"Eliminated: μ={np.mean(eliminated_ci):.4f}\nSurvived: μ={np.mean(survived_ci):.4f}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 3. 按行业的箱线图
    ax = axes[1, 0]
    by_industry = detailed_uncertainty.get("by_industry", {})
    if by_industry:
        industries = sorted(
            by_industry.keys(), key=lambda x: -by_industry[x]["mean_ci_width"]
        )[:10]
        industry_data = []
        for ind in industries:
            ind_ci = [d["ci_width"] for d in raw_data if d["industry"] == ind]
            industry_data.append(ind_ci)

        bp = ax.boxplot(
            industry_data,
            tick_labels=[i[:12] for i in industries],
            patch_artist=True,
            vert=True,
        )
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(plt.cm.viridis(i / len(industries)))
        ax.set_xticklabels([i[:12] for i in industries], rotation=45, ha="right")
        ax.set_ylabel("95% CI Width")
        ax.set_title("Uncertainty by Celebrity Industry")

    # 4. 按周位置的对比
    ax = axes[1, 1]
    by_week = detailed_uncertainty.get("by_week_position", {})
    if by_week:
        early_ci = [d["ci_width"] for d in raw_data if d["week"] <= 4]
        late_ci = [d["ci_width"] for d in raw_data if d["week"] > 4]

        bp = ax.boxplot(
            [early_ci, late_ci],
            tick_labels=["Early Weeks (1-4)", "Late Weeks (5+)"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor(COLORS["pre_s28"])
        bp["boxes"][1].set_facecolor(COLORS["post_s28"])
        ax.set_ylabel("95% CI Width")
        ax.set_title("Uncertainty by Week Position")

        # 添加统计文本
        ax.text(
            0.95,
            0.95,
            f"Early: μ={np.mean(early_ci):.4f}, n={len(early_ci)}\n"
            f"Late: μ={np.mean(late_ci):.4f}, n={len(late_ci)}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_controversial_vs_stable(
    detailed_uncertainty: dict,
    output_path: Path | None = None,
):
    """
    争议型 vs 稳态型选手可视化。

    高 CI 宽度 = "争议型选手"（粉丝投票不确定性大）
    低 CI 宽度 = "稳态型选手"（粉丝投票稳定）
    """
    controversial = detailed_uncertainty.get("controversial_contestants", [])
    stable = detailed_uncertainty.get("stable_contestants", [])

    if not controversial and not stable:
        print("没有争议型/稳态型选手数据可用。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 争议型选手（高不确定性）
    ax = axes[0]
    if controversial:
        top_controversial = controversial[:15]
        names = [
            f"S{c['season']}W{c['week']}\n{c['contestant_id'][:15]}"
            for c in top_controversial
        ]
        widths = [c["ci_width"] for c in top_controversial]

        bars = ax.barh(range(len(names)), widths, color=COLORS["warning"], alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("95% CI Width")
        ax.set_title("Controversial Contestants\n(High Uncertainty in Fan Votes)")
        ax.invert_yaxis()

        # 添加数值
        for bar, val in zip(bars, widths):
            ax.text(
                val + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=8,
            )

    # 2. 稳态型选手（低不确定性）
    ax = axes[1]
    if stable:
        top_stable = stable[:15]
        names = [
            f"S{c['season']}W{c['week']}\n{c['contestant_id'][:15]}" for c in top_stable
        ]
        widths = [c["ci_width"] for c in top_stable]

        bars = ax.barh(range(len(names)), widths, color=COLORS["success"], alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("95% CI Width")
        ax.set_title("Stable Contestants\n(Low Uncertainty in Fan Votes)")
        ax.invert_yaxis()

        # 添加数值
        for bar, val in zip(bars, widths):
            ax.text(
                val + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_finale_ranking_accuracy(
    finale_results: dict,
    output_path: Path | None = None,
):
    """
    决赛排名准确率可视化。

    决赛是验证模型硬实力的最佳场所（无评委救人）。
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 主要指标
    ax = axes[0]
    metrics = {
        "Winner\nAccuracy": finale_results.get("winner_accuracy", 0),
        "Top-3 Position\nAccuracy": finale_results.get("top3_position_accuracy", 0),
        "Mean\nKendall's τ": (finale_results.get("mean_kendall_tau", 0) + 1)
        / 2,  # 标准化到 0-1
    }

    colors = [COLORS["strict"], COLORS["bottom2"], COLORS["overall"]]
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, edgecolor="black")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Finale Ranking Performance")

    sorted_keys = sorted(metrics.keys())
    for bar, key in zip(bars, sorted_keys):
        val = metrics[key]
        if "Kendall" in key:
            # 显示原始 tau 值
            original_tau = finale_results.get("mean_kendall_tau", 0)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"τ={original_tau:.3f}",
                ha="center",
                fontsize=9,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.1%}",
                ha="center",
                fontsize=9,
            )

    # 2. 按赛季的冠军预测
    ax = axes[1]
    per_season = finale_results.get("per_season", {})
    if per_season:
        seasons = sorted([int(s) for s in per_season.keys()])
        correct = [
            1 if per_season[str(s)].get("winner_correct", False) else 0 for s in seasons
        ]

        colors = [COLORS["success"] if c else COLORS["warning"] for c in correct]
        ax.bar(seasons, [1] * len(seasons), color=colors, edgecolor="black", alpha=0.7)

        ax.set_xlabel("Season")
        ax.set_ylabel("Winner Prediction")
        ax.set_title("Finale Winner Prediction by Season")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Wrong", "Correct"])
        ax.axvline(
            27.5, color="orange", linestyle="--", linewidth=2, label="S28 Rule Change"
        )
        ax.legend()

    # 3. Kendall's Tau 分布
    ax = axes[2]
    tau_values = finale_results.get("kendall_tau_values", [])
    if tau_values:
        taus = [v["tau"] for v in tau_values]
        seasons = [v["season"] for v in tau_values]

        colors = [COLORS["pre_s28"] if s < 28 else COLORS["post_s28"] for s in seasons]
        ax.scatter(seasons, taus, c=colors, s=80, edgecolors="black", alpha=0.7)
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
        ax.axhline(
            1, color="green", linestyle="--", alpha=0.5, label="Perfect Agreement"
        )
        ax.axhline(
            -1, color="red", linestyle="--", alpha=0.5, label="Perfect Disagreement"
        )
        ax.axvline(27.5, color="orange", linestyle="--", linewidth=2)

        ax.set_xlabel("Season")
        ax.set_ylabel("Kendall's τ")
        ax.set_title("Finale Ranking Correlation by Season")
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc="lower left")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_anomaly_analysis(
    results: dict,
    output_path: Path | None = None,
):
    """可视化异常检测结果。"""
    anomalies = results.get("anomalies", {})
    n_anomalies = anomalies.get("n_anomalies", 0)
    cases = anomalies.get("cases", [])

    fig, ax = plt.subplots(figsize=(8, 5))

    if n_anomalies == 0:
        ax.text(
            0.5,
            0.5,
            "No Anomalies Detected\n\nAll eliminations are consistent\nwith the fan voting model.",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    else:
        # 按赛季绘制异常
        seasons = [c["season"] for c in cases]
        season_counts = {}
        for s in seasons:
            season_counts[s] = season_counts.get(s, 0) + 1

        ax.bar(
            season_counts.keys(),
            season_counts.values(),
            color="tomato",
            edgecolor="darkred",
        )
        ax.set_xlabel("Season")
        ax.set_ylabel("Number of Anomalies")
        ax.set_title(f"Anomalous Eliminations by Season (Total: {n_anomalies})")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_comprehensive_summary(
    results: dict,
    bottom2_results: dict | None = None,
    finale_results: dict | None = None,
    output_path: Path | None = None,
):
    """综合总结图：单页展示所有关键指标。"""
    fig = plt.figure(figsize=(16, 12))

    # 使用 GridSpec 灵活布局
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 核心验证指标 (2x1)
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = {
        "Reconstruction\nAccuracy": results["validation"]["reconstruction_accuracy"],
        "Posterior\nConsistency": results["validation"]["posterior_consistency_mean"],
        "Constraint\nSatisfaction": results["validation"][
            "constraint_satisfaction_rate"
        ],
    }

    if bottom2_results:
        metrics["Bottom-2\nRecall"] = bottom2_results["overall"]["bottom2_recall"]
    if finale_results:
        metrics["Winner\nAccuracy"] = finale_results.get("winner_accuracy", 0)

    colors = [
        COLORS["strict"],
        COLORS["success"],
        COLORS["neutral"],
        COLORS["bottom2"],
        COLORS["overall"],
    ][: len(metrics)]
    bars = ax1.bar(metrics.keys(), metrics.values(), color=colors, edgecolor="black")
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Rate")
    ax1.set_title("Core Validation Metrics", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, metrics.values()):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=10,
        )

    # 2. MCMC 诊断 (1x1)
    ax2 = fig.add_subplot(gs[0, 2])
    diag = results["diagnostics"]
    diag_metrics = {"Mean\nESS": diag["mean_ess"], "Min\nESS": diag["min_ess"]}

    colors = [COLORS["success"] if diag["min_ess"] >= 50 else COLORS["warning"]] * 2
    bars = ax2.bar(
        diag_metrics.keys(), diag_metrics.values(), color=colors, edgecolor="black"
    )
    ax2.axhline(100, color="red", linestyle="--", alpha=0.7, label="ESS=100")
    ax2.axhline(50, color="orange", linestyle="--", alpha=0.7, label="ESS=50")
    ax2.set_ylabel("ESS")
    ax2.set_title("MCMC Diagnostics", fontsize=12)
    ax2.legend(fontsize=8)

    # 3. 规则变化影响 (1x2)
    if bottom2_results:
        ax3 = fig.add_subplot(gs[1, :2])
        pre = bottom2_results["pre_s28"]
        post = bottom2_results["post_s28"]

        x = np.arange(2)
        width = 0.35

        ax3.bar(
            x - width / 2,
            [pre["strict_accuracy"], pre["bottom2_recall"]],
            width,
            label="S1-S27",
            color=COLORS["pre_s28"],
            alpha=0.8,
        )
        ax3.bar(
            x + width / 2,
            [post["strict_accuracy"], post["bottom2_recall"]],
            width,
            label="S28+",
            color=COLORS["post_s28"],
            alpha=0.8,
        )

        ax3.set_xticks(x)
        ax3.set_xticklabels(["Strict Accuracy", "Bottom-2 Recall"])
        ax3.set_ylabel("Accuracy")
        ax3.set_ylim(0, 1.1)
        ax3.set_title("Impact of Judges' Save Rule (S28+)", fontsize=12)
        ax3.legend()

    # 4. 稳定性/泛化 (1x1)
    ax4 = fig.add_subplot(gs[1, 2])
    if "stability" in results:
        stab = results["stability"]
        stab_data = {
            "Train\n(S1-20)": stab["train_accuracy"],
            "Test\n(S21+)": stab["test_accuracy"],
        }
        colors = [
            COLORS["success"],
            COLORS["warning"]
            if stab["generalization_gap"] > 0.1
            else COLORS["success"],
        ]
        bars = ax4.bar(
            stab_data.keys(), stab_data.values(), color=colors, edgecolor="black"
        )
        ax4.set_ylim(0, 1.1)
        ax4.set_ylabel("Accuracy")
        ax4.set_title(
            f"Generalization (Gap: {stab['generalization_gap']:.1%})", fontsize=12
        )

        for bar, val in zip(bars, stab_data.values()):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.1%}",
                ha="center",
                fontsize=10,
            )

    # 5. 不确定性 (1x1)
    ax5 = fig.add_subplot(gs[2, 0])
    unc = results["uncertainty"]
    ax5.bar(
        ["Mean CI\nWidth", "Median CI\nWidth"],
        [unc["mean_cri_width"], unc["median_cri_width"]],
        color=[COLORS["strict"], COLORS["bottom2"]],
        edgecolor="black",
    )
    ax5.set_ylabel("95% CI Width")
    ax5.set_title("Posterior Uncertainty", fontsize=12)

    # 6. 异常检测 (1x1)
    ax6 = fig.add_subplot(gs[2, 1])
    anomalies = results.get("anomalies", {})
    n_anomalies = anomalies.get("n_anomalies", 0)

    ax6.bar(
        ["Detected\nAnomalies"],
        [n_anomalies],
        color=COLORS["warning"] if n_anomalies > 0 else COLORS["success"],
        edgecolor="black",
    )
    ax6.set_ylabel("Count")
    ax6.set_title("Anomaly Detection", fontsize=12)
    ax6.text(
        0,
        n_anomalies + 0.1,
        str(n_anomalies),
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    plt.suptitle(
        "DWTS Fan Vote Estimation: Model Performance Summary",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_mcmc_diagnostics(results: dict, output_path: Path | None = None):
    """
    MCMC 诊断图：ESS 分布 + 接受率 + 低 ESS 赛季标注。

    展示采样质量的关键指标。
    """
    diagnostics = results.get("diagnostics", {})
    if not diagnostics:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. ESS 分布直方图
    ax = axes[0, 0]
    mean_ess = diagnostics["mean_ess"]
    min_ess = diagnostics["min_ess"]
    low_ess_seasons = diagnostics.get("low_ess_seasons", [])

    # 由于我们没有每个赛季的 ESS，这里只显示汇总统计
    ax.text(
        0.5,
        0.6,
        f"Mean ESS: {mean_ess:.0f}",
        ha="center",
        va="center",
        fontsize=14,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.4,
        f"Min ESS: {min_ess:.0f}",
        ha="center",
        va="center",
        fontsize=14,
        color="red" if min_ess < 50 else "black",
        transform=ax.transAxes,
    )
    ax.set_title("Effective Sample Size (ESS)", fontsize=12, fontweight="bold")
    ax.axis("off")

    # 2. 低 ESS 赛季列表
    ax = axes[0, 1]
    if low_ess_seasons:
        ax.text(
            0.5,
            0.7,
            "Low ESS Seasons (<50):",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            transform=ax.transAxes,
        )
        season_text = ", ".join([f"S{s}" for s in low_ess_seasons])
        ax.text(
            0.5,
            0.4,
            season_text,
            ha="center",
            va="center",
            fontsize=11,
            color="red",
            transform=ax.transAxes,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "All seasons have adequate ESS",
            ha="center",
            va="center",
            fontsize=12,
            color="green",
            transform=ax.transAxes,
        )
    ax.set_title("ESS Quality Check", fontsize=12, fontweight="bold")
    ax.axis("off")

    # 3. 接受率
    ax = axes[1, 0]
    mean_acc = diagnostics["mean_acceptance_rate"]
    target = 0.35  # 目标接受率

    ax.barh(
        ["Mean Acceptance", "Target"],
        [mean_acc, target],
        color=[
            COLORS["success"] if abs(mean_acc - target) < 0.1 else COLORS["warning"],
            COLORS["neutral"],
        ],
        edgecolor="black",
    )
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Acceptance Rate")
    ax.set_title("MCMC Acceptance Rate", fontsize=12, fontweight="bold")
    ax.axvline(target, color="red", linestyle="--", linewidth=2, label="Target (0.35)")
    ax.legend()

    # 4. 多峰分布警告
    ax = axes[1, 1]
    multimodal = diagnostics.get("multimodal_seasons", [])
    if multimodal:
        ax.text(
            0.5,
            0.6,
            "⚠ Multimodal Distribution Suspected:",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="red",
            transform=ax.transAxes,
        )
        season_text = ", ".join([f"S{s}" for s in multimodal])
        ax.text(
            0.5,
            0.4,
            season_text,
            ha="center",
            va="center",
            fontsize=11,
            color="red",
            transform=ax.transAxes,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "✓ No multimodal distributions detected",
            ha="center",
            va="center",
            fontsize=12,
            color="green",
            transform=ax.transAxes,
        )
    ax.set_title("Distribution Check", fontsize=12, fontweight="bold")
    ax.axis("off")

    plt.suptitle("MCMC Sampling Diagnostics", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_performance_heatmap(
    recon_details: dict, bottom2_results: dict, output_path: Path | None = None
):
    """
    赛季级别性能热图：展示每个赛季的 Strict Accuracy 和 Bottom-2 Recall。
    """
    per_season = recon_details.get("per_season", {})
    b2_per_season = bottom2_results.get("per_season", {})

    if not per_season:
        return

    seasons = sorted([int(s) for s in per_season.keys()])
    strict_acc = [per_season[str(s)]["accuracy"] for s in seasons]
    bottom2_rec = [
        b2_per_season.get(str(s), {}).get("bottom2_recall", 0) for s in seasons
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Strict Accuracy 热图
    ax = axes[0]
    colors_strict = [
        COLORS["success"]
        if a >= 0.9
        else COLORS["warning"]
        if a >= 0.7
        else COLORS["neutral"]
        for a in strict_acc
    ]
    bars = ax.bar(
        seasons, strict_acc, color=colors_strict, edgecolor="black", alpha=0.8
    )
    ax.axhline(0.95, color="green", linestyle="--", linewidth=1.5, label="Target (95%)")
    ax.set_ylabel("Strict Accuracy", fontsize=11)
    ax.set_title(
        "Strict Elimination Accuracy by Season", fontsize=12, fontweight="bold"
    )
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Bottom-2 Recall 热图
    ax = axes[1]
    colors_b2 = [
        COLORS["success"]
        if r >= 0.95
        else COLORS["warning"]
        if r >= 0.85
        else COLORS["neutral"]
        for r in bottom2_rec
    ]
    bars = ax.bar(seasons, bottom2_rec, color=colors_b2, edgecolor="black", alpha=0.8)
    ax.axhline(0.95, color="green", linestyle="--", linewidth=1.5, label="Target (95%)")
    ax.axvline(
        27.5, color="orange", linestyle=":", linewidth=2, label="Rule Change (S28)"
    )
    ax.set_xlabel("Season", fontsize=11)
    ax.set_ylabel("Bottom-2 Recall", fontsize=11)
    ax.set_title(
        "Bottom-2 Recall by Season (Relevant for S28+)", fontsize=12, fontweight="bold"
    )
    ax.set_ylim(0, 1.1)
    ax.set_xticks(seasons[::2])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Season-Level Performance Heatmap", fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_uncertainty_decomposition(
    detailed_uncertainty: dict, output_path: Path | None = None
):
    """
    不确定性来源分解：展示不同因素对不确定性的贡献。
    """
    by_industry = detailed_uncertainty.get("by_industry", {})
    by_week = detailed_uncertainty.get("by_week_position", {})

    if not by_industry and not by_week:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 行业贡献（Top 10）
    ax = axes[0]
    if by_industry:
        industries = sorted(
            by_industry.keys(),
            key=lambda x: by_industry[x]["mean_ci_width"],
            reverse=True,
        )[:10]
        widths = [by_industry[i]["mean_ci_width"] for i in industries]

        y_pos = np.arange(len(industries))
        bars = ax.barh(
            y_pos, widths, color=COLORS["strict"], edgecolor="black", alpha=0.8
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([i[:20] for i in industries], fontsize=9)
        ax.set_xlabel("Mean 95% CI Width", fontsize=11)
        ax.set_title(
            "Uncertainty by Celebrity Industry", fontsize=12, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)

    # 2. 周次位置贡献
    ax = axes[1]
    if by_week:
        positions = []
        widths = []
        for pos, data in by_week.items():
            positions.append(pos.capitalize())
            widths.append(data["mean_ci_width"])

        colors = [COLORS["pre_s28"], COLORS["post_s28"]][: len(positions)]
        bars = ax.bar(positions, widths, color=colors, edgecolor="black", alpha=0.8)
        ax.set_ylabel("Mean 95% CI Width", fontsize=11)
        ax.set_title("Uncertainty by Week Position", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, width in zip(bars, widths):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                width + 0.005,
                f"{width:.3f}",
                ha="center",
                fontsize=10,
            )

    plt.suptitle(
        "Uncertainty Source Decomposition", fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_temporal_trends(df_fan_shares: pl.DataFrame, output_path: Path | None = None):
    """
    时间序列趋势分析：展示粉丝份额随赛季和周次的变化。
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 每周平均粉丝份额（所有赛季）
    ax = axes[0, 0]
    weekly_avg = (
        df_fan_shares
        .group_by("week")
        .agg(pl.col("estimated_fan_share").mean().alias("mean_share"))
        .sort("week")
    )
    weeks = weekly_avg["week"].to_numpy()
    means = weekly_avg["mean_share"].to_numpy()

    ax.plot(weeks, means, marker="o", linewidth=2, color=COLORS["strict"])
    ax.fill_between(weeks, means * 0.9, means * 1.1, alpha=0.2, color=COLORS["strict"])
    ax.set_xlabel("Week", fontsize=11)
    ax.set_ylabel("Average Fan Share", fontsize=11)
    ax.set_title(
        "Average Fan Share by Week (All Seasons)", fontsize=12, fontweight="bold"
    )
    ax.grid(alpha=0.3)

    # 2. 赛季趋势：中位数粉丝份额
    ax = axes[0, 1]
    seasonal_median = (
        df_fan_shares
        .group_by("season")
        .agg(pl.col("estimated_fan_share").median().alias("median_share"))
        .sort("season")
    )
    seasons = seasonal_median["season"].to_numpy()
    medians = seasonal_median["median_share"].to_numpy()

    ax.plot(seasons, medians, marker="s", linewidth=2, color=COLORS["post_s28"])
    ax.axvline(
        27.5, color="orange", linestyle=":", linewidth=2, label="Rule Change (S28)"
    )
    ax.set_xlabel("Season", fontsize=11)
    ax.set_ylabel("Median Fan Share", fontsize=11)
    ax.set_title("Median Fan Share by Season", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. 淘汰者 vs 存活者的份额分布随时间变化
    ax = axes[1, 0]
    eliminated_by_season = (
        df_fan_shares
        .filter(pl.col("is_eliminated"))
        .group_by("season")
        .agg(pl.col("estimated_fan_share").mean().alias("eliminated_share"))
        .sort("season")
    )
    survived_by_season = (
        df_fan_shares
        .filter(~pl.col("is_eliminated"))
        .group_by("season")
        .agg(pl.col("estimated_fan_share").mean().alias("survived_share"))
        .sort("season")
    )

    seasons_e = eliminated_by_season["season"].to_numpy()
    elim_shares = eliminated_by_season["eliminated_share"].to_numpy()
    seasons_s = survived_by_season["season"].to_numpy()
    surv_shares = survived_by_season["survived_share"].to_numpy()

    ax.plot(
        seasons_e,
        elim_shares,
        marker="v",
        label="Eliminated",
        linewidth=2,
        color=COLORS["warning"],
    )
    ax.plot(
        seasons_s,
        surv_shares,
        marker="^",
        label="Survived",
        linewidth=2,
        color=COLORS["success"],
    )
    ax.axvline(27.5, color="orange", linestyle=":", linewidth=2, alpha=0.5)
    ax.set_xlabel("Season", fontsize=11)
    ax.set_ylabel("Average Fan Share", fontsize=11)
    ax.set_title(
        "Eliminated vs Survived Fan Shares Over Time", fontsize=12, fontweight="bold"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. 方差随赛季变化
    ax = axes[1, 1]
    seasonal_var = (
        df_fan_shares
        .group_by("season")
        .agg(pl.col("estimated_fan_share").var().alias("variance"))
        .sort("season")
    )
    seasons_v = seasonal_var["season"].to_numpy()
    variances = seasonal_var["variance"].to_numpy()

    ax.plot(seasons_v, variances, marker="D", linewidth=2, color=COLORS["bottom2"])
    ax.axvline(
        27.5, color="orange", linestyle=":", linewidth=2, label="Rule Change (S28)"
    )
    ax.set_xlabel("Season", fontsize=11)
    ax.set_ylabel("Variance of Fan Shares", fontsize=11)
    ax.set_title("Fan Share Variance by Season", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(
        "Temporal Trends in Fan Vote Patterns", fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_validation_radar(results: dict, output_path: Path | None = None):
    """
    验证指标雷达图：多维度展示模型性能。
    """
    validation = results.get("validation", {})
    if not validation:
        return

    # 准备数据
    categories = [
        "Reconstruction\nAccuracy",
        "Posterior\nConsistency",
        "Constraint\nSatisfaction",
        "Bottom-2\nRecall",
        "Finale\nAccuracy",
    ]

    values = [
        validation.get("reconstruction_accuracy", 0),
        validation.get("posterior_consistency_mean", 0),
        validation.get("constraint_satisfaction_rate", 0),
        validation.get("bottom2_overall_recall", 0),
        validation.get("finale_winner_accuracy", 0),
    ]

    # 添加第一个值到末尾以闭合雷达图
    values += values[:1]

    # 角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    # 绘制雷达图
    ax.plot(
        angles,
        values,
        "o-",
        linewidth=2,
        color=COLORS["strict"],
        label="Model Performance",
    )
    ax.fill(angles, values, alpha=0.25, color=COLORS["strict"])

    # 添加目标线（95%）
    target = [0.95] * len(angles)
    ax.plot(
        angles, target, "--", linewidth=2, color=COLORS["success"], label="Target (95%)"
    )

    # 设置刻度
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=9)
    ax.grid(True, alpha=0.3)

    # 添加值标签
    for angle, value, cat in zip(angles[:-1], values[:-1], categories):
        ax.text(
            angle,
            value + 0.05,
            f"{value:.1%}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=COLORS["strict"],
        )

    plt.title(
        "Model Validation Metrics (Radar Chart)",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    plt.close()


def main():
    """生成所有可视化图表。"""
    print("=" * 70)
    print("生成可视化图表")
    print("=" * 70)

    # 加载结果
    results_path = OUTPUT_DIR / "results.json"
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run run_inference.py first.")
        return

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    # 加载重构详情
    recon_path = OUTPUT_DIR / "reconstruction_details.json"
    recon_details = {}
    if recon_path.exists():
        with open(recon_path, encoding="utf-8") as f:
            recon_details = json.load(f)

    # 加载粉丝份额
    fan_shares_path = OUTPUT_DIR / "fan_shares.csv"
    df_fan_shares = None
    if fan_shares_path.exists():
        df_fan_shares = pl.read_csv(fan_shares_path)

    # 加载新增结果文件（如果存在）
    bottom2_path = OUTPUT_DIR / "bottom2_accuracy.json"
    bottom2_results = None
    if bottom2_path.exists():
        with open(bottom2_path, encoding="utf-8") as f:
            bottom2_results = json.load(f)

    finale_path = OUTPUT_DIR / "finale_ranking.json"
    finale_results = None
    if finale_path.exists():
        with open(finale_path, encoding="utf-8") as f:
            finale_results = json.load(f)

    detailed_unc_path = OUTPUT_DIR / "detailed_uncertainty.json"
    detailed_uncertainty = None
    if detailed_unc_path.exists():
        with open(detailed_unc_path, encoding="utf-8") as f:
            detailed_uncertainty = json.load(f)

    # 生成图表
    print("\n生成图表...")

    # 1. Validation Summary
    plot_validation_summary(results, FIGURE_DIR / "validation_summary.png")

    # 2. Reconstruction Accuracy by Season (简单版)
    if recon_details:
        plot_reconstruction_accuracy_by_season(
            results, recon_details, FIGURE_DIR / "reconstruction_by_season.png"
        )

    # 3. Fan Share Distribution
    if df_fan_shares is not None:
        plot_fan_share_distribution(
            df_fan_shares, FIGURE_DIR / "fan_share_distribution.png"
        )

    # 4. Credible Intervals (简单版)
    plot_credible_interval_distribution(results, FIGURE_DIR / "credible_intervals.png")

    # 5. Anomaly Analysis
    plot_anomaly_analysis(results, FIGURE_DIR / "anomaly_analysis.png")

    # ========== 新增图表 ==========

    # 6. 堆叠柱状图：Strict vs Bottom-2
    if bottom2_results:
        plot_stacked_accuracy_by_season(
            bottom2_results, FIGURE_DIR / "stacked_accuracy_by_season.png"
        )

    # 7. 结构漂移分析
    if bottom2_results:
        stability_results = results.get("stability", None)
        plot_structural_drift_analysis(
            bottom2_results,
            stability_results,
            FIGURE_DIR / "structural_drift_analysis.png",
        )

    # 8. 详细 CI 分布
    if detailed_uncertainty:
        plot_detailed_ci_distribution(
            detailed_uncertainty, FIGURE_DIR / "detailed_ci_distribution.png"
        )
        plot_controversial_vs_stable(
            detailed_uncertainty, FIGURE_DIR / "controversial_vs_stable.png"
        )

    # 9. 决赛排名准确率
    if finale_results:
        plot_finale_ranking_accuracy(
            finale_results, FIGURE_DIR / "finale_ranking_accuracy.png"
        )

    # 10. 综合总结图
    plot_comprehensive_summary(
        results,
        bottom2_results,
        finale_results,
        FIGURE_DIR / "comprehensive_summary.png",
    )

    # ========== 新增科研级别图表 ==========

    # 11. MCMC 诊断图（迭代轨迹 + ESS 分布）
    plot_mcmc_diagnostics(results, FIGURE_DIR / "mcmc_diagnostics.png")

    # 12. 赛季级别性能热图
    if recon_details and bottom2_results:
        plot_performance_heatmap(
            recon_details, bottom2_results, FIGURE_DIR / "performance_heatmap.png"
        )

    # 13. 不确定性来源分解
    if detailed_uncertainty:
        plot_uncertainty_decomposition(
            detailed_uncertainty, FIGURE_DIR / "uncertainty_decomposition.png"
        )

    # 14. 时间序列趋势分析
    if df_fan_shares is not None:
        plot_temporal_trends(df_fan_shares, FIGURE_DIR / "temporal_trends.png")

    # 15. 验证指标雷达图
    plot_validation_radar(results, FIGURE_DIR / "validation_radar.png")

    print("\n" + "=" * 70)
    print(f"所有图表已保存到: {FIGURE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
