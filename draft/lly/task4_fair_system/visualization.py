"""
Task 4 可视化模块。

与 Task1/Task3 保持一致的科研风格。
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation import SystemMetrics

# 设置随机种子
np.random.seed(42)
random.seed(42)

# 论文发表的样式设置 - 与 task1/task3 一致
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

# 配色方案 - 与 task1/task3 一致
COLORS = {
    "rank": "#C73E1D",  # 红色 - Rank-based
    "pct": "#2E86AB",  # 深蓝 - Percentage-based
    "ptfs": "#2ECC71",  # 绿色 - PTFS
    "neutral": "#95A5A6",  # 灰色
    "highlight": "#F18F01",  # 橙色 - 高亮
}


def plot_tau_by_season(
    systems_metrics: list[SystemMetrics],
    output_path: Path,
) -> None:
    """绘制各赛季 Kendall's τ 对比（核心图表）。

    展示 PTFS 在大多数赛季的改进。
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = [COLORS["rank"], COLORS["pct"], COLORS["ptfs"]]
    markers = ["^", "s", "o"]
    linestyles = ["--", "-.", "-"]

    for i, metrics in enumerate(systems_metrics):
        seasons = [s["season"] for s in metrics.season_details]
        taus = [
            s["kendall_tau"] if s["kendall_tau"] is not None else np.nan
            for s in metrics.season_details
        ]

        mean_tau = np.nanmean(taus)
        ax.plot(
            seasons,
            taus,
            marker=markers[i],
            linestyle=linestyles[i],
            color=colors[i],
            label=f"{metrics.system_name} (μ={mean_tau:.3f})",
            linewidth=1.5,
            markersize=4,
            alpha=0.85,
        )

    # 参考线
    ax.axhline(
        0.5,
        color=COLORS["neutral"],
        linestyle=":",
        alpha=0.6,
        label="τ = 0.5 threshold",
    )

    # S28 规则变化标记
    ax.axvline(
        27.5,
        color=COLORS["highlight"],
        linestyle="--",
        alpha=0.7,
        label="Rule Change (S28)",
    )

    ax.set_xlabel("Season")
    ax.set_ylabel("Kendall's τ (Technical-Final Correlation)")
    ax.set_title("Technical Fairness Across Seasons: PTFS Consistently Outperforms")
    ax.legend(loc="lower left", ncol=2)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlim(0.5, 34.5)
    ax.set_xticks(range(1, 35, 2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_improvement_distribution(
    systems_metrics: list[SystemMetrics],
    output_path: Path,
) -> None:
    """绘制 PTFS 相对于 Percentage 的改进分布。

    展示多少赛季有改进、改进幅度分布。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 假设顺序是 [rank, pct, ptfs]
    if len(systems_metrics) < 3:
        plt.close()
        return

    pct_taus = {
        s["season"]: s["kendall_tau"]
        for s in systems_metrics[1].season_details
        if s["kendall_tau"] is not None
    }
    ptfs_taus = {
        s["season"]: s["kendall_tau"]
        for s in systems_metrics[2].season_details
        if s["kendall_tau"] is not None
    }

    improvements = []
    seasons_improved = 0
    seasons_same = 0
    seasons_decreased = 0
    seasons_total = 0
    threshold = 0.001  # 用于判断是否相同的阈值

    for season in pct_taus:
        if season in ptfs_taus:
            diff = ptfs_taus[season] - pct_taus[season]
            improvements.append(diff)
            seasons_total += 1
            if diff > threshold:
                seasons_improved += 1
            elif diff < -threshold:
                seasons_decreased += 1
            else:
                seasons_same += 1

    # 左图：改进分布直方图
    ax1 = axes[0]
    bins = np.linspace(-0.15, 0.35, 26)
    n, bins_out, patches = ax1.hist(
        improvements, bins=bins, color=COLORS["ptfs"], edgecolor="white", alpha=0.8
    )

    # 标记正负区域 - 使用 bin 中心点判断，与右图饼图颜色一致
    for patch, left_edge, right_edge in zip(patches, bins_out[:-1], bins_out[1:]):
        center = (left_edge + right_edge) / 2
        if center < -threshold:
            patch.set_facecolor(COLORS["rank"])  # 红色 - 减小
        elif center > threshold:
            patch.set_facecolor(COLORS["ptfs"])  # 绿色 - 改进
        else:
            patch.set_facecolor(COLORS["neutral"])  # 灰色 - 不变

    ax1.axvline(0, color="black", linestyle="-", linewidth=2)
    ax1.axvline(
        np.mean(improvements),
        color=COLORS["highlight"],
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(improvements):+.4f}",
    )

    ax1.set_xlabel("Δτ (PTFS - Percentage)")
    ax1.set_ylabel("Number of Seasons")
    ax1.set_title("Distribution of τ Improvement by Season")
    ax1.legend()

    # 右图：三类比例饼图
    ax2 = axes[1]
    labels = ["Improved\n(PTFS > Pct)", "Same\n(PTFS ≈ Pct)", "Decreased\n(PTFS < Pct)"]
    sizes = [seasons_improved, seasons_same, seasons_decreased]
    colors_pie = [COLORS["ptfs"], COLORS["neutral"], COLORS["rank"]]

    # 过滤掉 0 值
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors_pie) if s > 0]
    if non_zero:
        labels, sizes, colors_pie = zip(*non_zero)

    explode = [0.05 if i == 0 else 0 for i in range(len(sizes))]

    wedges, texts, autotexts = ax2.pie(
        sizes,
        labels=labels,
        colors=colors_pie,
        autopct="%1.0f%%",
        startangle=90,
        explode=explode,
        textprops={"fontsize": 11},
    )
    if autotexts:
        autotexts[0].set_fontweight("bold")

    # 更准确的标题：显示在有差异的赛季中的改进率
    seasons_with_diff = seasons_improved + seasons_decreased
    if seasons_with_diff > 0:
        improvement_rate = seasons_improved / seasons_with_diff
        ax2.set_title(
            f"Season-Level Comparison\n"
            f"({seasons_improved} improved, {seasons_same} same, {seasons_decreased} decreased)"
        )
    else:
        ax2.set_title(f"Season-Level Comparison\n(All {seasons_total} seasons same)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison_bar(
    systems_metrics: list[SystemMetrics],
    output_path: Path,
) -> None:
    """绘制关键指标对比条形图。"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    metrics_config = [
        ("Kendall's τ", lambda m: m.technical_fairness.kendall_tau, "higher"),
        (
            "Tech Top3 → Final Top3",
            lambda m: m.technical_fairness.tech_top3_in_final_top3,
            "higher",
        ),
        (
            "Mean Rank Deviation",
            lambda m: m.technical_fairness.mean_rank_deviation,
            "lower",
        ),
        (
            "Consistent Seasons (τ>0.5)",
            lambda m: m.robustness.consistent_seasons_rate,
            "higher",
        ),
        (
            "τ Stability (1-Std)",
            lambda m: 1 - m.robustness.tau_std_across_seasons,
            "higher",
        ),
        ("Tech Winner Rate", lambda m: m.technical_fairness.tech_winner_rate, "higher"),
    ]

    colors = [COLORS["rank"], COLORS["pct"], COLORS["ptfs"]]
    system_names = [m.system_name for m in systems_metrics]

    for idx, (title, getter, better) in enumerate(metrics_config):
        ax = axes[idx // 3, idx % 3]
        values = [getter(m) for m in systems_metrics]
        x = np.arange(len(system_names))

        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=1.5)

        # 标注最佳
        if better == "higher":
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)

        bars[best_idx].set_edgecolor(COLORS["highlight"])
        bars[best_idx].set_linewidth(3)

        # 数值标签
        for j, (bar, val) in enumerate(zip(bars, values)):
            if "Rate" in title or "Top3" in title or "Consistent" in title:
                label = f"{val:.1%}"
            else:
                label = f"{val:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if j == best_idx else "normal",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(system_names, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # 方向标注
        direction = "↑ higher better" if better == "higher" else "↓ lower better"
        ax.text(
            0.98,
            0.02,
            direction,
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            alpha=0.6,
        )

    plt.suptitle(
        "System Comparison: Key Metrics\n(Gold border = Best in category)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dynamic_weights(
    params,
    max_week: int,
    output_path: Path,
) -> None:
    """绘制 PTFS 动态权重曲线。"""
    fig, ax = plt.subplots(figsize=(10, 5))

    weeks = np.arange(1, max_week + 1)
    judge_weights = []
    fan_weights = []

    for week in weeks:
        if max_week <= 1:
            w_judge = (params.judge_weight_start + params.judge_weight_end) / 2
        else:
            progress = (week - 1) / (max_week - 1)
            w_judge = (
                params.judge_weight_start
                + (params.judge_weight_end - params.judge_weight_start) * progress
            )
        judge_weights.append(w_judge)
        fan_weights.append(1 - w_judge)

    ax.fill_between(
        weeks, 0, judge_weights, alpha=0.4, color=COLORS["pct"], label="Judge Weight"
    )
    ax.fill_between(
        weeks, judge_weights, 1, alpha=0.4, color=COLORS["rank"], label="Fan Weight"
    )
    ax.plot(weeks, judge_weights, "o-", color=COLORS["pct"], linewidth=2, markersize=6)
    ax.plot(weeks, fan_weights, "s-", color=COLORS["rank"], linewidth=2, markersize=6)

    ax.set_xlabel("Week")
    ax.set_ylabel("Weight")
    ax.set_title(
        f"PTFS Dynamic Weights: Progressive Technical Emphasis\n"
        f"Judge: {params.judge_weight_start:.0%} → {params.judge_weight_end:.0%}"
    )
    ax.legend(loc="center right")
    ax.set_xlim(1, max_week)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    # 阶段标注
    ax.text(
        2,
        0.15,
        "Early:\nFan-focused\n(Engagement)",
        ha="center",
        fontsize=9,
        style="italic",
    )
    ax.text(
        max_week - 1,
        0.15,
        "Late:\nTech-focused\n(Fairness)",
        ha="center",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rank_deviation_boxplot(
    systems_metrics: list[SystemMetrics],
    output_path: Path,
) -> None:
    """绘制排名偏差分布箱线图。"""
    fig, ax = plt.subplots(figsize=(8, 6))

    data = []
    labels = []
    colors = [COLORS["rank"], COLORS["pct"], COLORS["ptfs"]]

    for i, metrics in enumerate(systems_metrics):
        # 使用赛季级别的 mean_deviation 数据
        devs = [
            s["mean_deviation"] for s in metrics.season_details if s["mean_deviation"]
        ]
        data.append(devs)
        labels.append(metrics.system_name)

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Mean Rank Deviation per Season")
    ax.set_title("Distribution of Rank Deviations Across Seasons\n(Lower = More Fair)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_statistical_summary(
    systems_metrics: list[SystemMetrics],
    output_path: Path,
) -> None:
    """绘制统计显著性汇总表格。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    if len(systems_metrics) < 3:
        plt.close()
        return

    rank_m, pct_m, ptfs_m = systems_metrics[0], systems_metrics[1], systems_metrics[2]

    headers = ["Metric", "Rank-based", "Percentage", "PTFS", "Δ (PTFS-Pct)", "p-value"]
    rows = [
        [
            "Kendall's τ",
            f"{rank_m.technical_fairness.kendall_tau:.4f}",
            f"{pct_m.technical_fairness.kendall_tau:.4f}",
            f"{ptfs_m.technical_fairness.kendall_tau:.4f}",
            f"{ptfs_m.tradeoff.tau_vs_pct:+.4f}" if ptfs_m.tradeoff else "-",
            f"{ptfs_m.tradeoff.tau_diff_p_value:.4f}*"
            if ptfs_m.tradeoff and ptfs_m.tradeoff.tau_diff_significant
            else f"{ptfs_m.tradeoff.tau_diff_p_value:.4f}"
            if ptfs_m.tradeoff
            else "-",
        ],
        [
            "Tech Top3 → Final Top3",
            f"{rank_m.technical_fairness.tech_top3_in_final_top3:.1%}",
            f"{pct_m.technical_fairness.tech_top3_in_final_top3:.1%}",
            f"{ptfs_m.technical_fairness.tech_top3_in_final_top3:.1%}",
            f"{ptfs_m.tradeoff.top3_retention_vs_pct:+.1%}" if ptfs_m.tradeoff else "-",
            "-",
        ],
        [
            "Mean Rank Deviation",
            f"{rank_m.technical_fairness.mean_rank_deviation:.2f}",
            f"{pct_m.technical_fairness.mean_rank_deviation:.2f}",
            f"{ptfs_m.technical_fairness.mean_rank_deviation:.2f}",
            f"{ptfs_m.technical_fairness.mean_rank_deviation - pct_m.technical_fairness.mean_rank_deviation:+.2f}",
            "-",
        ],
        [
            "Consistent Seasons (τ>0.5)",
            f"{rank_m.robustness.consistent_seasons_rate:.1%}",
            f"{pct_m.robustness.consistent_seasons_rate:.1%}",
            f"{ptfs_m.robustness.consistent_seasons_rate:.1%}",
            f"{(ptfs_m.robustness.consistent_seasons_rate - pct_m.robustness.consistent_seasons_rate):+.1%}",
            "-",
        ],
        [
            "Tech Winner Rate",
            f"{rank_m.technical_fairness.tech_winner_rate:.1%}",
            f"{pct_m.technical_fairness.tech_winner_rate:.1%}",
            f"{ptfs_m.technical_fairness.tech_winner_rate:.1%}",
            f"{(ptfs_m.technical_fairness.tech_winner_rate - pct_m.technical_fairness.tech_winner_rate):+.1%}",
            "-",
        ],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # 表头样式
    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#34495E")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # PTFS 列高亮
    for i in range(1, len(rows) + 1):
        table[(i, 3)].set_facecolor("#E8F8F5")

    ax.set_title(
        "Statistical Comparison Summary\n(* indicates p < 0.05)",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_engagement_comparison(
    systems_metrics: list[SystemMetrics],
    output_path: Path,
) -> None:
    """绘制参与度指标对比。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    colors = [COLORS["rank"], COLORS["pct"], COLORS["ptfs"]]
    system_names = [m.system_name for m in systems_metrics]
    x = np.arange(len(system_names))

    # Tech Top50 被淘汰率（爆冷）
    ax1 = axes[0]
    values = [m.engagement.tech_top50_eliminated_rate for m in systems_metrics]
    bars = ax1.bar(x, values, color=colors, edgecolor="white")
    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.1%}",
            ha="center",
            fontsize=10,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(system_names)
    ax1.set_ylabel("Rate")
    ax1.set_title("Tech Top 50% Eliminated\n(Upset Rate)")

    # 技术最低被淘汰率
    ax2 = axes[1]
    values = [m.engagement.tech_bottom_eliminated_rate for m in systems_metrics]
    bars = ax2.bar(x, values, color=colors, edgecolor="white")
    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.1%}",
            ha="center",
            fontsize=10,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(system_names)
    ax2.set_ylabel("Rate")
    ax2.set_title("Tech Lowest Eliminated\n(Expected Outcome)")

    # Close Call 率
    ax3 = axes[2]
    values = [m.engagement.close_call_rate for m in systems_metrics]
    bars = ax3.bar(x, values, color=colors, edgecolor="white")
    for bar, val in zip(bars, values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.1%}",
            ha="center",
            fontsize=10,
        )
    ax3.set_xticks(x)
    ax3.set_xticklabels(system_names)
    ax3.set_ylabel("Rate")
    ax3.set_title("Close Calls (<2% gap)\n(Suspense)")

    plt.suptitle("Engagement Metrics Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_plots(
    systems_metrics: list[SystemMetrics],
    sensitivity_analysis: dict,
    ptfs_params,
    figures_dir: Path,
    seasons_data: dict = None,
    results_dict: dict = None,
) -> None:
    """生成所有图表。"""
    figures_dir.mkdir(exist_ok=True)

    print("  [1/6] 赛季 τ 对比图...")
    plot_tau_by_season(systems_metrics, figures_dir / "tau_by_season.png")

    print("  [2/6] 改进分布图...")
    plot_improvement_distribution(
        systems_metrics, figures_dir / "improvement_distribution.png"
    )

    print("  [3/6] 关键指标对比...")
    plot_metrics_comparison_bar(systems_metrics, figures_dir / "metrics_comparison.png")

    print("  [4/6] 动态权重曲线...")
    plot_dynamic_weights(ptfs_params, 11, figures_dir / "dynamic_weights.png")

    print("  [5/6] 排名偏差分布...")
    plot_rank_deviation_boxplot(
        systems_metrics, figures_dir / "rank_deviation_boxplot.png"
    )

    print("  [6/6] 统计汇总表...")
    plot_statistical_summary(systems_metrics, figures_dir / "statistical_summary.png")

    print("  [Bonus] 参与度对比...")
    plot_engagement_comparison(
        systems_metrics, figures_dir / "engagement_comparison.png"
    )
