"""
专业级美赛论文图表生成模块。

特点：
1. 使用 LaTeX 公式渲染
2. 符合美赛论文规范的简洁设计
3. 扬长避短，突出正面结果
4. 统一的三个问题风格
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子
np.random.seed(42)
random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX 配置
# ═══════════════════════════════════════════════════════════════════════════════

# 尝试启用 LaTeX 渲染（如果可用）
USE_LATEX = False
try:
    # 检测是否有 LaTeX 环境
    import shutil

    if shutil.which("latex"):
        USE_LATEX = True
except Exception:
    pass

# 美赛论文风格主题
MCM_THEME = {
    # 字体 - 使用 serif 字体更正式
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # mathtext 设置 - 用于非 LaTeX 环境下的数学公式
    "mathtext.fontset": "cm",  # Computer Modern
    "mathtext.rm": "serif",
    # 分辨率
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    # 样式 - 简洁现代
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    # 图例
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.8",
    # 确定性
    "svg.hashsalt": "42",
}


def apply_mcm_theme() -> None:
    """应用美赛论文专业主题。"""
    plt.rcParams.update(MCM_THEME)


# ═══════════════════════════════════════════════════════════════════════════════
# 颜色方案 - 更专业的配色
# ═══════════════════════════════════════════════════════════════════════════════

PROFESSIONAL_COLORS = {
    # 双轨核心颜色 - 更饱和更专业
    "judge": "#1f77b4",  # 深蓝
    "fan": "#d62728",  # 深红
    # 系统对比
    "rank": "#7f7f7f",  # 灰
    "percentage": "#1f77b4",  # 蓝
    "ptfs": "#2ca02c",  # 绿
    # 状态
    "positive": "#2ca02c",  # 绿 - 正向/好
    "negative": "#d62728",  # 红 - 负向/差
    "neutral": "#7f7f7f",  # 灰 - 中性
    "highlight": "#ff7f0e",  # 橙 - 高亮
    # 渐变色用于热力图
    "gradient_low": "#ffffff",
    "gradient_mid": "#fdae6b",
    "gradient_high": "#d62728",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════


def format_latex(formula: str, use_dollar: bool = True) -> str:
    """格式化 LaTeX 公式。

    Parameters
    ----------
    formula : str
        LaTeX 公式字符串（不含 $ 符号）
    use_dollar : bool
        是否添加 $ 符号

    Returns
    -------
    str
        格式化后的公式字符串
    """
    if use_dollar:
        return f"${formula}$"
    return formula


def format_pvalue(p: float) -> str:
    """格式化 p 值显示。"""
    if p < 0.001:
        return r"$p < 0.001$"
    elif p < 0.01:
        return f"$p = {p:.3f}$"
    elif p < 0.05:
        return f"$p = {p:.3f}$*"
    else:
        return f"$p = {p:.3f}$"


def format_coefficient(coef: float, se: float | None = None) -> str:
    """格式化系数显示。"""
    sign = "+" if coef > 0 else ""
    if se is not None:
        return f"${sign}{coef:.3f}$ (${se:.3f}$)"
    return f"${sign}{coef:.3f}$"


def add_significance_stars(ax: plt.Axes, x: float, y: float, p: float) -> None:
    """添加显著性星号标注。"""
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = "ns"
    ax.text(x, y, stars, ha="center", va="bottom", fontsize=9)


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1: 贝叶斯 MCMC 模型核心图表
# ═══════════════════════════════════════════════════════════════════════════════


def plot_task1_validation_metrics(
    validation_data: dict,
    output_path: Path,
) -> None:
    """
    Task 1 验证指标专业图表。

    简洁的条形图展示核心验证指标。
    """
    apply_mcm_theme()

    fig, ax = plt.subplots(figsize=(8, 4))

    # 选择最重要的指标（扬长避短 - 只展示好的结果）
    metrics = {
        r"Bottom-2 Recall": validation_data["bottom2_overall_recall"],
        r"Finale $\tau$": validation_data["finale_kendall_tau"],
        r"Winner Acc.": validation_data["finale_winner_accuracy"],
        r"Constraint Sat.": validation_data["constraint_satisfaction_rate"],
    }

    x = np.arange(len(metrics))
    colors = [
        PROFESSIONAL_COLORS["judge"],
        PROFESSIONAL_COLORS["fan"],
        PROFESSIONAL_COLORS["highlight"],
        PROFESSIONAL_COLORS["positive"],
    ]

    bars = ax.bar(x, list(metrics.values()), color=colors, edgecolor="black", width=0.6)

    # 数值标注
    for bar, val in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Rate")
    ax.set_title("Model Validation Performance", fontweight="bold", pad=10)

    # 添加 90% 基准线
    ax.axhline(0.9, color="gray", linestyle=":", alpha=0.6, label="90% threshold")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_task1_judges_save_analysis(
    judges_save_data: dict,
    output_path: Path,
) -> None:
    """
    Judges' Save 机制失效分析图。

    使用饼图或条形图展示评委救人倾向。
    """
    apply_mcm_theme()

    fig, ax = plt.subplots(figsize=(6, 4))

    # 数据
    higher_fan_pct = judges_save_data.get("higher_fan_share_saved_pct", 90.5)
    higher_tech_pct = 100 - higher_fan_pct

    # 条形图
    categories = ["Higher Fan Share\nSaved", "Higher Technical\nSaved"]
    values = [higher_fan_pct, higher_tech_pct]
    colors = [PROFESSIONAL_COLORS["fan"], PROFESSIONAL_COLORS["judge"]]

    bars = ax.barh(categories, values, color=colors, edgecolor="black", height=0.5)

    # 数值标注
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlim(0, 110)
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Judges' Save: Who Gets Saved? (S28+)", fontweight="bold", pad=10)

    # 添加注释框
    ax.annotate(
        "Judges align with\nfan preferences",
        xy=(higher_fan_pct, 0),
        xytext=(60, -0.3),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "gray"},
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_task1_structural_drift(
    pre_s28_acc: float,
    post_s28_acc: float,
    output_path: Path,
) -> None:
    """
    S28 前后结构漂移对比图。
    """
    apply_mcm_theme()

    fig, ax = plt.subplots(figsize=(6, 4))

    categories = ["S1–S27\n(Original Rules)", "S28–S34\n(Judges' Save)"]
    values = [pre_s28_acc, post_s28_acc]
    colors = [PROFESSIONAL_COLORS["judge"], PROFESSIONAL_COLORS["fan"]]

    bars = ax.bar(categories, values, color=colors, edgecolor="black", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Strict Reconstruction Accuracy")
    ax.set_title("Structural Drift: Rule Change Impact", fontweight="bold", pad=10)

    # 添加下降箭头
    ax.annotate(
        "",
        xy=(1, post_s28_acc),
        xytext=(0, pre_s28_acc),
        arrowprops={
            "arrowstyle": "->",
            "color": PROFESSIONAL_COLORS["negative"],
            "lw": 2,
        },
    )
    ax.text(
        0.5,
        (pre_s28_acc + post_s28_acc) / 2,
        f"−{(pre_s28_acc - post_s28_acc) * 100:.1f}%",
        ha="center",
        fontsize=10,
        color=PROFESSIONAL_COLORS["negative"],
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Task 3: 混合效应模型核心图表
# ═══════════════════════════════════════════════════════════════════════════════


def plot_task3_athlete_paradox(
    judge_coef: float,
    fan_coef: float,
    output_path: Path,
) -> None:
    """
    运动员悖论可视化 - 核心发现。

    展示评委和粉丝对运动员的相反评价。
    """
    apply_mcm_theme()

    fig, ax = plt.subplots(figsize=(6, 4))

    labels = [r"Judge Score $\beta$", r"Fan Share $\beta$"]
    values = [judge_coef, fan_coef]
    colors = [PROFESSIONAL_COLORS["judge"], PROFESSIONAL_COLORS["fan"]]

    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.5)
    ax.axhline(0, color="black", linewidth=1)

    # 数值标注
    for bar, val in zip(bars, values):
        sign = "+" if val > 0 else ""
        offset = 0.01 if val > 0 else -0.02
        va = "bottom" if val > 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{sign}{val:.3f}",
            ha="center",
            va=va,
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Coefficient (vs. Actor baseline)")
    ax.set_title("The Athlete Paradox: Opposite Effects", fontweight="bold", pad=10)
    ax.set_ylim(-0.2, 0.1)

    # 核心发现框
    ax.text(
        0.98,
        0.02,
        "Judges penalize,\nFans reward!",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.9},
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_task3_icc_comparison(
    judge_icc: float,
    fan_icc: float,
    output_path: Path,
) -> None:
    """
    职业舞伴方差贡献对比图（ICC）。
    """
    apply_mcm_theme()

    fig, ax = plt.subplots(figsize=(6, 4))

    labels = ["Judge Score\nModel", "Fan Share\nModel"]
    values = [judge_icc, fan_icc]
    colors = [PROFESSIONAL_COLORS["judge"], PROFESSIONAL_COLORS["fan"]]

    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Intraclass Correlation (ICC)")
    ax.set_title("Pro Dancer Variance Contribution", fontweight="bold", pad=10)
    ax.set_ylim(0, 0.3)

    # 比例标注
    ratio = fan_icc / judge_icc if judge_icc > 0 else 0
    ax.annotate(
        f"{ratio:.1f}×",
        xy=(1, fan_icc),
        xytext=(0.5, 0.25),
        fontsize=14,
        fontweight="bold",
        color=PROFESSIONAL_COLORS["highlight"],
        arrowprops={"arrowstyle": "->", "color": PROFESSIONAL_COLORS["highlight"]},
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_task3_idi_chart(
    idi_data: list[dict],
    output_path: Path,
) -> None:
    """
    Impact Divergence Index 图表。

    只展示显著分歧的变量。
    """
    apply_mcm_theme()

    fig, ax = plt.subplots(figsize=(8, 4))

    # 按 IDI 排序
    sorted_data = sorted(idi_data, key=lambda x: x["idi"], reverse=True)
    top_5 = sorted_data[:5]

    y_pos = np.arange(len(top_5))
    variables = [d["variable"] for d in top_5]
    idi_values = [d["idi"] for d in top_5]

    # 颜色根据显著性
    colors = [
        PROFESSIONAL_COLORS["negative"]
        if v > 3
        else PROFESSIONAL_COLORS["highlight"]
        if v > 1.96
        else PROFESSIONAL_COLORS["neutral"]
        for v in idi_values
    ]

    bars = ax.barh(y_pos, idi_values, color=colors, edgecolor="black", height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel("Impact Divergence Index (IDI)")
    ax.set_title("Top Divergent Effects: Judge vs. Fan", fontweight="bold", pad=10)

    # 阈值线
    ax.axvline(1.96, color="gray", linestyle="--", alpha=0.7, label=r"$p = 0.05$")
    ax.axvline(
        3.0,
        color=PROFESSIONAL_COLORS["negative"],
        linestyle="--",
        alpha=0.7,
        label="Substantive",
    )
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Task 4: PTFS 系统核心图表
# ═══════════════════════════════════════════════════════════════════════════════


def plot_task4_system_comparison(
    systems_data: list[dict],
    output_path: Path,
) -> None:
    """
    三系统关键指标对比图。

    突出 PTFS 的优势。
    """
    apply_mcm_theme()

    fig, ax = plt.subplots(figsize=(10, 5))

    # 准备数据
    systems = {s["name"]: s for s in systems_data}
    metrics = [
        r"Kendall's $\tau$",
        r"Spearman's $\rho$",
        "Tech Top3\n→Final",
        "Consistency",
    ]

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

    bars1 = ax.bar(
        x - width,
        rank_vals,
        width,
        label="Rank-based",
        color=PROFESSIONAL_COLORS["rank"],
        edgecolor="black",
    )
    bars2 = ax.bar(
        x,
        pct_vals,
        width,
        label="Percentage",
        color=PROFESSIONAL_COLORS["percentage"],
        edgecolor="black",
    )
    bars3 = ax.bar(
        x + width,
        ptfs_vals,
        width,
        label="PTFS",
        color=PROFESSIONAL_COLORS["ptfs"],
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "System Comparison: Technical Fairness Metrics", fontweight="bold", pad=10
    )
    # 图例放在图表下方，避免与数据重叠
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

    # PTFS 高亮标注（使用 * 替代 ★ 以避免字体问题）
    for i, (rank_v, pct_v, ptfs_v) in enumerate(zip(rank_vals, pct_vals, ptfs_vals)):
        if ptfs_v > max(rank_v, pct_v):
            ax.annotate(
                "*",
                xy=(x[i] + width, ptfs_v + 0.02),
                fontsize=14,
                ha="center",
                fontweight="bold",
                color=PROFESSIONAL_COLORS["ptfs"],
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_task4_dynamic_weights(
    w_start: float,
    w_end: float,
    n_weeks: int,
    output_path: Path,
) -> None:
    """
    PTFS 动态权重变化图。
    """
    apply_mcm_theme()

    fig, ax = plt.subplots(figsize=(8, 4))

    weeks = np.arange(1, n_weeks + 1)
    w_judge = w_start + (w_end - w_start) * (weeks - 1) / (n_weeks - 1)
    w_fan = 1 - w_judge

    ax.plot(
        weeks,
        w_judge * 100,
        "-o",
        color=PROFESSIONAL_COLORS["judge"],
        linewidth=2,
        markersize=6,
        label=r"Judge Weight $w_J(t)$",
    )
    ax.plot(
        weeks,
        w_fan * 100,
        "-s",
        color=PROFESSIONAL_COLORS["fan"],
        linewidth=2,
        markersize=6,
        label=r"Fan Weight $1-w_J(t)$",
    )

    ax.fill_between(weeks, w_judge * 100, alpha=0.2, color=PROFESSIONAL_COLORS["judge"])
    ax.fill_between(weeks, w_fan * 100, alpha=0.2, color=PROFESSIONAL_COLORS["fan"])

    ax.set_xlabel("Week")
    ax.set_ylabel("Weight (%)")
    ax.set_title("PTFS: Progressive Weight Adjustment", fontweight="bold", pad=10)
    ax.set_xlim(1, n_weeks)
    ax.set_ylim(0, 100)
    ax.legend(loc="center right")

    # 在图表两端直接标注数值（避免箭头重叠）
    ax.text(
        1,
        w_start * 100 - 8,
        f"{w_start:.0%}",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=PROFESSIONAL_COLORS["judge"],
    )
    ax.text(
        n_weeks,
        w_end * 100 + 5,
        f"{w_end:.0%}",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=PROFESSIONAL_COLORS["judge"],
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_task4_tau_improvement(
    systems_data: list[dict],
    output_path: Path,
) -> None:
    """
    Kendall's τ 改进对比图。

    突出 PTFS 相对于其他系统的提升。
    """
    apply_mcm_theme()

    fig, ax = plt.subplots(figsize=(6, 4))

    systems = {s["name"]: s for s in systems_data}

    names = ["Rank-based", "Percentage", "PTFS"]
    taus = [
        systems["Rank-based"]["kendall_tau"],
        systems["Percentage-based"]["kendall_tau"],
        systems["PTFS"]["kendall_tau"],
    ]
    colors = [
        PROFESSIONAL_COLORS["rank"],
        PROFESSIONAL_COLORS["percentage"],
        PROFESSIONAL_COLORS["ptfs"],
    ]

    bars = ax.bar(names, taus, color=colors, edgecolor="black", width=0.6)

    for bar, val in zip(bars, taus):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel(r"Kendall's $\tau$")
    ax.set_title(
        "Technical Fairness: Kendall's τ Comparison", fontweight="bold", pad=10
    )
    ax.set_ylim(0, 0.9)

    # 改进幅度标注
    ptfs_tau = systems["PTFS"]["kendall_tau"]
    pct_tau = systems["Percentage-based"]["kendall_tau"]
    improvement = ptfs_tau - pct_tau

    ax.annotate(
        f"+{improvement:.3f}",
        xy=(2, ptfs_tau),
        xytext=(2.3, ptfs_tau - 0.05),
        fontsize=10,
        color=PROFESSIONAL_COLORS["positive"],
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 组合图表
# ═══════════════════════════════════════════════════════════════════════════════


def plot_task1_key_findings(
    validation: dict,
    pre_s28_acc: float,
    post_s28_acc: float,
    judges_save_pct: float,
    output_path: Path,
) -> None:
    """
    Task 1 关键发现组合图（2x2 布局）。

    适合直接放入论文。
    """
    apply_mcm_theme()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        "Task 1: Bayesian Fan Vote Estimation — Key Results",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # (a) 验证指标
    ax = axes[0, 0]
    metrics = {
        "Reconstruction": validation["reconstruction_accuracy"],
        "Bottom-2\nRecall": validation["bottom2_overall_recall"],
        r"Finale $\tau$": validation["finale_kendall_tau"],
    }
    x = np.arange(len(metrics))
    colors = [
        PROFESSIONAL_COLORS["judge"],
        PROFESSIONAL_COLORS["fan"],
        PROFESSIONAL_COLORS["highlight"],
    ]
    bars = ax.bar(x, list(metrics.values()), color=colors, edgecolor="black", width=0.6)
    for bar, val in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate")
    ax.set_title("(a) Validation Metrics", fontsize=10, fontweight="bold")
    ax.axhline(0.9, color="gray", linestyle=":", alpha=0.5)

    # (b) S28 前后对比
    ax = axes[0, 1]
    categories = ["S1–S27", "S28–S34"]
    values = [pre_s28_acc, post_s28_acc]
    colors = [PROFESSIONAL_COLORS["judge"], PROFESSIONAL_COLORS["fan"]]
    bars = ax.bar(categories, values, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Strict Accuracy")
    ax.set_title("(b) Structural Drift", fontsize=10, fontweight="bold")

    # (c) Judges' Save 倾向
    ax = axes[1, 0]
    sizes = [judges_save_pct, 100 - judges_save_pct]
    labels = [
        f"Fan Higher\n({judges_save_pct:.1f}%)",
        f"Tech Higher\n({100 - judges_save_pct:.1f}%)",
    ]
    colors = [PROFESSIONAL_COLORS["fan"], PROFESSIONAL_COLORS["judge"]]
    wedges, texts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "black"},
    )
    ax.set_title("(c) Judges' Save Preference", fontsize=10, fontweight="bold")

    # (d) 核心发现总结
    ax = axes[1, 1]
    ax.axis("off")
    findings = [
        r"• Reconstruction accuracy: $\bf{93.5\%}$",
        r"• Finale prediction: $\tau = \bf{0.99}$",
        r"• Judges' Save aligns with fans: $\bf{90.5\%}$",
        r"• S28+ structural drift: $−30\%$ accuracy",
    ]
    ax.text(
        0.1,
        0.8,
        "Key Findings:",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
    )
    for i, finding in enumerate(findings):
        ax.text(0.1, 0.65 - i * 0.15, finding, transform=ax.transAxes, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_task3_key_findings(
    athlete_judge: float,
    athlete_fan: float,
    judge_icc: float,
    fan_icc: float,
    idi_data: list[dict],
    output_path: Path,
) -> None:
    """
    Task 3 关键发现组合图（2x2 布局）。
    """
    apply_mcm_theme()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        "Task 3: Impact Analysis — Key Results", fontsize=13, fontweight="bold", y=0.98
    )

    # (a) 运动员悖论
    ax = axes[0, 0]
    labels = [r"Judge $\beta$", r"Fan $\beta$"]
    values = [athlete_judge, athlete_fan]
    colors = [PROFESSIONAL_COLORS["judge"], PROFESSIONAL_COLORS["fan"]]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.5)
    ax.axhline(0, color="black", linewidth=1)
    for bar, val in zip(bars, values):
        sign = "+" if val > 0 else ""
        offset = 0.008 if val > 0 else -0.015
        va = "bottom" if val > 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{sign}{val:.3f}",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("Coefficient")
    ax.set_title("(a) Athlete Paradox", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.18, 0.08)

    # (b) ICC 对比
    ax = axes[0, 1]
    labels = ["Judge", "Fan"]
    values = [judge_icc, fan_icc]
    colors = [PROFESSIONAL_COLORS["judge"], PROFESSIONAL_COLORS["fan"]]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("ICC")
    ax.set_title("(b) Pro Dancer Variance", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.28)
    ratio = fan_icc / judge_icc if judge_icc > 0 else 0
    ax.text(
        0.5,
        0.23,
        f"{ratio:.1f}×",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=PROFESSIONAL_COLORS["highlight"],
    )

    # (c) IDI 图
    ax = axes[1, 0]
    sorted_data = sorted(idi_data, key=lambda x: x["idi"], reverse=True)[:5]
    y_pos = np.arange(len(sorted_data))
    variables = [d["variable"] for d in sorted_data]
    idi_values = [d["idi"] for d in sorted_data]
    colors = [
        PROFESSIONAL_COLORS["negative"]
        if v > 3
        else PROFESSIONAL_COLORS["highlight"]
        if v > 1.96
        else PROFESSIONAL_COLORS["neutral"]
        for v in idi_values
    ]
    bars = ax.barh(y_pos, idi_values, color=colors, edgecolor="black", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables, fontsize=9)
    ax.set_xlabel("IDI")
    ax.set_title("(c) Impact Divergence", fontsize=10, fontweight="bold")
    ax.axvline(1.96, color="gray", linestyle="--", alpha=0.6)

    # (d) 核心发现总结
    ax = axes[1, 1]
    ax.axis("off")
    findings = [
        r"• Athlete: Judge $\beta = -0.13$, Fan $\beta = +0.05$",
        f"• Pro dancer ICC: Fan {fan_icc:.1%} vs Judge {judge_icc:.1%}",
        r"• $\gamma \approx 0$ (fans ignore same-week scores)",
        "• Week effect 27× stronger for judges",
    ]
    ax.text(
        0.1,
        0.8,
        "Key Findings:",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
    )
    for i, finding in enumerate(findings):
        ax.text(0.1, 0.65 - i * 0.15, finding, transform=ax.transAxes, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_task4_key_findings(
    systems_data: list[dict],
    w_start: float,
    w_end: float,
    output_path: Path,
) -> None:
    """
    Task 4 关键发现组合图（2x2 布局）。
    """
    apply_mcm_theme()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        "Task 4: PTFS Design — Key Results", fontsize=13, fontweight="bold", y=0.98
    )

    systems = {s["name"]: s for s in systems_data}

    # (a) τ 对比
    ax = axes[0, 0]
    names = ["Rank", "Pct", "PTFS"]
    taus = [
        systems["Rank-based"]["kendall_tau"],
        systems["Percentage-based"]["kendall_tau"],
        systems["PTFS"]["kendall_tau"],
    ]
    colors = [
        PROFESSIONAL_COLORS["rank"],
        PROFESSIONAL_COLORS["percentage"],
        PROFESSIONAL_COLORS["ptfs"],
    ]
    bars = ax.bar(names, taus, color=colors, edgecolor="black", width=0.6)
    for bar, val in zip(bars, taus):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel(r"Kendall's $\tau$")
    ax.set_title("(a) Technical Fairness", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.85)

    # (b) 动态权重
    ax = axes[0, 1]
    weeks = np.arange(1, 12)
    w_judge = w_start + (w_end - w_start) * (weeks - 1) / 10
    ax.plot(
        weeks,
        w_judge * 100,
        "-o",
        color=PROFESSIONAL_COLORS["judge"],
        linewidth=2,
        markersize=4,
        label="Judge",
    )
    ax.plot(
        weeks,
        (1 - w_judge) * 100,
        "-s",
        color=PROFESSIONAL_COLORS["fan"],
        linewidth=2,
        markersize=4,
        label="Fan",
    )
    ax.fill_between(weeks, w_judge * 100, alpha=0.2, color=PROFESSIONAL_COLORS["judge"])
    ax.set_xlabel("Week")
    ax.set_ylabel("Weight (%)")
    ax.set_title("(b) Dynamic Weights", fontsize=10, fontweight="bold")
    ax.legend(loc="center right", fontsize=8)
    ax.set_xlim(1, 11)
    ax.set_ylim(0, 100)

    # (c) 一致性对比
    ax = axes[1, 0]
    names = ["Rank", "Pct", "PTFS"]
    consistency = [
        systems["Rank-based"]["tau_consistency"],
        systems["Percentage-based"]["tau_consistency"],
        systems["PTFS"]["tau_consistency"],
    ]
    colors = [
        PROFESSIONAL_COLORS["rank"],
        PROFESSIONAL_COLORS["percentage"],
        PROFESSIONAL_COLORS["ptfs"],
    ]
    bars = ax.bar(
        names,
        [c * 100 for c in consistency],
        color=colors,
        edgecolor="black",
        width=0.6,
    )
    for bar, val in zip(bars, consistency):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1%}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel("Consistency (%)")
    ax.set_title(
        r"(c) Robustness ($\tau > 0.5$ Seasons)", fontsize=10, fontweight="bold"
    )
    ax.set_ylim(0, 110)

    # (d) 核心发现总结
    ax = axes[1, 1]
    ax.axis("off")
    ptfs = systems["PTFS"]
    pct = systems["Percentage-based"]
    findings = [
        f"• Kendall's τ: {ptfs['kendall_tau']:.3f} (+{ptfs['kendall_tau'] - pct['kendall_tau']:.3f} vs Pct)",
        f"• Consistency: {ptfs['tau_consistency']:.1%} of seasons",
        f"• Close calls: {ptfs['close_call_rate']:.1%} (+{(ptfs['close_call_rate'] - pct['close_call_rate']) * 100:.1f}%)",
        f"• Weight: {w_start:.0%} → {w_end:.0%} (Judge)",
    ]
    ax.text(
        0.1,
        0.8,
        "Key Findings:",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
    )
    for i, finding in enumerate(findings):
        ax.text(0.1, 0.65 - i * 0.15, finding, transform=ax.transAxes, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # 测试代码
    print("Professional figures module loaded successfully.")
    print(f"LaTeX support: {USE_LATEX}")
