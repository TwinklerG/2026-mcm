"""
可视化模块 - 生成 Task 3 的图表。

图表列表：
1. 职业舞伴效应散点图（技术 vs 流量）
2. 固定效应系数对比图
3. 方差分解图
4. 行业效应对比图
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# 设置随机种子以确保可重复性
np.random.seed(42)
import random

random.seed(42)

# 论文发表的样式设置 - 与 task1 一致
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.hashsalt": "42",  # 确保 SVG 输出的确定性
})

# 配色方案 - 与 task1 一致
COLORS = {
    "Kingmaker": "#2ECC71",  # 绿色 - 与 task1 的 success 一致
    "Technician": "#2E86AB",  # 深蓝 - 与 task1 的 strict 一致
    "Fan Favorite": "#A23B72",  # 紫红 - 与 task1 的 bottom2 一致
    "Underperformer": "#95A5A6",  # 灰色 - 与 task1 的 neutral 一致
    "judge": "#2E86AB",  # 深蓝
    "fan": "#A23B72",  # 紫红
    "overall": "#F18F01",  # 橙色
}


def plot_pro_scatter(
    pro_rankings: pd.DataFrame,
    output_path: Path,
    top_n_labels: int = 10,
) -> None:
    """
    绘制职业舞伴效应散点图。

    X轴：技术加成 (Tech Boost)
    Y轴：流量加成 (Popularity Boost)
    颜色：舞伴类别

    Parameters
    ----------
    pro_rankings : pd.DataFrame
        职业舞伴排名数据
    output_path : Path
        输出路径
    top_n_labels : int
        标注前 N 名舞伴的名字
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制散点
    for category in ["Kingmaker", "Technician", "Fan Favorite", "Underperformer"]:
        mask = pro_rankings["category"] == category
        data = pro_rankings[mask]
        ax.scatter(
            data["tech_boost"],
            data["pop_boost"],
            c=COLORS[category],
            label=category,
            s=120,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

    # 添加参考线
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    # 标注 top performers
    top_performers = pro_rankings.nlargest(top_n_labels, "combined_score")
    for _, row in top_performers.iterrows():
        ax.annotate(
            row["pro_id"],
            (row["tech_boost"], row["pop_boost"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
        )

    ax.set_xlabel("Technical Boost (Judge Score BLUP)", fontsize=11)
    ax.set_ylabel("Popularity Boost (Fan Share BLUP)", fontsize=11)
    ax.set_title(
        "Professional Dancer Impact Classification", fontsize=12, fontweight="bold"
    )
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle=":")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_coefficient_comparison(
    judge_effects: dict,
    fan_effects: dict,
    output_path: Path,
) -> None:
    """
    绘制固定效应系数对比图。

    Parameters
    ----------
    judge_effects : dict
        评委模型固定效应
    fan_effects : dict
        粉丝模型固定效应
    output_path : Path
        输出路径
    """
    # 准备数据
    common_vars = set(judge_effects.keys()) & set(fan_effects.keys())
    common_vars.discard("Intercept")

    data = []
    for var in sorted(common_vars):
        data.append({
            "variable": var,
            "model": "Judge Score",
            "coefficient": judge_effects[var]["estimate"],
            "se": judge_effects[var]["std_error"],
        })
        data.append({
            "variable": var,
            "model": "Fan Share",
            "coefficient": fan_effects[var]["estimate"],
            "se": fan_effects[var]["std_error"],
        })

    df = pd.DataFrame(data)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    variables = sorted(common_vars)
    x = np.arange(len(variables))
    width = 0.35

    judge_data = df[df["model"] == "Judge Score"].set_index("variable")
    fan_data = df[df["model"] == "Fan Share"].set_index("variable")

    # 绘制条形图
    ax.bar(
        x - width / 2,
        [judge_data.loc[v, "coefficient"] for v in variables],
        width,
        yerr=[judge_data.loc[v, "se"] for v in variables],
        label="Judge Score Model",
        color=COLORS["judge"],
        alpha=0.8,
        capsize=3,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        [fan_data.loc[v, "coefficient"] for v in variables],
        width,
        yerr=[fan_data.loc[v, "se"] for v in variables],
        label="Fan Share Model",
        color=COLORS["fan"],
        alpha=0.8,
        capsize=3,
        edgecolor="black",
        linewidth=0.5,
    )

    # 添加参考线
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Variable", fontsize=11)
    ax.set_ylabel("Coefficient Estimate", fontsize=11)
    ax.set_title(
        "Fixed Effects Comparison: Judge Score vs Fan Share",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(variables, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_variance_decomposition(
    variance_decomp: dict,
    output_path: Path,
) -> None:
    """
    绘制方差分解饼图。

    Parameters
    ----------
    variance_decomp : dict
        方差分解结果
    output_path : Path
        输出路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sorted_models = sorted(variance_decomp.keys())
    for ax, model in zip(axes, sorted_models):
        data = variance_decomp[model]
        sizes = [data["pro_variance_pct"], 100 - data["pro_variance_pct"]]
        labels = [
            f"Pro Dancer\n({data['pro_variance_pct']:.1f}%)",
            f"Residual\n({100 - data['pro_variance_pct']:.1f}%)",
        ]
        colors = [COLORS["judge"] if model == "judge" else COLORS["fan"], "#BDC3C7"]
        explode = (0.05, 0)

        wedges, texts, autotexts = ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 9},
        )
        ax.set_title(
            f"{model.capitalize()} Model\nICC = {data['icc']:.3f}",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(
        "Variance Decomposition: Pro Dancer Effect",
        fontsize=12,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_industry_effects(
    celebrity_analysis: dict,
    output_path: Path,
) -> None:
    """
    绘制行业效应对比图。

    Parameters
    ----------
    celebrity_analysis : dict
        明星特征分析结果
    output_path : Path
        输出路径
    """
    industry_effects = celebrity_analysis.get("industry_effects", {})
    if not industry_effects:
        print("没有行业效应数据")
        return

    # 准备数据
    data = []
    sorted_industries = sorted(industry_effects.keys())
    for industry in sorted_industries:
        effects = industry_effects[industry]
        if effects["judge"]:
            data.append({
                "industry": industry,
                "model": "Judge Score",
                "coefficient": effects["judge"]["coefficient"],
                "significant": effects["judge"]["significant"],
            })
        if effects["fan"]:
            data.append({
                "industry": industry,
                "model": "Fan Share",
                "coefficient": effects["fan"]["coefficient"],
                "significant": effects["fan"]["significant"],
            })

    df = pd.DataFrame(data)
    if df.empty:
        print("没有足够的行业效应数据")
        return

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))

    industries = sorted(df["industry"].unique())
    x = np.arange(len(industries))
    width = 0.35

    judge_data = df[df["model"] == "Judge Score"]
    fan_data = df[df["model"] == "Fan Share"]

    # 创建索引映射
    judge_dict = dict(zip(judge_data["industry"], judge_data["coefficient"]))
    fan_dict = dict(zip(fan_data["industry"], fan_data["coefficient"]))

    judge_coeffs = [judge_dict.get(ind, 0) for ind in industries]
    fan_coeffs = [fan_dict.get(ind, 0) for ind in industries]

    ax.bar(
        x - width / 2,
        judge_coeffs,
        width,
        label="Judge Score",
        color=COLORS["judge"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        fan_coeffs,
        width,
        label="Fan Share",
        color=COLORS["fan"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Industry (vs Actor/Actress baseline)", fontsize=11)
    ax.set_ylabel("Coefficient", fontsize=11)
    ax.set_title(
        "Industry Effects on Judge Scores and Fan Shares",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(industries, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_learning_curve(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    绘制学习曲线（周次效应）。

    Parameters
    ----------
    data : pd.DataFrame
        面板数据
    output_path : Path
        输出路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 按周次汇总
    weekly_stats = (
        data
        .groupby("week")
        .agg({
            "judge_score": ["mean", "std"],
            "log_fan_share": ["mean", "std"],
        })
        .reset_index()
    )
    weekly_stats.columns = [
        "week",
        "judge_mean",
        "judge_std",
        "fan_mean",
        "fan_std",
    ]

    # 评委得分趋势
    ax1 = axes[0]
    ax1.errorbar(
        weekly_stats["week"],
        weekly_stats["judge_mean"],
        yerr=weekly_stats["judge_std"] / np.sqrt(len(data)),
        marker="o",
        capsize=3,
        color=COLORS["judge"],
        linewidth=2,
        markersize=6,
        alpha=0.8,
    )
    ax1.set_xlabel("Week", fontsize=11)
    ax1.set_ylabel("Mean Judge Score", fontsize=11)
    ax1.set_title("Learning Curve: Judge Scores", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.25, linestyle=":")

    # 粉丝份额趋势
    ax2 = axes[1]
    ax2.errorbar(
        weekly_stats["week"],
        weekly_stats["fan_mean"],
        yerr=weekly_stats["fan_std"] / np.sqrt(len(data)),
        marker="o",
        capsize=3,
        color=COLORS["fan"],
        linewidth=2,
        markersize=6,
        alpha=0.8,
    )
    ax2.set_xlabel("Week", fontsize=11)
    ax2.set_ylabel("Mean Log Fan Share", fontsize=11)
    ax2.set_title("Trend: Fan Shares Over Time", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.25, linestyle=":")

    fig.suptitle(
        "Weekly Trends in Performance and Popularity",
        fontsize=12,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_impact_divergence(
    idi_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    绘制影响差异度指数 (IDI) 图。

    Parameters
    ----------
    idi_df : pd.DataFrame
        IDI 数据
    output_path : Path
        输出路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 按 IDI 排序
    idi_df = idi_df.sort_values("IDI", ascending=True)

    # 绘制水平条形图
    colors = [
        COLORS["overall"] if sig else COLORS["Underperformer"]
        for sig in idi_df["significant"]
    ]
    ax.barh(
        idi_df["variable"],
        idi_df["IDI"],
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # 添加显著性阈值线
    ax.axvline(x=1.96, color="red", linestyle="--", linewidth=1.5, label="p < 0.05")

    ax.set_xlabel("Impact Divergence Index (IDI)", fontsize=11)
    ax.set_ylabel("Variable", fontsize=11)
    ax.set_title(
        "Do Variables Impact Judge Scores and Fan Votes Differently?",
        fontsize=12,
        fontweight="bold",
    )

    # 图例
    legend_elements = [
        Patch(facecolor=COLORS["overall"], alpha=0.8, label="Significant Difference"),
        Patch(
            facecolor=COLORS["Underperformer"],
            alpha=0.8,
            label="No Significant Difference",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pro_heatmap(pro_rankings: pd.DataFrame, output_path: Path) -> None:
    """
    Pro 效应热图：展示每个舞伴的技术和流量加成。
    """
    # 按 combined_score 排序，取前 30
    top_pros = pro_rankings.nlargest(30, "combined_score")

    fig, ax = plt.subplots(figsize=(12, 10))

    # 准备热图数据
    pro_names = top_pros["pro_id"].tolist()
    tech_boosts = top_pros["tech_boost"].tolist()
    pop_boosts = top_pros["pop_boost"].tolist()

    # 创建热图矩阵
    data_matrix = np.array([tech_boosts, pop_boosts]).T

    # 绘制热图
    im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)

    # 设置刻度
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Tech Boost", "Pop Boost"], fontsize=11)
    ax.set_yticks(np.arange(len(pro_names)))
    ax.set_yticklabels(pro_names, fontsize=8)

    # 添加数值标签
    for i in range(len(pro_names)):
        for j in range(2):
            text = ax.text(
                j, i, f"{data_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7
            )

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Random Effect (BLUP)", rotation=270, labelpad=20, fontsize=10)

    ax.set_title(
        "Top 30 Professional Dancers: Random Effects Heatmap",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_random_effects_distribution(
    judge_blups: pd.DataFrame, fan_blups: pd.DataFrame, output_path: Path
) -> None:
    """
    随机效应分布对比：Judge 模型 vs Fan 模型。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 直方图对比
    ax = axes[0]
    ax.hist(
        judge_blups["intercept"],
        bins=30,
        alpha=0.6,
        color=COLORS["judge"],
        edgecolor="black",
        label="Judge Model",
    )
    ax.hist(
        fan_blups["intercept"],
        bins=30,
        alpha=0.6,
        color=COLORS["fan"],
        edgecolor="black",
        label="Fan Model",
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Line")
    ax.set_xlabel("Random Effect (BLUP)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of Pro Random Effects", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # 2. 散点图对比
    ax = axes[1]
    merged = pd.merge(
        judge_blups,
        fan_blups,
        on="pro_id",
        suffixes=("_judge", "_fan"),
    )
    ax.scatter(
        merged["intercept_judge"],
        merged["intercept_fan"],
        alpha=0.6,
        s=80,
        edgecolor="black",
        linewidth=0.5,
    )

    # 添加对角线
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "r--", alpha=0.75, linewidth=2, label="x=y")

    ax.set_xlabel("Judge Model BLUP", fontsize=11)
    ax.set_ylabel("Fan Model BLUP", fontsize=11)
    ax.set_title("Pro Effects Correlation", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # 计算相关系数
    corr = merged["intercept_judge"].corr(merged["intercept_fan"])
    ax.text(
        0.05,
        0.95,
        f"Correlation: {corr:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_age_effect_curve(data: pd.DataFrame, model, output_path: Path) -> None:
    """
    年龄效应曲线：展示年龄对评委分数和粉丝投票的非线性影响。
    """
    if model.judge_model is None or model.fan_model is None:
        return

    # 提取年龄系数
    age_coef_judge = model.judge_model.fixed_effects.get("age_centered", {}).get(
        "estimate", 0
    )
    age_coef_fan = model.fan_model.fixed_effects.get("age_centered", {}).get(
        "estimate", 0
    )

    # 创建年龄范围
    age_range = np.linspace(
        data["celebrity_age_during_season"].min(),
        data["celebrity_age_during_season"].max(),
        100,
    )
    age_mean = data["celebrity_age_during_season"].mean()
    age_centered = age_range - age_mean

    # 预测效应（仅年龄部分）
    effect_judge = age_coef_judge * age_centered
    effect_fan = age_coef_fan * age_centered

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 年龄效应曲线
    ax = axes[0]
    ax.plot(
        age_range,
        effect_judge,
        linewidth=2,
        color=COLORS["judge"],
        label="Judge Score",
    )
    ax.plot(age_range, effect_fan, linewidth=2, color=COLORS["fan"], label="Fan Share")
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(age_mean, color="gray", linestyle=":", linewidth=1.5, label="Mean Age")
    ax.set_xlabel("Celebrity Age", fontsize=11)
    ax.set_ylabel("Predicted Effect", fontsize=11)
    ax.set_title("Age Effect on Performance", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. 年龄分布 + 密度
    ax = axes[1]
    ax.hist(
        data["celebrity_age_during_season"],
        bins=30,
        alpha=0.6,
        color=COLORS["overall"],
        edgecolor="black",
        density=True,
    )
    ax.axvline(
        age_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {age_mean:.1f}",
    )
    ax.set_xlabel("Celebrity Age", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Age Distribution in Sample", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_variance_radar(variance_decomp: dict, output_path: Path) -> None:
    """
    方差解释雷达图：多维度展示方差分解结果。
    """
    # 准备数据
    categories = [
        "Pro Variance\n(Judge)",
        "Pro Variance\n(Fan)",
        "ICC\n(Judge)",
        "ICC\n(Fan)",
    ]

    values = [
        variance_decomp["judge"]["pro_variance_pct"] / 100,
        variance_decomp["fan"]["pro_variance_pct"] / 100,
        variance_decomp["judge"]["icc"],
        variance_decomp["fan"]["icc"],
    ]

    # 闭合雷达图
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
        color=COLORS["overall"],
        label="Observed Values",
    )
    ax.fill(angles, values, alpha=0.25, color=COLORS["overall"])

    # 设置刻度
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["20%", "40%", "60%", "80%"], fontsize=9)
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
            color=COLORS["overall"],
        )

    plt.title(
        "Variance Decomposition (Radar Chart)",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pro_rankings_bar(pro_rankings: pd.DataFrame, output_path: Path) -> None:
    """
    Pro 排名条形图：Top 20 舞伴的综合得分。
    """
    # 取前 20
    top_20 = pro_rankings.nlargest(20, "combined_score")

    fig, ax = plt.subplots(figsize=(12, 8))

    # 按类别着色
    colors = [COLORS[cat] for cat in top_20["category"]]

    ax.barh(
        range(len(top_20)),
        top_20["combined_score"],
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20["pro_id"], fontsize=9)
    ax.set_xlabel("Combined Score (Standardized)", fontsize=11)
    ax.set_title(
        "Top 20 Professional Dancers (Combined Score)",
        fontsize=12,
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # 图例
    legend_elements = [
        Patch(facecolor=COLORS["Kingmaker"], alpha=0.8, label="Kingmaker"),
        Patch(facecolor=COLORS["Technician"], alpha=0.8, label="Technician"),
        Patch(facecolor=COLORS["Fan Favorite"], alpha=0.8, label="Fan Favorite"),
        Patch(facecolor=COLORS["Underperformer"], alpha=0.8, label="Underperformer"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_plots(
    model,
    data: pd.DataFrame,
    pro_rankings: pd.DataFrame,
    variance_decomp: dict,
    celebrity_analysis: dict,
    idi_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    生成所有图表。

    Parameters
    ----------
    model : DualTrackHMEM
        拟合后的模型
    data : pd.DataFrame
        面板数据
    pro_rankings : pd.DataFrame
        职业舞伴排名
    variance_decomp : dict
        方差分解结果
    celebrity_analysis : dict
        明星特征分析结果
    idi_df : pd.DataFrame
        IDI 数据
    output_dir : Path
        输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 职业舞伴散点图
    plot_pro_scatter(pro_rankings, output_dir / "pro_scatter.png")

    # 2. 系数对比图
    if model.judge_model and model.fan_model:
        plot_coefficient_comparison(
            model.judge_model.fixed_effects,
            model.fan_model.fixed_effects,
            output_dir / "coefficient_comparison.png",
        )

    # 3. 方差分解图
    plot_variance_decomposition(
        variance_decomp, output_dir / "variance_decomposition.png"
    )

    # 4. 行业效应图
    plot_industry_effects(celebrity_analysis, output_dir / "industry_effects.png")

    # 5. 学习曲线图
    plot_learning_curve(data, output_dir / "learning_curve.png")

    # 6. IDI 图
    plot_impact_divergence(idi_df, output_dir / "impact_divergence.png")

    # ========== 新增科研级别图表 ==========

    # 7. Pro 效应热图
    plot_pro_heatmap(pro_rankings, output_dir / "pro_heatmap.png")

    # 8. 随机效应分布对比
    if model.judge_model and model.fan_model:
        plot_random_effects_distribution(
            model.judge_model.blups["pro_id"],
            model.fan_model.blups["pro_id"],
            output_dir / "random_effects_distribution.png",
        )

    # 9. 年龄效应曲线
    plot_age_effect_curve(data, model, output_dir / "age_effect_curve.png")

    # 10. 方差解释雷达图
    plot_variance_radar(variance_decomp, output_dir / "variance_radar.png")

    # 11. Pro 排名条形图
    plot_pro_rankings_bar(pro_rankings, output_dir / "pro_rankings_bar.png")

    print(f"\n所有图表已保存至: {output_dir}")
