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
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# 设置随机种子以确保可重复性
np.random.seed(42)
import random

random.seed(42)

# 字体 1.4 倍，图例不放大
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 17,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 11,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.hashsalt": "42",
})

# 红蓝配色（原方案），绘图时加透明度
COLORS = {
    "Kingmaker": "#2E7D32",
    "Technician": "#1565C0",
    "Fan Favorite": "#C62828",
    "Underperformer": "#78909C",
    "judge": "#1565C0",   # 蓝
    "fan": "#C62828",     # 红
    "overall": "#F57C00",
}
# 饼图残差块用弱化灰，不刺眼
PIE_RESIDUAL_GRAY = "#E8E8E8"
# 网格与零线
GRID_COLOR = "#E8E8E8"
GRID_ALPHA = 0.6
ZERO_LINE_COLOR = "#9E9E9E"
EDGE_COLOR = "#424242"

VAR_LABELS = {
    "age_centered": "Age",
    "week": "Week",
    "industry_Athlete": "Athlete",
    "industry_Entertainment": "Entertainment",
    "industry_Model": "Model",
    "industry_Other": "Other",
    "industry_Singer": "Singer",
    "industry_TV": "TV",
}


def plot_pro_scatter(
    pro_rankings: pd.DataFrame,
    output_path: Path,
    top_n_labels: int = 8,
) -> None:
    """
    绘制职业舞伴效应散点图：技术–人气平面，四象限；BLUP 点旁不标注名字。

    X轴：技术加成 (Judge BLUP)；Y轴：人气加成 (Fan BLUP)；颜色：四类角色。
    """
    fig, ax = plt.subplots(figsize=(10, 9))

    for category in ["Kingmaker", "Technician", "Fan Favorite", "Underperformer"]:
        mask = pro_rankings["category"] == category
        data = pro_rankings[mask]
        ax.scatter(
            data["tech_boost"],
            data["pop_boost"],
            c=COLORS[category],
            label=category,
            s=110,
            alpha=0.88,
            edgecolors=EDGE_COLOR,
            linewidth=0.6,
        )

    ax.axhline(y=0, color=ZERO_LINE_COLOR, linestyle="-", linewidth=1)
    ax.axvline(x=0, color=ZERO_LINE_COLOR, linestyle="-", linewidth=1)

    ax.set_xlabel("Technical Boost (Judge BLUP)", fontsize=18)
    ax.set_ylabel("Popularity Boost (Fan BLUP)", fontsize=18)
    ax.set_title(
        "Technology–Popularity Plane",
        fontsize=20,
        fontweight="bold",
        pad=14,
    )
    ax.legend(loc="upper left", framealpha=0.95, fontsize=11)
    ax.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linestyle="-")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout(pad=1.2)
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_coefficient_comparison(
    judge_effects: dict,
    fan_effects: dict,
    output_path: Path,
) -> None:
    """
    绘制固定效应系数对比图：可读变量名、双轨柱状、显著性标记。

    Parameters
    ----------
    judge_effects : dict
        评委模型固定效应（含 estimate, std_error, p_value）
    fan_effects : dict
        粉丝模型固定效应
    output_path : Path
        输出路径
    """
    common_vars = set(judge_effects.keys()) & set(fan_effects.keys())
    common_vars.discard("Intercept")
    variables = sorted(common_vars)
    display_names = [VAR_LABELS.get(v, v.replace("industry_", "")) for v in variables]

    judge_coeffs = [judge_effects[v]["estimate"] for v in variables]
    judge_ses = [judge_effects[v]["std_error"] for v in variables]
    fan_coeffs = [fan_effects[v]["estimate"] for v in variables]
    fan_ses = [fan_effects[v]["std_error"] for v in variables]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    x = np.arange(len(variables))
    width = 0.36

    ax.bar(
        x - width / 2,
        judge_coeffs,
        width,
        yerr=judge_ses,
        label="Judge Score",
        color=COLORS["judge"],
        alpha=0.88,
        capsize=3,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        fan_coeffs,
        width,
        yerr=fan_ses,
        label="Fan Share",
        color=COLORS["fan"],
        alpha=0.88,
        capsize=3,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
    )

    ax.axhline(y=0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=1)
    ax.set_xlabel("Covariate", fontsize=18)
    ax.set_ylabel("Coefficient Estimate", fontsize=18)
    ax.set_title(
        "Fixed Effects: Judge Score vs Fan Share",
        fontsize=20,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=35, ha="right", fontsize=15)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=GRID_ALPHA, linestyle="-")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_dual_track_factor_distributions(panel_df: pd.DataFrame, output_path: Path) -> None:
    """
    双轨模型考虑的因素：各因子在样本中的分布（写公式前的描述性分析）。

    四栏：Age, Week, Industry, Judge Score。用于 7.2 节「我们的双轨考虑了哪些因素」。
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    age_col = "celebrity_age_during_season"
    if age_col not in panel_df.columns:
        age_col = "age_centered"
        if age_col in panel_df.columns:
            age_mean = float(panel_df["age_centered"].mean())
            age_vals = panel_df["age_centered"].to_numpy() + age_mean
        else:
            age_vals = np.array([])
    else:
        age_vals = panel_df[age_col].to_numpy()

    # 1. Age distribution
    ax = axes[0]
    if len(age_vals) > 0:
        ax.hist(
            age_vals,
            bins=min(30, max(10, len(np.unique(age_vals)) // 2)),
            alpha=0.7,
            color=COLORS["judge"],
            edgecolor=EDGE_COLOR,
            density=True,
        )
        ax.axvline(
            np.mean(age_vals),
            color="#757575",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {np.mean(age_vals):.1f}",
        )
    ax.set_xlabel("Celebrity Age", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.set_title("Age (factor in both tracks)", fontsize=17, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=GRID_ALPHA)

    # 2. Week distribution
    ax = axes[1]
    week_vals = panel_df["week"].to_numpy()
    week_counts = pd.Series(week_vals).value_counts().sort_index()
    ax.bar(
        week_counts.index,
        week_counts.values,
        color=COLORS["fan"],
        alpha=0.88,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
    )
    ax.set_xlabel("Week in Season", fontsize=16)
    ax.set_ylabel("Observation Count", fontsize=16)
    ax.set_title("Week (factor in both tracks)", fontsize=17, fontweight="bold")
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=GRID_ALPHA)

    # 3. Industry distribution
    ax = axes[2]
    ind_col = "industry_std" if "industry_std" in panel_df.columns else "celebrity_industry"
    ind_counts = panel_df[ind_col].value_counts()
    ind_labels = [str(x)[:12] for x in ind_counts.index]
    x_pos = np.arange(len(ind_counts))
    ax.bar(
        x_pos,
        ind_counts.values,
        color=COLORS["overall"],
        alpha=0.88,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ind_labels, rotation=35, ha="right", fontsize=12)
    ax.set_xlabel("Industry", fontsize=16)
    ax.set_ylabel("Observation Count", fontsize=16)
    ax.set_title("Industry (factor in both tracks)", fontsize=17, fontweight="bold")
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=GRID_ALPHA)

    # 4. Judge score distribution (covariate in fan track)
    ax = axes[3]
    j_vals = panel_df["judge_score"].to_numpy()
    ax.hist(
        j_vals,
        bins=min(35, max(15, len(np.unique(j_vals)) // 2)),
        alpha=0.7,
        color=COLORS["fan"],
        edgecolor=EDGE_COLOR,
        density=True,
    )
    ax.axvline(
        np.mean(j_vals),
        color="#757575",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {np.mean(j_vals):.2f}",
    )
    ax.set_xlabel("Judge Score", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.set_title("Judge Score (covariate in fan track)", fontsize=17, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=GRID_ALPHA)

    fig.suptitle(
        "Distributions of Factors in the Dual-Track Model (7.2)",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_separation_rationale(
    panel_df: pd.DataFrame,
    idi_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    用指标说明「评委–粉丝分轨」的合理性：左图观测相关，右图 IDI（预测因子效应差异）。

    - 左：judge_score vs log_fan_share 散点 + Pearson r（相关但非同一维度）。
    - 右：IDI 条形图，阈值 1.96；IDI > 1.96 表示评委与粉丝对该因子反应显著不同。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left: scatter judge_score vs log_fan_share
    ax = axes[0]
    j = panel_df["judge_score"].to_numpy()
    f = panel_df["log_fan_share"].to_numpy()
    if len(j) > 4000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(j), 4000, replace=False)
        j, f = j[idx], f[idx]
    r = np.corrcoef(j, f)[0, 1] if len(j) > 1 else 0.0
    ax.scatter(j, f, alpha=0.25, s=12, c=COLORS["judge"], edgecolors="none")
    ax.set_xlabel("Judge Score", fontsize=18)
    ax.set_ylabel("Log Fan Share", fontsize=18)
    ax.set_title(
        f"Judge vs Fan at Observation Level\n(Pearson r = {r:.3f}; distinct dimensions)",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linestyle="-")

    # Right: IDI by variable
    ax = axes[1]
    variables = idi_df["variable"].tolist()
    display_names = [VAR_LABELS.get(v, v.replace("industry_", "")) for v in variables]
    idi_vals = idi_df["IDI"].tolist()
    colors_idi = [COLORS["fan"] if float(x) > 1.96 else COLORS["judge"] for x in idi_vals]
    x = np.arange(len(variables))
    bars = ax.bar(
        x,
        idi_vals,
        color=colors_idi,
        alpha=0.88,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
    )
    ax.axhline(1.96, color="#757575", linestyle="--", linewidth=1.2, label="IDI = 1.96 (α=0.05)")
    ax.axhline(3.0, color="#9E9E9E", linestyle=":", linewidth=1.0, label="IDI = 3 (substantial)")
    ax.set_xlabel("Predictor", fontsize=18)
    ax.set_ylabel("Impact Divergence Index (IDI)", fontsize=18)
    ax.set_title(
        "Judge vs Fan: Divergent Effects by Predictor\n(IDI > 1.96 ⇒ separate tracks justified)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=35, ha="right", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=GRID_ALPHA, linestyle="-")

    fig.suptitle(
        "Rationale for Separating Judge and Fan Tracks",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_variance_decomposition(
    variance_decomp: dict,
    output_path: Path,
) -> None:
    """
    绘制方差分解：两张饼图（Judge / Fan），职业舞伴占比 vs 残差；红蓝主色加透明度，灰色弱化。
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sorted_models = sorted(variance_decomp.keys())
    bar_alpha = 0.88
    for ax, model in zip(axes, sorted_models):
        data = variance_decomp[model]
        sizes = [data["pro_variance_pct"], 100 - data["pro_variance_pct"]]
        labels = [
            f"Pro Dancer\n({data['pro_variance_pct']:.1f}%)",
            f"Residual\n({100 - data['pro_variance_pct']:.1f}%)",
        ]
        main_hex = COLORS["judge"] if model == "judge" else COLORS["fan"]
        main_color = mcolors.to_rgba(main_hex, alpha=bar_alpha)
        colors = [main_color, PIE_RESIDUAL_GRAY]
        explode = (0.04, 0)

        ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            textprops={"fontsize": 17},
        )
        ax.set_title(
            f"{model.capitalize()} Model\nICC = {data['icc']:.3f}",
            fontsize=18,
            fontweight="bold",
        )

    fig.suptitle(
        "Variance Decomposition: Pro Dancer Effect",
        fontsize=20,
        fontweight="bold",
        y=1.02,
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
        alpha=0.92,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        fan_coeffs,
        width,
        label="Fan Share",
        color=COLORS["fan"],
        alpha=0.92,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
    )

    ax.axhline(y=0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=1)
    ax.set_xlabel("Industry (vs Actor/Actress baseline)", fontsize=18)
    ax.set_ylabel("Coefficient", fontsize=18)
    ax.set_title(
        "Industry Effects on Judge Scores and Fan Shares",
        fontsize=20,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(industries, rotation=45, ha="right", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=GRID_ALPHA, linestyle="-")

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
    ax1.set_xlabel("Week", fontsize=18)
    ax1.set_ylabel("Mean Judge Score", fontsize=18)
    ax1.set_title("Learning Curve: Judge Scores", fontsize=20, fontweight="bold")
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
    ax2.set_xlabel("Week", fontsize=18)
    ax2.set_ylabel("Mean Log Fan Share", fontsize=18)
    ax2.set_title("Trend: Fan Shares Over Time", fontsize=20, fontweight="bold")
    ax2.grid(True, alpha=0.25, linestyle=":")

    fig.suptitle(
        "Weekly Trends in Performance and Popularity",
        fontsize=20,
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
        alpha=0.9,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
    )

    ax.axvline(
        x=1.96,
        color="#C62828",
        linestyle="--",
        linewidth=1.5,
        label="p < 0.05",
    )

    ax.set_xlabel("Impact Divergence Index (IDI)", fontsize=18)
    ax.set_ylabel("Variable", fontsize=18)
    ax.set_title(
        "Do Variables Impact Judge Scores and Fan Votes Differently?",
        fontsize=20,
        fontweight="bold",
    )

    legend_elements = [
        Patch(facecolor=COLORS["overall"], alpha=0.9, label="Significant Difference"),
        Patch(
            facecolor=COLORS["Underperformer"],
            alpha=0.9,
            label="No Significant Difference",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11)
    ax.grid(True, axis="x", color=GRID_COLOR, alpha=GRID_ALPHA, linestyle="-")

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
    ax.set_xticklabels(["Tech Boost", "Pop Boost"], fontsize=15)
    ax.set_yticks(np.arange(len(pro_names)))
    ax.set_yticklabels(pro_names, fontsize=11)

    for i in range(len(pro_names)):
        for j in range(2):
            ax.text(
                j, i, f"{data_matrix[i, j]:.2f}", ha="center", va="center", fontsize=10
            )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Random Effect (BLUP)", rotation=270, labelpad=20, fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    ax.set_title(
        "Top 30 Professional Dancers: Random Effects Heatmap",
        fontsize=20,
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
    ax.set_xlabel("Random Effect (BLUP)", fontsize=18)
    ax.set_ylabel("Frequency", fontsize=18)
    ax.set_title("Distribution of Pro Random Effects", fontsize=20, fontweight="bold")
    ax.legend(fontsize=11)
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

    ax.set_xlabel("Judge Model BLUP", fontsize=18)
    ax.set_ylabel("Fan Model BLUP", fontsize=18)
    ax.set_title("Pro Effects Correlation", fontsize=20, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # 计算相关系数
    corr = merged["intercept_judge"].corr(merged["intercept_fan"])
    ax.text(
        0.05,
        0.95,
        f"Correlation: {corr:.3f}",
        transform=ax.transAxes,
        fontsize=15,
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
    ax.set_xlabel("Celebrity Age", fontsize=18)
    ax.set_ylabel("Predicted Effect", fontsize=18)
    ax.set_title("Age Effect on Performance", fontsize=20, fontweight="bold")
    ax.legend(fontsize=11)
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
    ax.set_xlabel("Celebrity Age", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.set_title("Age Distribution in Sample", fontsize=20, fontweight="bold")
    ax.legend(fontsize=11)
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
    ax.set_xticklabels(categories, fontsize=15)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["20%", "40%", "60%", "80%"], fontsize=13)
    ax.grid(True, alpha=0.3)

    # 添加值标签
    for angle, value, cat in zip(angles[:-1], values[:-1], categories):
        ax.text(
            angle,
            value + 0.05,
            f"{value:.1%}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=COLORS["overall"],
        )

    plt.title(
        "Variance Decomposition (Radar Chart)",
        fontsize=20,
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
    ax.set_yticklabels(top_20["pro_id"], fontsize=13)
    ax.set_xlabel("Combined Score (Standardized)", fontsize=18)
    ax.set_title(
        "Top 20 Professional Dancers (Combined Score)",
        fontsize=20,
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    legend_elements = [
        Patch(facecolor=COLORS["Kingmaker"], alpha=0.8, label="Kingmaker"),
        Patch(facecolor=COLORS["Technician"], alpha=0.8, label="Technician"),
        Patch(facecolor=COLORS["Fan Favorite"], alpha=0.8, label="Fan Favorite"),
        Patch(facecolor=COLORS["Underperformer"], alpha=0.8, label="Underperformer"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11)

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
