"""
统一可视化主题配置。

所有三个任务的图表都应使用此配置，确保风格一致性。

使用方法:
    from shared.viz_theme import apply_theme, COLORS, SYSTEM_COLORS, save_figure
    apply_theme()
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保可重复性
np.random.seed(42)
random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# 颜色方案
# ═══════════════════════════════════════════════════════════════════════════════

# 主色调
COLORS = {
    # 核心双轨颜色
    "judge": "#2E86AB",  # 深蓝 - 评委/技术
    "fan": "#A23B72",  # 紫红 - 粉丝/流量
    # 状态颜色
    "success": "#2ECC71",  # 绿色 - 成功/正向
    "warning": "#E74C3C",  # 红色 - 警告/负向
    "neutral": "#95A5A6",  # 灰色 - 中性
    "highlight": "#F18F01",  # 橙色 - 高亮
    # 时间段
    "pre_s28": "#2E86AB",  # S1-S27（使用深蓝）
    "post_s28": "#A23B72",  # S28+（使用紫红）
    # 验证结果
    "strict": "#2E86AB",  # 严格准确
    "bottom2": "#A23B72",  # Bottom-2 Only
}

# 投票系统颜色
SYSTEM_COLORS = {
    "Rank-based": "#C73E1D",  # 红色 - 排名法
    "Percentage-based": "#2E86AB",  # 深蓝 - 百分比法
    "PTFS": "#2ECC71",  # 绿色 - 新系统
}

# 职业舞伴类别颜色
PRO_CATEGORY_COLORS = {
    "Kingmaker": "#2ECC71",  # 绿色 - 造王者
    "Technician": "#2E86AB",  # 深蓝 - 技术型
    "Fan Favorite": "#A23B72",  # 紫红 - 流量型
    "Underperformer": "#95A5A6",  # 灰色 - 表现不佳
}

# 行业颜色（按观赛顺序）
INDUSTRY_COLORS = {
    "Actor": "#2E86AB",
    "Athlete": "#C73E1D",
    "Singer": "#F18F01",
    "TV": "#9B59B6",
    "Model": "#1ABC9C",
    "Entertainment": "#E74C3C",
    "Other": "#95A5A6",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 主题设置
# ═══════════════════════════════════════════════════════════════════════════════

THEME_PARAMS = {
    # 字体
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # 分辨率
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    # 样式
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    # 图例
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.8",
    # SVG 确定性
    "svg.hashsalt": "42",
}


def apply_theme() -> None:
    """应用统一主题到 matplotlib。"""
    plt.rcParams.update(THEME_PARAMS)


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    formats: list[str] | None = None,
) -> None:
    """
    保存图表为多种格式。

    Parameters
    ----------
    fig : plt.Figure
        图表对象
    output_path : Path
        输出路径（不含扩展名）
    formats : list[str], optional
        输出格式列表，默认 ["png", "pdf"]
    """
    if formats is None:
        formats = ["png", "pdf"]

    output_path = Path(output_path)

    for fmt in formats:
        path = output_path.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 常用图表元素
# ═══════════════════════════════════════════════════════════════════════════════


def add_rule_change_marker(
    ax: plt.Axes,
    x: float = 27.5,
    label: str = "Rule Change (S28)",
) -> None:
    """添加 S28 规则变化的垂直标记线。"""
    ax.axvline(
        x,
        color=COLORS["highlight"],
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=label,
    )


def add_threshold_line(
    ax: plt.Axes,
    y: float,
    label: str | None = None,
    color: str = "gray",
) -> None:
    """添加水平阈值线。"""
    ax.axhline(y, color=color, linestyle=":", linewidth=1.5, alpha=0.6, label=label)


def annotate_key_point(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    xytext: tuple[float, float] | None = None,
    **kwargs: Any,
) -> None:
    """添加带箭头的注释。"""
    default_kwargs = {
        "fontsize": 9,
        "arrowprops": {"arrowstyle": "->", "color": COLORS["neutral"]},
        "bbox": {
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": COLORS["neutral"],
            "alpha": 0.9,
        },
    }
    default_kwargs.update(kwargs)

    if xytext is None:
        xytext = (xy[0] + 2, xy[1] + 0.1)

    ax.annotate(text, xy=xy, xytext=xytext, **default_kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# 预设图表尺寸
# ═══════════════════════════════════════════════════════════════════════════════

FIGURE_SIZES = {
    "single": (8, 5),  # 单图
    "wide": (14, 5),  # 宽图（时间序列）
    "double": (12, 5),  # 双图并排
    "triple": (15, 5),  # 三图并排
    "square": (8, 8),  # 正方形（散点图）
    "heatmap": (10, 8),  # 热力图
}


def create_figure(
    size: str = "single",
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Any,
) -> tuple[plt.Figure, Any]:
    """
    创建标准尺寸的图表。

    Parameters
    ----------
    size : str
        图表尺寸预设名称
    nrows, ncols : int
        子图行列数
    **kwargs
        传递给 plt.subplots 的额外参数

    Returns
    -------
    tuple[plt.Figure, Any]
        图表和轴对象
    """
    apply_theme()
    figsize = FIGURE_SIZES.get(size, FIGURE_SIZES["single"])
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
