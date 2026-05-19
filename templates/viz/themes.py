"""
主题与配色模块
==============
"""

from typing import Literal

import plotly.graph_objects as go
import plotly.io as pio

# Nature/Science 风格配色
NATURE_COLORS = [
    "#E64B35",  # Nature Red
    "#4DBBD5",  # Nature Blue
    "#00A087",  # Nature Green
    "#3C5488",  # Nature Dark Blue
    "#F39B7F",  # Nature Light Red
    "#8491B4",  # Nature Grey Blue
    "#91D1C2",  # Nature Light Green
    "#DC0000",  # Nature Bright Red
    "#7E6148",  # Nature Brown
    "#B09C85",  # Nature Tan
]

# 现代科研配色
MODERN_COLORS = [
    "#2E86AB",  # 深蓝
    "#A23B72",  # 玫红
    "#F18F01",  # 橙色
    "#C73E1D",  # 砖红
    "#95C623",  # 黄绿
    "#5C4D7D",  # 紫色
    "#00A896",  # 青色
    "#F7B538",  # 金黄
]

# 高对比度配色 (适合论文打印)
HIGH_CONTRAST_COLORS = [
    "#000000",  # 黑
    "#E69F00",  # 橙
    "#56B4E9",  # 天蓝
    "#009E73",  # 蓝绿
    "#F0E442",  # 黄
    "#0072B2",  # 蓝
    "#D55E00",  # 朱红
    "#CC79A7",  # 粉紫
]

# 渐变色系
SEQUENTIAL_BLUES = [
    "#f7fbff",
    "#deebf7",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#084594",
]
SEQUENTIAL_REDS = [
    "#fff5f0",
    "#fee0d2",
    "#fcbba1",
    "#fc9272",
    "#fb6a4a",
    "#ef3b2c",
    "#cb181d",
    "#99000d",
]
DIVERGING_RdBu = [
    "#b2182b",
    "#d6604d",
    "#f4a582",
    "#fddbc7",
    "#d1e5f0",
    "#92c5de",
    "#4393c3",
    "#2166ac",
]


def get_color_palette(
    name: Literal[
        "nature", "modern", "high_contrast", "blues", "reds", "diverging"
    ] = "nature",
    n_colors: int | None = None,
) -> list[str]:
    """
    获取配色方案

    Parameters
    ----------
    name : str, default "nature"
        配色方案名称:
        - "nature": Nature/Science 期刊风格
        - "modern": 现代科研风格
        - "high_contrast": 高对比度 (适合打印)
        - "blues": 蓝色渐变
        - "reds": 红色渐变
        - "diverging": 红蓝发散
    n_colors : int, optional
        返回的颜色数量

    Returns
    -------
    list[str]
        颜色列表

    Examples
    --------
    >>> colors = get_color_palette("nature", n_colors=5)
    """
    palettes = {
        "nature": NATURE_COLORS,
        "modern": MODERN_COLORS,
        "high_contrast": HIGH_CONTRAST_COLORS,
        "blues": SEQUENTIAL_BLUES,
        "reds": SEQUENTIAL_REDS,
        "diverging": DIVERGING_RdBu,
    }

    colors = palettes.get(name, NATURE_COLORS)

    if n_colors is not None:
        # 循环使用颜色
        colors = [colors[i % len(colors)] for i in range(n_colors)]

    return colors


def apply_theme(
    fig: go.Figure,
    *,
    theme: Literal["nature", "modern", "minimal", "dark"] = "nature",
    title_size: int = 18,
    axis_size: int = 14,
    tick_size: int = 12,
    font_family: str = "Arial, sans-serif",
    show_grid: bool = True,
    transparent_bg: bool = False,
) -> go.Figure:
    """
    应用预设主题到图表

    Parameters
    ----------
    fig : go.Figure
        Plotly 图表对象
    theme : str, default "nature"
        主题名称
    title_size : int, default 18
        标题字号
    axis_size : int, default 14
        轴标签字号
    tick_size : int, default 12
        刻度字号
    font_family : str
        字体
    show_grid : bool, default True
        是否显示网格
    transparent_bg : bool, default False
        是否透明背景

    Returns
    -------
    go.Figure
        应用主题后的图表

    Examples
    --------
    >>> fig = apply_theme(fig, theme="nature")
    """
    # 基础布局
    layout_updates = {
        "font": {
            "family": font_family,
            "size": tick_size,
        },
        "title": {
            "font": {"size": title_size, "family": font_family},
            "x": 0.5,
            "xanchor": "center",
        },
        "xaxis": {
            "title_font": {"size": axis_size},
            "tickfont": {"size": tick_size},
            "showgrid": show_grid,
            "gridwidth": 1,
            "zeroline": False,
        },
        "yaxis": {
            "title_font": {"size": axis_size},
            "tickfont": {"size": tick_size},
            "showgrid": show_grid,
            "gridwidth": 1,
            "zeroline": False,
        },
        "legend": {
            "font": {"size": tick_size},
            "bgcolor": "rgba(255,255,255,0.8)",
            "borderwidth": 1,
        },
        "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
    }

    # 主题特定设置
    if theme == "nature":
        layout_updates["plot_bgcolor"] = "white"
        layout_updates["paper_bgcolor"] = (
            "white" if not transparent_bg else "rgba(0,0,0,0)"
        )
        layout_updates["xaxis"]["gridcolor"] = "#E5E5E5"
        layout_updates["yaxis"]["gridcolor"] = "#E5E5E5"
        layout_updates["xaxis"]["linecolor"] = "#333333"
        layout_updates["yaxis"]["linecolor"] = "#333333"
        layout_updates["xaxis"]["linewidth"] = 1.5
        layout_updates["yaxis"]["linewidth"] = 1.5

    elif theme == "modern":
        layout_updates["plot_bgcolor"] = "#FAFAFA"
        layout_updates["paper_bgcolor"] = (
            "#FAFAFA" if not transparent_bg else "rgba(0,0,0,0)"
        )
        layout_updates["xaxis"]["gridcolor"] = "#E0E0E0"
        layout_updates["yaxis"]["gridcolor"] = "#E0E0E0"

    elif theme == "minimal":
        layout_updates["plot_bgcolor"] = "white"
        layout_updates["paper_bgcolor"] = (
            "white" if not transparent_bg else "rgba(0,0,0,0)"
        )
        layout_updates["xaxis"]["showgrid"] = False
        layout_updates["yaxis"]["showgrid"] = False
        layout_updates["xaxis"]["showline"] = True
        layout_updates["yaxis"]["showline"] = True

    elif theme == "dark":
        layout_updates["plot_bgcolor"] = "#1E1E1E"
        layout_updates["paper_bgcolor"] = (
            "#1E1E1E" if not transparent_bg else "rgba(0,0,0,0)"
        )
        layout_updates["font"]["color"] = "#E0E0E0"
        layout_updates["xaxis"]["gridcolor"] = "#333333"
        layout_updates["yaxis"]["gridcolor"] = "#333333"
        layout_updates["xaxis"]["linecolor"] = "#555555"
        layout_updates["yaxis"]["linecolor"] = "#555555"

    fig.update_layout(**layout_updates)

    return fig


# 注册自定义模板
def _register_custom_templates():
    """注册自定义 Plotly 模板"""

    # Nature 模板
    nature_template = go.layout.Template()
    nature_template.layout = go.Layout(
        font=dict(family="Arial, sans-serif", size=12, color="#333333"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        colorway=NATURE_COLORS,
        xaxis=dict(
            showgrid=True,
            gridcolor="#E5E5E5",
            gridwidth=1,
            linecolor="#333333",
            linewidth=1.5,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E5E5E5",
            gridwidth=1,
            linecolor="#333333",
            linewidth=1.5,
            zeroline=False,
        ),
    )
    pio.templates["nature"] = nature_template

    # Modern 模板
    modern_template = go.layout.Template()
    modern_template.layout = go.Layout(
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FAFAFA",
        colorway=MODERN_COLORS,
        xaxis=dict(showgrid=True, gridcolor="#E0E0E0"),
        yaxis=dict(showgrid=True, gridcolor="#E0E0E0"),
    )
    pio.templates["modern"] = modern_template


# 初始化时注册模板
_register_custom_templates()
