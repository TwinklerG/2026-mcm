"""
Matplotlib/Seaborn 可视化后端
=============================

提供科研出版级高质量静态图表，适合论文导出。

特点
----
- 高 DPI 矢量图输出（PDF/SVG/PNG）
- Nature/Science 期刊风格配色
- 精细的排版控制（字体、刻度、图例）
- 支持 LaTeX 数学公式渲染
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure

# 科研配色方案
NATURE_COLORS = [
    "#E64B35",  # Nature Red
    "#4DBBD5",  # Nature Blue
    "#00A087",  # Nature Green
    "#3C5488",  # Nature Dark Blue
    "#F39B7F",  # Nature Light Red
    "#8491B4",  # Nature Grey Blue
    "#91D1C2",  # Nature Light Green
    "#DC0000",  # Nature Bright Red
]

SCIENCE_COLORS = [
    "#3B4992",  # Science Blue
    "#EE0000",  # Science Red
    "#008B45",  # Science Green
    "#631879",  # Science Purple
    "#008280",  # Science Teal
    "#BB0021",  # Science Dark Red
    "#5F559B",  # Science Violet
    "#A20056",  # Science Magenta
]

COLORBLIND_SAFE = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Green
    "#CC79A7",  # Pink
    "#F0E442",  # Yellow
    "#56B4E9",  # Light Blue
    "#D55E00",  # Vermillion
    "#000000",  # Black
]


def setup_style(
    style: Literal["nature", "science", "minimal"] = "nature",
    font_family: str = "Arial",
    font_size: int = 11,
    use_latex: bool = False,
) -> None:
    """
    配置全局绑图样式

    Parameters
    ----------
    style : str, default "nature"
        样式预设："nature"、"science"、"minimal"
    font_family : str, default "Arial"
        字体族
    font_size : int, default 11
        基础字号
    use_latex : bool, default False
        是否使用 LaTeX 渲染文本（需要安装 LaTeX）
    """
    # 重置为默认
    plt.rcdefaults()

    # 通用设置
    plt.rcParams.update(
        {
            # 字体
            "font.family": font_family,
            "font.size": font_size,
            "axes.labelsize": font_size + 1,
            "axes.titlesize": font_size + 2,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            # 线条
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            # 网格
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            # 图例
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            # 保存
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            # 布局
            "figure.autolayout": True,
            "figure.figsize": (7, 5),
        }
    )

    if style == "nature":
        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.prop_cycle": plt.cycler(color=NATURE_COLORS),
            }
        )
    elif style == "science":
        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.prop_cycle": plt.cycler(color=SCIENCE_COLORS),
            }
        )
    elif style == "minimal":
        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.grid": False,
                "axes.prop_cycle": plt.cycler(color=COLORBLIND_SAFE),
            }
        )

    if use_latex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
            }
        )


def get_colors(
    palette: Literal["nature", "science", "colorblind"] = "nature",
    n: int | None = None,
) -> list[str]:
    """
    获取配色方案

    Parameters
    ----------
    palette : str, default "nature"
        配色方案名称
    n : int, optional
        返回颜色数量

    Returns
    -------
    list[str]
        颜色列表
    """
    palettes = {
        "nature": NATURE_COLORS,
        "science": SCIENCE_COLORS,
        "colorblind": COLORBLIND_SAFE,
    }
    colors = palettes.get(palette, NATURE_COLORS)

    if n is not None:
        colors = [colors[i % len(colors)] for i in range(n)]

    return colors


def line_plot(
    df: pl.DataFrame,
    x: str,
    y: str | list[str],
    *,
    color: str | None = None,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_markers: bool = True,
    style: str = "nature",
    figsize: tuple[float, float] = (7, 5),
    legend_loc: str = "best",
    colors: list[str] | None = None,
) -> Figure:
    """
    绑制折线图

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    x : str
        X 轴列名
    y : str | list[str]
        Y 轴列名，支持多条线
    color : str, optional
        分组颜色列
    title : str
        图表标题
    xlabel, ylabel : str, optional
        轴标签
    show_markers : bool, default True
        是否显示数据点标记
    style : str, default "nature"
        绘图风格
    figsize : tuple, default (7, 5)
        图表尺寸（英寸）
    legend_loc : str, default "best"
        图例位置
    colors : list[str], optional
        自定义颜色列表

    Returns
    -------
    Figure
        Matplotlib 图表对象
    """
    setup_style(style)
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = get_colors(style)

    y_cols = [y] if isinstance(y, str) else y
    marker = "o" if show_markers else None

    if color is not None:
        groups = df[color].unique().to_list()
        for i, group in enumerate(groups):
            group_df = df.filter(pl.col(color) == group)
            for j, y_col in enumerate(y_cols):
                label = f"{group}" if len(y_cols) == 1 else f"{group} - {y_col}"
                ax.plot(
                    group_df[x].to_numpy(),
                    group_df[y_col].to_numpy(),
                    marker=marker,
                    color=colors[(i * len(y_cols) + j) % len(colors)],
                    label=label,
                    markersize=5,
                )
    else:
        for i, y_col in enumerate(y_cols):
            ax.plot(
                df[x].to_numpy(),
                df[y_col].to_numpy(),
                marker=marker,
                color=colors[i % len(colors)],
                label=y_col,
                markersize=5,
            )

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or (y_cols[0] if len(y_cols) == 1 else "Value"))
    ax.set_title(title)

    if len(y_cols) > 1 or color is not None:
        ax.legend(loc=legend_loc)

    plt.tight_layout()
    return fig


def scatter_plot(
    df: pl.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    size: str | None = None,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_trendline: bool = True,
    show_r2: bool = True,
    style: str = "nature",
    figsize: tuple[float, float] = (7, 5),
    alpha: float = 0.7,
    colors: list[str] | None = None,
) -> Figure:
    """
    绘制散点图

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    x : str
        X 轴列名
    y : str
        Y 轴列名
    color : str, optional
        分组颜色列
    size : str, optional
        点大小列
    title : str
        图表标题
    xlabel, ylabel : str, optional
        轴标签
    show_trendline : bool, default True
        是否显示趋势线
    show_r2 : bool, default True
        是否显示 R² 值
    style : str, default "nature"
        绘图风格
    figsize : tuple, default (7, 5)
        图表尺寸
    alpha : float, default 0.7
        透明度
    colors : list[str], optional
        自定义颜色列表

    Returns
    -------
    Figure
        Matplotlib 图表对象
    """
    setup_style(style)
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = get_colors(style)

    # 准备数据
    valid_df = df.select([x, y]).drop_nulls()
    x_arr = valid_df[x].to_numpy()
    y_arr = valid_df[y].to_numpy()

    # 计算 R²
    r2 = None
    if show_r2 and len(x_arr) > 1:
        correlation = np.corrcoef(x_arr, y_arr)[0, 1]
        r2 = correlation**2

    # 点大小
    s = df[size].to_numpy() * 10 if size else 50

    if color is not None:
        groups = df[color].unique().to_list()
        for i, group in enumerate(groups):
            group_df = df.filter(pl.col(color) == group)
            group_s = group_df[size].to_numpy() * 10 if size else 50
            ax.scatter(
                group_df[x].to_numpy(),
                group_df[y].to_numpy(),
                c=colors[i % len(colors)],
                s=group_s,
                alpha=alpha,
                label=str(group),
                edgecolors="white",
                linewidth=0.5,
            )
        ax.legend()
    else:
        ax.scatter(
            x_arr,
            y_arr,
            c=colors[0],
            s=s,
            alpha=alpha,
            edgecolors="white",
            linewidth=0.5,
        )

    # 趋势线
    if show_trendline and len(x_arr) > 1:
        z = np.polyfit(x_arr, y_arr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
        label = f"Trendline (R² = {r2:.3f})" if r2 else "Trendline"
        ax.plot(x_line, p(x_line), "--", color="#333333", linewidth=1.5, label=label)
        ax.legend()

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)

    title_text = title
    if show_r2 and r2 is not None and not show_trendline:
        title_text = f"{title} (R² = {r2:.4f})" if title else f"R² = {r2:.4f}"
    ax.set_title(title_text)

    plt.tight_layout()
    return fig


def bar_plot(
    df: pl.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    orientation: Literal["v", "h"] = "v",
    show_values: bool = True,
    sort_values: bool = True,
    style: str = "nature",
    figsize: tuple[float, float] = (7, 5),
    colors: list[str] | None = None,
) -> Figure:
    """
    绘制柱状图

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    x : str
        X 轴列名（类别）
    y : str
        Y 轴列名（数值）
    color : str, optional
        分组颜色列
    title : str
        图表标题
    xlabel, ylabel : str, optional
        轴标签
    orientation : str, default "v"
        方向："v" 垂直，"h" 水平
    show_values : bool, default True
        是否在柱顶显示数值
    sort_values : bool, default True
        是否按值排序
    style : str, default "nature"
        绘图风格
    figsize : tuple, default (7, 5)
        图表尺寸
    colors : list[str], optional
        自定义颜色列表

    Returns
    -------
    Figure
        Matplotlib 图表对象
    """
    setup_style(style)
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = get_colors(style)

    data = df.clone()
    if sort_values and color is None:
        data = data.sort(y, descending=True)

    if color is not None:
        # 分组柱状图
        groups = data[color].unique().to_list()
        categories = data[x].unique().to_list()
        n_groups = len(groups)
        bar_width = 0.8 / n_groups
        positions = np.arange(len(categories))

        for i, group in enumerate(groups):
            group_df = data.filter(pl.col(color) == group)
            # 确保顺序一致
            values = []
            for cat in categories:
                val = group_df.filter(pl.col(x) == cat)[y].to_list()
                values.append(val[0] if val else 0)

            offset = (i - n_groups / 2 + 0.5) * bar_width
            if orientation == "v":
                bars = ax.bar(
                    positions + offset,
                    values,
                    bar_width,
                    label=str(group),
                    color=colors[i % len(colors)],
                )
                if show_values:
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(
                            f"{height:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
            else:
                bars = ax.barh(
                    positions + offset,
                    values,
                    bar_width,
                    label=str(group),
                    color=colors[i % len(colors)],
                )

        if orientation == "v":
            ax.set_xticks(positions)
            ax.set_xticklabels(categories)
        else:
            ax.set_yticks(positions)
            ax.set_yticklabels(categories)

        ax.legend()
    else:
        x_vals = data[x].to_list()
        y_vals = data[y].to_numpy()

        if orientation == "v":
            bars = ax.bar(x_vals, y_vals, color=colors[0])
            if show_values:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            plt.xticks(rotation=45, ha="right")
        else:
            bars = ax.barh(x_vals, y_vals, color=colors[0])
            if show_values:
                for bar in bars:
                    width = bar.get_width()
                    ax.annotate(
                        f"{width:.1f}",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=8,
                    )

    ax.set_xlabel(xlabel or (x if orientation == "v" else y))
    ax.set_ylabel(ylabel or (y if orientation == "v" else x))
    ax.set_title(title)

    plt.tight_layout()
    return fig


def heatmap(
    df: pl.DataFrame,
    *,
    x: str | None = None,
    y: str | None = None,
    z: str | None = None,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_values: bool = True,
    cmap: str = "RdBu_r",
    style: str = "nature",
    figsize: tuple[float, float] = (8, 6),
    vmin: float | None = None,
    vmax: float | None = None,
) -> Figure:
    """
    绘制热力图

    Parameters
    ----------
    df : pl.DataFrame
        数据框（相关矩阵或透视表）
    x, y, z : str, optional
        列名（如果是长格式数据）
    title : str
        图表标题
    xlabel, ylabel : str, optional
        轴标签
    show_values : bool, default True
        是否显示数值
    cmap : str, default "RdBu_r"
        配色方案
    style : str, default "nature"
        绘图风格
    figsize : tuple, default (8, 6)
        图表尺寸
    vmin, vmax : float, optional
        颜色范围

    Returns
    -------
    Figure
        Matplotlib 图表对象
    """
    setup_style(style)
    fig, ax = plt.subplots(figsize=figsize)

    # 准备数据
    if z is None:
        if "variable" in df.columns:
            x_labels = df["variable"].to_list()
            numeric_cols = [c for c in df.columns if c != "variable"]
            z_data = df.select(numeric_cols).to_numpy()
            y_labels = numeric_cols
        else:
            numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
            data_np = df.select(numeric_cols).to_numpy()
            z_data = np.corrcoef(data_np, rowvar=False)
            x_labels = numeric_cols
            y_labels = numeric_cols
            if vmin is None:
                vmin = -1
            if vmax is None:
                vmax = 1
    else:
        pivot_df = df.pivot(on=x, index=y, values=z)
        y_labels = pivot_df[y].to_list()
        x_labels = [c for c in pivot_df.columns if c != y]
        z_data = pivot_df.select(x_labels).to_numpy()

    # 绘制热力图
    im = ax.imshow(z_data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # 设置刻度
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # 旋转 x 轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 显示数值
    if show_values:
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                val = z_data[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    ax.set_title(title or "Correlation Matrix")

    plt.tight_layout()
    return fig


def box_plot(
    df: pl.DataFrame,
    y: str,
    *,
    x: str | None = None,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_points: bool = True,
    style: str = "nature",
    figsize: tuple[float, float] = (7, 5),
    colors: list[str] | None = None,
) -> Figure:
    """
    绘制箱线图

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    y : str
        数值列名
    x : str, optional
        分组列名
    title : str
        图表标题
    xlabel, ylabel : str, optional
        轴标签
    show_points : bool, default True
        是否显示数据点
    style : str, default "nature"
        绘图风格
    figsize : tuple, default (7, 5)
        图表尺寸
    colors : list[str], optional
        自定义颜色列表

    Returns
    -------
    Figure
        Matplotlib 图表对象
    """
    setup_style(style)
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = get_colors(style)

    if x is not None:
        groups = df[x].unique().sort().to_list()
        data = [df.filter(pl.col(x) == g)[y].to_numpy() for g in groups]
        positions = range(1, len(groups) + 1)

        bp = ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,
            widths=0.6,
        )

        # 设置颜色
        for i, (patch, median) in enumerate(zip(bp["boxes"], bp["medians"])):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
            median.set_color("black")

        # 显示数据点
        if show_points:
            for i, d in enumerate(data):
                jitter = np.random.normal(0, 0.04, size=len(d))
                ax.scatter(
                    np.ones(len(d)) * (i + 1) + jitter,
                    d,
                    alpha=0.4,
                    s=20,
                    c=colors[i % len(colors)],
                    edgecolors="white",
                    linewidth=0.5,
                )

        ax.set_xticks(positions)
        ax.set_xticklabels(groups)
    else:
        bp = ax.boxplot(
            [df[y].drop_nulls().to_numpy()],
            patch_artist=True,
            widths=0.6,
        )
        bp["boxes"][0].set_facecolor(colors[0])
        bp["boxes"][0].set_alpha(0.7)
        bp["medians"][0].set_color("black")

        if show_points:
            data = df[y].drop_nulls().to_numpy()
            jitter = np.random.normal(0, 0.04, size=len(data))
            ax.scatter(
                np.ones(len(data)) + jitter,
                data,
                alpha=0.4,
                s=20,
                c=colors[0],
                edgecolors="white",
                linewidth=0.5,
            )

    ax.set_xlabel(xlabel or (x or ""))
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)

    plt.tight_layout()
    return fig


def histogram(
    df: pl.DataFrame,
    x: str,
    *,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str = "Frequency",
    bins: int = 30,
    show_kde: bool = True,
    show_mean: bool = True,
    show_median: bool = True,
    style: str = "nature",
    figsize: tuple[float, float] = (7, 5),
    colors: list[str] | None = None,
    alpha: float = 0.7,
) -> Figure:
    """
    绘制直方图

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    x : str
        数值列名
    title : str
        图表标题
    xlabel : str, optional
        X 轴标签
    ylabel : str, default "Frequency"
        Y 轴标签
    bins : int, default 30
        直方图区间数
    show_kde : bool, default True
        是否显示核密度估计曲线
    show_mean : bool, default True
        是否显示均值线
    show_median : bool, default True
        是否显示中位数线
    style : str, default "nature"
        绘图风格
    figsize : tuple, default (7, 5)
        图表尺寸
    colors : list[str], optional
        自定义颜色列表
    alpha : float, default 0.7
        透明度

    Returns
    -------
    Figure
        Matplotlib 图表对象
    """
    setup_style(style)
    fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = get_colors(style)

    data = df[x].drop_nulls().to_numpy()

    # 绘制直方图
    n, bins_edges, patches = ax.hist(
        data, bins=bins, color=colors[0], alpha=alpha, edgecolor="white", linewidth=0.5
    )

    # KDE 曲线
    if show_kde:
        try:
            from scipy import stats

            kde = stats.gaussian_kde(data)
            x_kde = np.linspace(data.min(), data.max(), 200)
            y_kde = kde(x_kde) * len(data) * (bins_edges[1] - bins_edges[0])
            ax.plot(x_kde, y_kde, color=colors[1], linewidth=2, label="KDE")
        except ImportError:
            pass

    # 均值线
    if show_mean:
        mean_val = np.mean(data)
        ax.axvline(
            mean_val,
            color="#E64B35",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_val:.2f}",
        )

    # 中位数线
    if show_median:
        median_val = np.median(data)
        ax.axvline(
            median_val,
            color="#00A087",
            linestyle=":",
            linewidth=1.5,
            label=f"Median: {median_val:.2f}",
        )

    if show_mean or show_median or show_kde:
        ax.legend()

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    return fig


def radar_plot(
    df: pl.DataFrame,
    categories: list[str],
    *,
    name_col: str | None = None,
    title: str = "",
    fill: bool = True,
    style: str = "nature",
    figsize: tuple[float, float] = (7, 7),
    colors: list[str] | None = None,
) -> Figure:
    """
    绘制雷达图

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    categories : list[str]
        维度列名列表
    name_col : str, optional
        分组名称列
    title : str
        图表标题
    fill : bool, default True
        是否填充
    style : str, default "nature"
        绘图风格
    figsize : tuple, default (7, 7)
        图表尺寸
    colors : list[str], optional
        自定义颜色列表

    Returns
    -------
    Figure
        Matplotlib 图表对象
    """
    setup_style(style)

    if colors is None:
        colors = get_colors(style)

    # 角度
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

    if name_col is not None:
        names = df[name_col].unique().to_list()
        for i, name in enumerate(names):
            row = df.filter(pl.col(name_col) == name)
            values = [row[cat].to_list()[0] for cat in categories]
            values += values[:1]  # 闭合

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                color=colors[i % len(colors)],
                label=str(name),
            )
            if fill:
                ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    else:
        for i in range(len(df)):
            row = df.row(i, named=True)
            values = [row[cat] for cat in categories]
            values += values[:1]

            name = row.get("name", f"Series {i + 1}")
            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                color=colors[i % len(colors)],
                label=str(name),
            )
            if fill:
                ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

    # 设置类别标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    ax.set_title(title, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    return fig


def save_figure(
    fig: Figure,
    filename: str,
    *,
    output_dir: str | Path = "figures",
    formats: list[str] | None = None,
    dpi: int = 300,
) -> list[Path]:
    """
    保存图表到文件

    Parameters
    ----------
    fig : Figure
        Matplotlib 图表对象
    filename : str
        文件名（不含扩展名）
    output_dir : str | Path, default "figures"
        输出目录
    formats : list[str], optional
        输出格式列表，默认 ["png", "pdf"]
    dpi : int, default 300
        分辨率

    Returns
    -------
    list[Path]
        保存的文件路径列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if formats is None:
        formats = ["png", "pdf"]

    saved_paths = []
    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
        saved_paths.append(filepath)

    return saved_paths


def close_all() -> None:
    """关闭所有图表"""
    plt.close("all")
