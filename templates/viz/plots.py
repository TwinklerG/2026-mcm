"""
核心绑图函数
============

提供各类科研级可视化图表
"""

from typing import Literal

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from .themes import apply_theme, get_color_palette


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
    show_confidence: bool = False,
    ci_lower: str | None = None,
    ci_upper: str | None = None,
    theme: str = "nature",
    width: int = 900,
    height: int = 600,
    colors: list[str] | None = None,
) -> go.Figure:
    """
    绘制折线图（趋势分析、时序预测）

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    x : str
        X 轴列名
    y : str | list[str]
        Y 轴列名，可以是多个
    color : str, optional
        分组颜色列
    title : str
        图表标题
    xlabel, ylabel : str, optional
        轴标签
    show_markers : bool, default True
        是否显示数据点
    show_confidence : bool, default False
        是否显示置信区间
    ci_lower, ci_upper : str, optional
        置信区间下界和上界列名
    theme : str, default "nature"
        主题
    width, height : int
        图表尺寸
    colors : list[str], optional
        自定义颜色列表

    Returns
    -------
    go.Figure
        Plotly 图表对象
    """
    fig = go.Figure()

    if colors is None:
        colors = get_color_palette("nature")

    y_cols = [y] if isinstance(y, str) else y

    if color is not None:
        groups = df[color].unique().to_list()
        for i, group in enumerate(groups):
            group_df = df.filter(pl.col(color) == group)
            for j, y_col in enumerate(y_cols):
                fig.add_trace(
                    go.Scatter(
                        x=group_df[x].to_list(),
                        y=group_df[y_col].to_list(),
                        mode="lines+markers" if show_markers else "lines",
                        name=f"{group}" if len(y_cols) == 1 else f"{group} - {y_col}",
                        line=dict(
                            color=colors[(i * len(y_cols) + j) % len(colors)], width=2
                        ),
                        marker=dict(size=6),
                    )
                )
    else:
        for i, y_col in enumerate(y_cols):
            fig.add_trace(
                go.Scatter(
                    x=df[x].to_list(),
                    y=df[y_col].to_list(),
                    mode="lines+markers" if show_markers else "lines",
                    name=y_col,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                )
            )

    if show_confidence and ci_lower and ci_upper:
        x_vals = df[x].to_list()
        fig.add_trace(
            go.Scatter(
                x=x_vals + x_vals[::-1],
                y=df[ci_upper].to_list() + df[ci_lower].reverse().to_list(),
                fill="toself",
                fillcolor="rgba(68, 68, 68, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel or x,
        yaxis_title=ylabel or (y_cols[0] if len(y_cols) == 1 else "Value"),
        width=width,
        height=height,
        hovermode="x unified",
    )

    return apply_theme(fig, theme=theme)


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
    theme: str = "nature",
    width: int = 900,
    height: int = 600,
    colors: list[str] | None = None,
    barmode: Literal["group", "stack", "relative"] = "group",
) -> go.Figure:
    """
    绘制柱状图（分组对比）

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    x : str
        X 轴列名 (类别)
    y : str
        Y 轴列名 (数值)
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
    theme : str, default "nature"
        主题
    width, height : int
        图表尺寸
    colors : list[str], optional
        自定义颜色列表
    barmode : str, default "group"
        柱状图模式

    Returns
    -------
    go.Figure
        Plotly 图表对象
    """
    if colors is None:
        colors = get_color_palette("nature")

    data = df.clone()
    if sort_values and color is None:
        data = data.sort(y, descending=(orientation == "v"))

    fig = go.Figure()

    if color is not None:
        groups = data[color].unique().to_list()
        for i, group in enumerate(groups):
            group_df = data.filter(pl.col(color) == group)
            x_vals = group_df[x].to_list()
            y_vals = group_df[y].to_list()
            text_vals = [round(v, 2) for v in y_vals] if show_values else None

            if orientation == "v":
                fig.add_trace(
                    go.Bar(
                        x=x_vals,
                        y=y_vals,
                        name=str(group),
                        marker_color=colors[i % len(colors)],
                        text=text_vals,
                        textposition="outside" if show_values else None,
                    )
                )
            else:
                fig.add_trace(
                    go.Bar(
                        y=x_vals,
                        x=y_vals,
                        name=str(group),
                        orientation="h",
                        marker_color=colors[i % len(colors)],
                        text=text_vals,
                        textposition="outside" if show_values else None,
                    )
                )
    else:
        x_vals = data[x].to_list()
        y_vals = data[y].to_list()
        text_vals = [round(v, 2) for v in y_vals] if show_values else None

        if orientation == "v":
            fig.add_trace(
                go.Bar(
                    x=x_vals,
                    y=y_vals,
                    marker_color=colors[0],
                    text=text_vals,
                    textposition="outside" if show_values else None,
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    y=x_vals,
                    x=y_vals,
                    orientation="h",
                    marker_color=colors[0],
                    text=text_vals,
                    textposition="outside" if show_values else None,
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel or (x if orientation == "v" else y),
        yaxis_title=ylabel or (y if orientation == "v" else x),
        width=width,
        height=height,
        barmode=barmode,
    )

    return apply_theme(fig, theme=theme)


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
    theme: str = "nature",
    width: int = 900,
    height: int = 600,
    colors: list[str] | None = None,
    opacity: float = 0.7,
) -> go.Figure:
    """
    绑制散点图 (相关性分析、预测评估)

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    x : str
        X 轴列名
    y : str
        Y 轴列名
    color : str, optional
        颜色分组列
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
    theme : str, default "nature"
        主题
    width, height : int
        图表尺寸
    colors : list[str], optional
        自定义颜色列表
    opacity : float, default 0.7
        透明度

    Returns
    -------
    go.Figure
        Plotly 图表对象
    """
    if colors is None:
        colors = get_color_palette("nature")

    fig = go.Figure()

    valid_df = df.select([x, y]).drop_nulls()
    x_arr = valid_df[x].to_numpy()
    y_arr = valid_df[y].to_numpy()

    r2 = None
    if show_r2 and len(x_arr) > 1:
        correlation = np.corrcoef(x_arr, y_arr)[0, 1]
        r2 = correlation**2

    if color is not None:
        groups = df[color].unique().to_list()
        for i, group in enumerate(groups):
            group_df = df.filter(pl.col(color) == group)
            marker_size = group_df[size].to_list() if size else 8
            fig.add_trace(
                go.Scatter(
                    x=group_df[x].to_list(),
                    y=group_df[y].to_list(),
                    mode="markers",
                    name=str(group),
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=marker_size,
                        opacity=opacity,
                    ),
                )
            )
    else:
        marker_size = df[size].to_list() if size else 8
        fig.add_trace(
            go.Scatter(
                x=df[x].to_list(),
                y=df[y].to_list(),
                mode="markers",
                marker=dict(
                    color=colors[0],
                    size=marker_size,
                    opacity=opacity,
                ),
                name="Data",
            )
        )

    if show_trendline and len(x_arr) > 1:
        z = np.polyfit(x_arr, y_arr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 100)

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=p(x_line),
                mode="lines",
                name=f"Trendline (R²={r2:.3f})" if r2 else "Trendline",
                line=dict(color="#333333", dash="dash", width=2),
            )
        )

    if xlabel and ylabel and "actual" in xlabel.lower() and "predict" in ylabel.lower():
        min_val = min(x_arr.min(), y_arr.min())
        max_val = max(x_arr.max(), y_arr.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="gray", dash="dot", width=1),
            )
        )

    title_text = title
    if show_r2 and r2 is not None:
        title_text = f"{title} (R² = {r2:.4f})" if title else f"R² = {r2:.4f}"

    fig.update_layout(
        title=title_text,
        xaxis_title=xlabel or x,
        yaxis_title=ylabel or y,
        width=width,
        height=height,
    )

    return apply_theme(fig, theme=theme)


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
    colorscale: str = "RdBu_r",
    theme: str = "nature",
    width: int = 800,
    height: int = 700,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """
    绑制热力图 (相关矩阵、特征分析)

    Parameters
    ----------
    df : pl.DataFrame
        数据框。如果只传入数据框，会识别相关矩阵格式
    x, y, z : str, optional
        X轴、Y轴、值列名。如不指定，使用数据框作为矩阵
    title : str
        图表标题
    xlabel, ylabel : str, optional
        轴标签
    show_values : bool, default True
        是否在格子中显示数值
    colorscale : str, default "RdBu_r"
        配色方案
    theme : str, default "nature"
        主题
    width, height : int
        图表尺寸
    zmin, zmax : float, optional
        颜色范围

    Returns
    -------
    go.Figure
        Plotly 图表对象
    """
    if z is None:
        if "variable" in df.columns:
            x_labels = df["variable"].to_list()
            numeric_cols = [c for c in df.columns if c != "variable"]
            z_data = df.select(numeric_cols).to_numpy()
            y_labels = numeric_cols
        else:
            numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
            data_np = df.select(numeric_cols).to_numpy()
            corr_matrix = np.corrcoef(data_np, rowvar=False)
            x_labels = numeric_cols
            y_labels = numeric_cols
            z_data = corr_matrix
    else:
        pivot_df = df.pivot(on=x, index=y, values=z)
        y_labels = pivot_df[y].to_list()
        x_labels = [c for c in pivot_df.columns if c != y]
        z_data = pivot_df.select(x_labels).to_numpy()

    text = [[f"{val:.2f}" for val in row] for row in z_data] if show_values else None

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            zmin=zmin if zmin is not None else (-1 if z is None else None),
            zmax=zmax if zmax is not None else (1 if z is None else None),
            text=text,
            texttemplate="%{text}" if show_values else None,
            textfont={"size": 10},
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title or "Correlation Matrix",
        xaxis_title=xlabel or "",
        yaxis_title=ylabel or "",
        width=width,
        height=height,
    )

    return apply_theme(fig, theme=theme)


def box_plot(
    df: pl.DataFrame,
    y: str,
    *,
    x: str | None = None,
    color: str | None = None,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_points: bool = True,
    theme: str = "nature",
    width: int = 900,
    height: int = 600,
    colors: list[str] | None = None,
) -> go.Figure:
    """
    绑制箱线图 (分布分析、异常值检测)

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    y : str
        数值列名
    x : str, optional
        分组列名
    color : str, optional
        颜色分组列
    title : str
        图表标题
    xlabel, ylabel : str, optional
        轴标签
    show_points : bool, default True
        是否显示散点
    theme : str, default "nature"
        主题
    width, height : int
        图表尺寸
    colors : list[str], optional
        自定义颜色列表

    Returns
    -------
    go.Figure
        Plotly 图表对象
    """
    if colors is None:
        colors = get_color_palette("nature")

    fig = go.Figure()
    boxpoints = "all" if show_points else "outliers"

    if x is not None:
        groups = df[x].unique().to_list()
        for i, group in enumerate(groups):
            group_df = df.filter(pl.col(x) == group)
            fig.add_trace(
                go.Box(
                    y=group_df[y].to_list(),
                    name=str(group),
                    marker_color=colors[i % len(colors)],
                    boxpoints=boxpoints,
                    jitter=0.3,
                    pointpos=-1.5,
                )
            )
    else:
        fig.add_trace(
            go.Box(
                y=df[y].to_list(),
                name=y,
                marker_color=colors[0],
                boxpoints=boxpoints,
                jitter=0.3,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel or (x or ""),
        yaxis_title=ylabel or y,
        width=width,
        height=height,
        showlegend=x is not None,
    )

    return apply_theme(fig, theme=theme)


def histogram(
    df: pl.DataFrame,
    x: str,
    *,
    color: str | None = None,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str = "Frequency",
    bins: int = 30,
    show_kde: bool = True,
    show_mean: bool = True,
    show_median: bool = True,
    theme: str = "nature",
    width: int = 900,
    height: int = 600,
    colors: list[str] | None = None,
    opacity: float = 0.7,
) -> go.Figure:
    """
    绘制直方图 (分布分析)

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    x : str
        数值列名
    color : str, optional
        分组列
    title : str
        图表标题
    xlabel : str, optional
        X 轴标签
    ylabel : str, default "Frequency"
        Y 轴标签
    bins : int, default 30
        直方图区间数
    show_kde : bool, default True
        是否显示核密度估计曲线 (当前未实现)
    show_mean : bool, default True
        是否显示均值线
    show_median : bool, default True
        是否显示中位数线
    theme : str, default "nature"
        主题
    width, height : int
        图表尺寸
    colors : list[str], optional
        自定义颜色列表
    opacity : float, default 0.7
        透明度

    Returns
    -------
    go.Figure
        Plotly 图表对象
    """
    if colors is None:
        colors = get_color_palette("nature")

    fig = go.Figure()

    data = df[x].drop_nulls()
    data_list = data.to_list()

    fig.add_trace(
        go.Histogram(
            x=data_list,
            nbinsx=bins,
            marker_color=colors[0],
            opacity=opacity,
            name="Histogram",
        )
    )

    if show_mean:
        mean_val = data.mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="#E64B35",
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="top right",
        )

    if show_median:
        median_val = data.median()
        fig.add_vline(
            x=median_val,
            line_dash="dot",
            line_color="#00A087",
            annotation_text=f"Median: {median_val:.2f}",
            annotation_position="top left",
        )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel or x,
        yaxis_title=ylabel,
        width=width,
        height=height,
        bargap=0.05,
    )

    return apply_theme(fig, theme=theme)


def radar_plot(
    df: pl.DataFrame,
    categories: list[str],
    *,
    values_col: str | None = None,
    name_col: str | None = None,
    title: str = "",
    fill: bool = True,
    theme: str = "nature",
    width: int = 700,
    height: int = 600,
    colors: list[str] | None = None,
) -> go.Figure:
    """
    绘制雷达图 (多维度对比)

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    categories : list[str]
        维度列名列表
    values_col : str, optional
        如果数据是长格式，指定值列
    name_col : str, optional
        分组名称列
    title : str
        图表标题
    fill : bool, default True
        是否填充
    theme : str, default "nature"
        主题
    width, height : int
        图表尺寸
    colors : list[str], optional
        自定义颜色列表

    Returns
    -------
    go.Figure
        Plotly 图表对象
    """
    if colors is None:
        colors = get_color_palette("nature")

    fig = go.Figure()
    categories_closed = categories + [categories[0]]

    if name_col is not None:
        names = df[name_col].unique().to_list()
        for i, name in enumerate(names):
            row = df.filter(pl.col(name_col) == name)
            values = [row[cat].to_list()[0] for cat in categories]
            values_closed = values + [values[0]]

            fig.add_trace(
                go.Scatterpolar(
                    r=values_closed,
                    theta=categories_closed,
                    fill="toself" if fill else None,
                    name=str(name),
                    line_color=colors[i % len(colors)],
                    fillcolor=colors[i % len(colors)] if fill else None,
                    opacity=0.6 if fill else 1,
                )
            )
    else:
        for i in range(len(df)):
            row = df.row(i, named=True)
            values = [row[cat] for cat in categories]
            values_closed = values + [values[0]]

            name = row.get("name", f"Series {i + 1}")

            fig.add_trace(
                go.Scatterpolar(
                    r=values_closed,
                    theta=categories_closed,
                    fill="toself" if fill else None,
                    name=str(name),
                    line_color=colors[i % len(colors)],
                )
            )

    max_val = df.select(categories).max().to_numpy().max()
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_val * 1.1],
            ),
        ),
        title=title,
        width=width,
        height=height,
        showlegend=True,
    )

    return apply_theme(fig, theme=theme)


def subplot_grid(
    figures: list[go.Figure],
    *,
    rows: int | None = None,
    cols: int | None = None,
    titles: list[str] | None = None,
    main_title: str = "",
    width: int = 1200,
    height: int = 800,
    theme: str = "nature",
) -> go.Figure:
    """
    创建子图网格

    Parameters
    ----------
    figures : list[go.Figure]
        图表列表
    rows, cols : int, optional
        行数和列数，自动计算如果未指定
    titles : list[str], optional
        子图标题
    main_title : str
        主标题
    width, height : int
        总尺寸
    theme : str, default "nature"
        主题

    Returns
    -------
    go.Figure
        组合后的图表
    """
    n = len(figures)

    if rows is None and cols is None:
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
    elif rows is None:
        rows = (n + cols - 1) // cols
    elif cols is None:
        cols = (n + rows - 1) // rows

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
    )

    for i, source_fig in enumerate(figures):
        row = i // cols + 1
        col = i % cols + 1

        for trace in source_fig.data:
            fig.add_trace(trace, row=row, col=col)

    fig.update_layout(
        title=main_title,
        width=width,
        height=height,
        showlegend=False,
    )

    return apply_theme(fig, theme=theme)
