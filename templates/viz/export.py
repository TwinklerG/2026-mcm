"""
图表导出模块
============

支持多种格式的高质量图表导出。
"""

from pathlib import Path
from typing import Literal

import plotly.graph_objects as go


def save_figure(
    fig: go.Figure,
    filename: str,
    *,
    output_dir: str | Path = "figures",
    format: Literal["png", "pdf", "svg", "html", "json"] = "png",
    width: int | None = None,
    height: int | None = None,
    scale: float = 2.0,
) -> Path:
    """
    保存图表到文件

    Parameters
    ----------
    fig : go.Figure
        Plotly 图表对象
    filename : str
        文件名 (不含扩展名)
    output_dir : str | Path, default "figures"
        输出目录
    format : str, default "png"
        输出格式："png", "pdf", "svg", "html", "json"
    width, height : int, optional
        图片尺寸，使用图表设置如果未指定
    scale : float, default 2.0
        缩放因子（仅对图片格式有效，2.0 表示 2x 分辨率）

    Returns
    -------
    Path
        保存的文件路径

    Examples
    --------
    >>> save_figure(fig, "trend_analysis", format="png")
    >>> save_figure(fig, "model_comparison", format="pdf", scale=3.0)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{filename}.{format}"

    # 获取图表尺寸
    if width is None:
        width = fig.layout.width or 900
    if height is None:
        height = fig.layout.height or 600

    if format == "html":
        fig.write_html(filepath, include_plotlyjs="cdn")
    elif format == "json":
        fig.write_json(filepath)
    else:
        # 图片格式
        fig.write_image(
            filepath,
            width=width,
            height=height,
            scale=scale,
            format=format,
        )

    return filepath


def save_all_figures(
    figures: dict[str, go.Figure],
    *,
    output_dir: str | Path = "figures",
    formats: list[Literal["png", "pdf", "svg", "html"]] | None = None,
    scale: float = 2.0,
) -> list[Path]:
    """
    批量保存多个图表

    Parameters
    ----------
    figures : dict[str, go.Figure]
        图表字典，键为文件名
    output_dir : str | Path, default "figures"
        输出目录
    formats : list[str], optional
        输出格式列表，默认 ["png"]
    scale : float, default 2.0
        缩放因子

    Returns
    -------
    list[Path]
        保存的文件路径列表

    Examples
    --------
    >>> figures = {
    ...     "fig1_trend": fig1,
    ...     "fig2_comparison": fig2,
    ...     "fig3_correlation": fig3,
    ... }
    >>> save_all_figures(figures, formats=["png", "pdf"])
    """
    if formats is None:
        formats = ["png"]

    saved_paths = []

    for name, fig in figures.items():
        for fmt in formats:
            path = save_figure(
                fig, name, output_dir=output_dir, format=fmt, scale=scale
            )
            saved_paths.append(path)
            print(f"✅ Saved: {path}")

    return saved_paths


def export_for_paper(
    fig: go.Figure,
    filename: str,
    *,
    output_dir: str | Path = "figures",
    dpi: int = 300,
) -> dict[str, Path]:
    """
    为论文导出图表（同时生成 PNG、PDF、SVG）

    Parameters
    ----------
    fig : go.Figure
        Plotly 图表对象
    filename : str
        文件名
    output_dir : str | Path, default "figures"
        输出目录
    dpi : int, default 300
        DPI (用于计算缩放因子)

    Returns
    -------
    dict[str, Path]
        各格式的文件路径字典

    Examples
    --------
    >>> paths = export_for_paper(fig, "figure1_model_results")
    >>> print(paths["pdf"])  # 用于 LaTeX
    """
    output_dir = Path(output_dir)

    # 计算缩放因子以达到目标 DPI
    # Plotly 默认 72 DPI
    scale = dpi / 72

    paths = {}

    for fmt in ["png", "pdf", "svg"]:
        paths[fmt] = save_figure(
            fig,
            filename,
            output_dir=output_dir,
            format=fmt,
            scale=scale if fmt == "png" else 1.0,
        )

    # 同时保存交互式 HTML 版本
    paths["html"] = save_figure(fig, filename, output_dir=output_dir, format="html")

    return paths
