"""
可视化模块
==========

提供科研级高质量可视化，支持两种后端：

1. **Plotly（默认）**：交互式图表，适合数据探索
2. **Matplotlib**：静态图表，适合论文导出

主要图表
--------
- line_plot：折线图（趋势分析、时序预测）
- bar_plot：柱状图（分组对比）
- scatter_plot：散点图（相关性、预测评估）
- heatmap：热力图（相关矩阵）
- box_plot：箱线图（分布、异常值）
- histogram：直方图（分布分析）
- radar_plot：雷达图（多维对比）

使用示例
--------
>>> # Plotly 后端（交互式）
>>> from templates import viz
>>> fig = viz.scatter_plot(df, x="x", y="y", show_trendline=True)
>>> fig.show()

>>> # Matplotlib 后端（论文级）
>>> from templates.viz import mpl
>>> fig = mpl.scatter_plot(df, x="x", y="y", style="nature")
>>> mpl.save_figure(fig, "scatter", formats=["png", "pdf"])
"""

# Plotly 后端（默认）
# Matplotlib 后端
from . import mpl_bindots as mpl
from .export import save_all_figures, save_figure
from .plots import (
    bar_plot,
    box_plot,
    heatmap,
    histogram,
    line_plot,
    radar_plot,
    scatter_plot,
    subplot_grid,
)
from .themes import apply_theme, get_color_palette

__all__ = [
    # Plotly 图表
    "line_plot",
    "bar_plot",
    "scatter_plot",
    "heatmap",
    "box_plot",
    "histogram",
    "radar_plot",
    "subplot_grid",
    # 主题
    "apply_theme",
    "get_color_palette",
    # 导出
    "save_figure",
    "save_all_figures",
    # Matplotlib 后端
    "mpl",
]
