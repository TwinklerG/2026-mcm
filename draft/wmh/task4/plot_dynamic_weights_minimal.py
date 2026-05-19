"""
Task 4: Minimal Dynamic Weights SVG (PPT style) — 极简风格，仅坐标轴和曲线。

参考 Task 1 贝叶斯框图中的先验/后验分布图风格：
- 只有横纵坐标轴（无刻度标签）
- 只有两条曲线（评委权重、粉丝权重）
- 无标题、无图例、无网格、无数值标注
- 纯 SVG 矢量输出
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# 配色：饱和蓝/红
COLOR_JUDGE = "#2563EB"   # 评委权重（蓝）
COLOR_FAN = "#DC2626"     # 粉丝权重（红）


def dynamic_weight(t: np.ndarray, w_start: float, w_end: float, T: int) -> np.ndarray:
    """计算动态评委权重 w_J(t)."""
    return w_start + (w_end - w_start) * (t - 1) / (T - 1)


def plot_minimal_dynamic_weights(
    w_start: float = 0.45,
    w_end: float = 0.80,
    T: int = 11,
    output_svg: Path | None = None,
) -> None:
    """
    绘制极简动态权重曲线（仅坐标轴和曲线）。
    
    Parameters
    ----------
    w_start : float
        初始评委权重（默认 0.45）
    w_end : float
        最终评委权重（默认 0.80）
    T : int
        总周数（默认 11）
    output_svg : Path
        SVG 输出路径
    """
    weeks = np.arange(1, T + 1)
    w_judge = dynamic_weight(weeks, w_start, w_end, T)
    w_fan = 1 - w_judge
    
    # 创建图形（小尺寸，适合 PPT 嵌入）- 透明背景
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_alpha(0)  # 图形背景透明
    ax.patch.set_alpha(0)   # 坐标轴背景透明
    
    # 填充曲线下方区域（半透明）
    ax.fill_between(weeks, w_judge * 100, alpha=0.25, color=COLOR_JUDGE, zorder=1)
    ax.fill_between(weeks, w_fan * 100, alpha=0.25, color=COLOR_FAN, zorder=1)
    
    # 绘制曲线（带标记点）
    ax.plot(
        weeks, w_judge * 100,
        color=COLOR_JUDGE, linewidth=3, zorder=3,
        marker="o", markersize=8, markerfacecolor=COLOR_JUDGE, markeredgecolor="white", markeredgewidth=1.5,
    )
    ax.plot(
        weeks, w_fan * 100,
        color=COLOR_FAN, linewidth=3, zorder=3,
        marker="s", markersize=7, markerfacecolor=COLOR_FAN, markeredgecolor="white", markeredgewidth=1.5,
    )
    
    # 设置坐标轴范围
    ax.set_xlim(0.5, T + 0.5)
    ax.set_ylim(0, 105)
    
    # 移除所有装饰元素
    ax.set_xticks([])  # 无 X 轴刻度
    ax.set_yticks([])  # 无 Y 轴刻度
    ax.spines["top"].set_visible(False)    # 隐藏上边框
    ax.spines["right"].set_visible(False)  # 隐藏右边框
    ax.spines["left"].set_linewidth(2)     # 左边框（Y 轴）加粗
    ax.spines["bottom"].set_linewidth(2)   # 下边框（X 轴）加粗
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    
    # 添加坐标轴箭头
    # X 轴箭头（向右）
    ax.annotate(
        "",
        xy=(T + 0.45, 0),
        xytext=(T + 0.1, 0),
        arrowprops=dict(arrowstyle="->", lw=2, color="#333333"),
        annotation_clip=False,
    )
    # Y 轴箭头（向上）
    ax.annotate(
        "",
        xy=(0.5, 108),
        xytext=(0.5, 102),
        arrowprops=dict(arrowstyle="->", lw=2, color="#333333"),
        annotation_clip=False,
    )
    
    
    # 移除网格
    ax.grid(False)
    
    # 紧凑布局
    plt.tight_layout(pad=0.2)
    
    # 保存 SVG（透明背景）
    if output_svg:
        fig.savefig(output_svg, format="svg", bbox_inches="tight", pad_inches=0.05, transparent=True)
        print(f"  Saved: {output_svg}")
    
    plt.close(fig)


def main() -> None:
    """生成极简动态权重 SVG（PPT 用）。"""
    print("Generating minimal dynamic weights SVG (PPT style)...")
    
    plot_minimal_dynamic_weights(
        output_svg=FIGURES_DIR / "dynamic_weights_minimal.svg",
    )
    
    print(f"\nMinimal SVG saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
