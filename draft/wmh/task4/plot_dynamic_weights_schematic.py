"""
Task 4: Dynamic Weights Schematic (SVG) — ICCV/CVPR style for PPT.

绘制 PTFS 动态权重曲线，风格与 Task 1 一致：
- 大号加粗坐标轴标签和标题
- 衬线字体（serif）
- 高对比度红蓝配色
- SVG 矢量输出
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ICCV/CVPR 风格设置 — 与 Task 1 一致
FONT_TITLE = 28
FONT_LABEL = 24
FONT_TICK = 18
FONT_LEGEND = 16
FONT_ANNO = 20

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": FONT_LABEL,
    "axes.titlesize": FONT_TITLE,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_LEGEND,
    "axes.linewidth": 1.2,
    "axes.edgecolor": ".25",
    "grid.alpha": 0.35,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# 配色：饱和蓝/红（与 Task 3 ICCV 风格一致）
COLOR_JUDGE = "#2563EB"   # 饱和蓝（评委）
COLOR_FAN = "#DC2626"     # 饱和红（粉丝）


def dynamic_weight(t: np.ndarray, w_start: float, w_end: float, T: int) -> np.ndarray:
    """
    计算动态评委权重。
    
    w_J(t) = w_J^start + (w_J^end - w_J^start) * (t-1) / (T-1)
    """
    return w_start + (w_end - w_start) * (t - 1) / (T - 1)


def plot_dynamic_weights_schematic(
    w_start: float = 0.45,
    w_end: float = 0.80,
    T: int = 11,
    output_svg: Path | None = None,
    output_png: Path | None = None,
) -> None:
    """
    绘制 PTFS 动态权重示意图。
    
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
    output_png : Path
        PNG 输出路径（可选）
    """
    weeks = np.arange(1, T + 1)
    w_judge = dynamic_weight(weeks, w_start, w_end, T)
    w_fan = 1 - w_judge
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 填充区域（半透明）
    ax.fill_between(weeks, w_judge * 100, alpha=0.25, color=COLOR_JUDGE, zorder=1)
    ax.fill_between(weeks, w_fan * 100, alpha=0.25, color=COLOR_FAN, zorder=1)
    
    # 主线条
    ax.plot(
        weeks, w_judge * 100,
        marker="o", markersize=10, linewidth=3,
        color=COLOR_JUDGE, label=r"Judge Weight $w_J(t)$",
        zorder=3,
    )
    ax.plot(
        weeks, w_fan * 100,
        marker="s", markersize=9, linewidth=3,
        color=COLOR_FAN, label=r"Fan Weight $1 - w_J(t)$",
        zorder=3,
    )
    
    # 起点/终点标注
    ax.annotate(
        f"{w_start*100:.0f}%",
        xy=(1, w_start * 100),
        xytext=(-25, -20),
        textcoords="offset points",
        fontsize=FONT_ANNO,
        fontweight="bold",
        color=COLOR_JUDGE,
        ha="center",
    )
    ax.annotate(
        f"{w_end*100:.0f}%",
        xy=(T, w_end * 100),
        xytext=(25, 10),
        textcoords="offset points",
        fontsize=FONT_ANNO,
        fontweight="bold",
        color=COLOR_JUDGE,
        ha="center",
    )
    ax.annotate(
        f"{(1-w_start)*100:.0f}%",
        xy=(1, (1 - w_start) * 100),
        xytext=(-25, 15),
        textcoords="offset points",
        fontsize=FONT_ANNO,
        fontweight="bold",
        color=COLOR_FAN,
        ha="center",
    )
    ax.annotate(
        f"{(1-w_end)*100:.0f}%",
        xy=(T, (1 - w_end) * 100),
        xytext=(25, -15),
        textcoords="offset points",
        fontsize=FONT_ANNO,
        fontweight="bold",
        color=COLOR_FAN,
        ha="center",
    )
    
    # 坐标轴
    ax.set_xlabel("Week", fontweight="bold")
    ax.set_ylabel("Weight (%)", fontweight="bold")
    ax.set_title("PTFS: Progressive Weight Adjustment", fontweight="bold")
    
    ax.set_xlim(0.5, T + 0.5)
    ax.set_ylim(0, 100)
    ax.set_xticks(weeks)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    
    # 网格
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    
    # 图例
    ax.legend(loc="center right", frameon=True, framealpha=0.95, edgecolor="gray")
    
    plt.tight_layout()
    
    # 保存
    if output_svg:
        fig.savefig(output_svg, format="svg")
        print(f"  Saved: {output_svg}")
    if output_png:
        fig.savefig(output_png, format="png", dpi=300)
        print(f"  Saved: {output_png}")
    
    plt.close(fig)


def plot_weight_formula_diagram(
    output_svg: Path | None = None,
    output_png: Path | None = None,
) -> None:
    """
    绘制动态权重公式示意图（纯公式展示，用于 PPT）。
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    
    # 标题
    ax.text(
        0.5, 0.88,
        "PTFS: Dynamic Weight Formula",
        fontsize=FONT_TITLE,
        fontweight="bold",
        ha="center",
        va="top",
        transform=ax.transAxes,
    )
    
    # 综合得分公式
    ax.text(
        0.5, 0.62,
        r"$S_{i,t} = w_J(t) \cdot \tilde{J}_{i,t} + (1 - w_J(t)) \cdot \tilde{F}_{i,t}$",
        fontsize=24,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    
    # 动态权重公式
    ax.text(
        0.5, 0.38,
        r"$w_J(t) = w_J^{start} + (w_J^{end} - w_J^{start}) \cdot \frac{t - 1}{T - 1}$",
        fontsize=22,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    
    # 参数说明
    param_text = (
        r"$w_J^{\mathrm{start}} = 45\%$    "
        r"$w_J^{\mathrm{end}} = 80\%$    "
        r"Early: Fan voice    →    Late: Technical merit"
    )
    ax.text(
        0.5, 0.12,
        param_text,
        fontsize=18,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="#333333",
    )
    
    plt.tight_layout()
    
    if output_svg:
        fig.savefig(output_svg, format="svg")
        print(f"  Saved: {output_svg}")
    if output_png:
        fig.savefig(output_png, format="png", dpi=300)
        print(f"  Saved: {output_png}")
    
    plt.close(fig)


def main() -> None:
    """生成 Task 4 动态权重 SVG 图。"""
    print("Generating Task 4 dynamic weights schematics (SVG + PNG)...")
    
    # 1. 动态权重曲线图
    plot_dynamic_weights_schematic(
        output_svg=FIGURES_DIR / "dynamic_weights_schematic.svg",
        output_png=FIGURES_DIR / "dynamic_weights_schematic.png",
    )
    
    # 2. 公式示意图
    plot_weight_formula_diagram(
        output_svg=FIGURES_DIR / "weight_formula_diagram.svg",
        output_png=FIGURES_DIR / "weight_formula_diagram.png",
    )
    
    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
