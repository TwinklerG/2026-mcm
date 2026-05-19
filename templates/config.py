"""
全局配置模块
============

提供统一的配置管理，包括：
- 可视化主题和配色
- 导出设置
- 日志配置
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class VizConfig:
    """可视化配置"""

    # 主题设置
    theme: Literal["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"] = (
        "plotly_white"
    )

    # 科研级配色方案
    color_palette: list[str] = field(
        default_factory=lambda: [
            "#2E86AB",  # 深蓝
            "#A23B72",  # 玫红
            "#F18F01",  # 橙色
            "#C73E1D",  # 砖红
            "#3B1F2B",  # 深紫
            "#95C623",  # 黄绿
            "#5C4D7D",  # 紫色
            "#00A896",  # 青色
        ]
    )

    # Nature/Science 风格配色
    nature_palette: list[str] = field(
        default_factory=lambda: [
            "#E64B35",  # Nature Red
            "#4DBBD5",  # Nature Blue
            "#00A087",  # Nature Green
            "#3C5488",  # Nature Dark Blue
            "#F39B7F",  # Nature Light Red
            "#8491B4",  # Nature Grey Blue
            "#91D1C2",  # Nature Light Green
            "#DC0000",  # Nature Bright Red
        ]
    )

    # 图表尺寸
    figure_width: int = 900
    figure_height: int = 600

    # 字体设置
    font_family: str = "Arial, sans-serif"
    title_font_size: int = 18
    axis_font_size: int = 14
    tick_font_size: int = 12

    # 导出设置
    export_format: Literal["png", "pdf", "svg", "html"] = "png"
    export_dpi: int = 300
    export_scale: float = 2.0


@dataclass
class DataConfig:
    """数据处理配置"""

    # 缺失值处理
    missing_threshold: float = 0.3  # 缺失率超过此值的列将被删除

    # 异常值检测
    outlier_method: Literal["iqr", "zscore", "isolation_forest"] = "iqr"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0

    # 标准化方法
    normalize_method: Literal["minmax", "zscore", "robust"] = "minmax"


@dataclass
class MLConfig:
    """机器学习配置"""

    # 交叉验证
    cv_folds: int = 5
    random_state: int = 42

    # 特征重要性
    shap_sample_size: int = 100

    # 集成学习
    n_estimators: int = 100


@dataclass
class Config:
    """
    主配置类

    使用示例
    --------
    >>> config = Config()
    >>> config.viz.theme = "plotly_dark"
    >>> config.viz.color_palette = ["#FF0000", "#00FF00", "#0000FF"]
    """

    viz: VizConfig = field(default_factory=VizConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ml: MLConfig = field(default_factory=MLConfig)

    # 项目路径
    project_root: Path = field(default_factory=lambda: Path.cwd())
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "outputs")
    figures_dir: Path = field(default_factory=lambda: Path.cwd() / "figures")

    def __post_init__(self):
        """初始化后创建输出目录"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def use_nature_style(self):
        """使用 Nature 期刊风格"""
        self.viz.color_palette = self.viz.nature_palette.copy()
        self.viz.theme = "plotly_white"
        self.viz.font_family = "Arial, sans-serif"

    def use_dark_theme(self):
        """使用深色主题"""
        self.viz.theme = "plotly_dark"
        self.viz.color_palette = [
            "#00D4FF",
            "#FF6B6B",
            "#4ECDC4",
            "#FFE66D",
            "#95E1D3",
            "#F38181",
            "#AA96DA",
            "#FCBAD3",
        ]


# 全局默认配置实例
default_config = Config()
