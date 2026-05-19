"""Task 3: 双轨分层混合效应模型模块。"""

from .analysis import (
    analyze_celebrity_characteristics,
    analyze_fixed_effects,
    analyze_pro_effects,
    generate_paper_conclusions,
    save_analysis_results,
)
from .data_preparation import (
    build_panel_data,
    get_summary_statistics,
    load_raw_data,
    prepare_model_data,
)
from .model import DualTrackHMEM, ModelResults
from .visualization import (
    generate_all_plots,
    plot_coefficient_comparison,
    plot_impact_divergence,
    plot_industry_effects,
    plot_learning_curve,
    plot_pro_scatter,
    plot_variance_decomposition,
)

__all__ = [
    # 数据准备
    "load_raw_data",
    "build_panel_data",
    "prepare_model_data",
    "get_summary_statistics",
    # 模型
    "DualTrackHMEM",
    "ModelResults",
    # 分析
    "analyze_fixed_effects",
    "analyze_pro_effects",
    "analyze_celebrity_characteristics",
    "generate_paper_conclusions",
    "save_analysis_results",
    # 可视化
    "generate_all_plots",
    "plot_pro_scatter",
    "plot_coefficient_comparison",
    "plot_variance_decomposition",
    "plot_industry_effects",
    "plot_learning_curve",
    "plot_impact_divergence",
]
