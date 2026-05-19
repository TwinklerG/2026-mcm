"""
网络科学分析模块
================

提供网络科学相关的分析工具，包括：
- 网络构建与基本操作
- 中心性分析（度中心性、介数中心性、接近中心性、特征向量中心性、PageRank）
- 社区检测（Louvain、标签传播、Girvan-Newman）
- 网络指标（密度、聚类系数、平均路径长度、模块度）
- 网络模型生成（ER随机图、BA无标度网络、WS小世界网络）
- 传播动力学（SIR/SIS模型）
- 网络可视化
"""

from .analysis import (
    NetworkAnalyzer,
    analyze_network,
    calculate_centralities,
    detect_communities,
    network_metrics,
)
from .dynamics import (
    compute_basic_reproduction_number,
    independent_cascade,
    simulate_cascade,
    sir_model,
    sis_model,
)
from .io import (
    adjacency_to_network,
    load_network,
    network_to_adjacency,
    network_to_polars,
    polars_to_network,
    save_network,
)
from .models import (
    generate_ba_network,
    generate_er_network,
    generate_sbm_network,
    generate_ws_network,
)
from .visualization import (
    plot_adjacency_matrix,
    plot_centrality_comparison,
    plot_community_network,
    plot_degree_distribution,
    plot_network,
    plot_sir_dynamics,
)

__all__ = [
    # 分析
    "NetworkAnalyzer",
    "analyze_network",
    "calculate_centralities",
    "detect_communities",
    "network_metrics",
    # 模型生成
    "generate_er_network",
    "generate_ba_network",
    "generate_ws_network",
    "generate_sbm_network",
    # 传播动力学
    "sir_model",
    "sis_model",
    "simulate_cascade",
    "independent_cascade",
    "compute_basic_reproduction_number",
    # 数据IO
    "load_network",
    "save_network",
    "network_to_polars",
    "polars_to_network",
    "adjacency_to_network",
    "network_to_adjacency",
    # 可视化
    "plot_network",
    "plot_degree_distribution",
    "plot_centrality_comparison",
    "plot_community_network",
    "plot_sir_dynamics",
    "plot_adjacency_matrix",
]
