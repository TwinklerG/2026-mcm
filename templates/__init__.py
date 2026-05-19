"""
MCM 2026 模板库
================================

模块结构
--------
- data: 数据加载、预处理、特征工程
- viz: 科研级可视化（折线、柱状、热力图、雷达图等）
- ml: 机器学习模型封装（预测、分类、聚类）
- stats: 统计分析与评价模型（相关分析、假设检验、AHP/TOPSIS）
- utils: 工具函数（导出、日志、配置）
- network: 网络科学分析（中心性、社区检测、传播模型）

使用示例
--------
>>> from templates import data, viz, ml, stats, network
>>> df = data.load_csv("data.csv")
>>> df = data.preprocess(df, handle_missing="mean", normalize=True)
>>> viz.line_plot(df, x="year", y="value", title="Trend Analysis")
>>> G = network.generate_ba_network(1000, 3)
>>> results = network.analyze_network(G)
"""

__version__ = "0.1.0"
__author__ = "MCM Team 2026"

from . import data, ml, network, stats, utils, viz
from .config import Config

__all__ = ["data", "viz", "ml", "stats", "utils", "network", "Config"]
