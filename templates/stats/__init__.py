"""
统计分析与评价模块
==================

提供统计检验、评价模型（AHP/TOPSIS/熵权法）等功能。

主要功能
--------
- 相关分析: correlation_test, partial_correlation
- 假设检验: t_test, anova, chi_square_test
- 评价模型: ahp_weight, entropy_weight, topsis
- 敏感性分析: sensitivity_analysis
"""

from .correlation import (
    correlation_summary,
    correlation_test,
    partial_correlation,
)
from .evaluation import (
    ahp_weight,
    combine_weights,
    entropy_weight,
    topsis,
)
from .hypothesis import (
    anova,
    chi_square_test,
    normality_test,
    paired_t_test,
    t_test,
)
from .sensitivity import (
    parameter_sweep,
    sensitivity_analysis,
    tornado_diagram_data,
)

__all__ = [
    # 相关分析
    "correlation_test",
    "partial_correlation",
    "correlation_summary",
    # 假设检验
    "t_test",
    "paired_t_test",
    "anova",
    "chi_square_test",
    "normality_test",
    # 评价模型
    "ahp_weight",
    "entropy_weight",
    "topsis",
    "combine_weights",
    # 敏感性分析
    "sensitivity_analysis",
    "parameter_sweep",
    "tornado_diagram_data",
]
