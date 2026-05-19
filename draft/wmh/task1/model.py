"""
DWTS 粉丝投票估计的贝叶斯 MCMC 模型 - 统一接口。

本模块提供向后兼容的接口，导入所有拆分后的模块。

模块结构：
- data_structures.py - 数据结构和配置类
- priors.py - 数据驱动的先验估计
- voting.py - 投票方法和约束检查
- sampler.py - 自适应 MCMC 采样器
- validation.py - 验证函数

核心特性：
1. 数据驱动的先验估计
2. 带稀疏采样的自适应 MCMC（优化 ESS）
3. 异常检测能力
4. 快速迭代的缓存机制
5. 正确的贝叶斯验证（无 CV、无 Spearman）

验证方法：
- 重构准确率：估计的最低得分是否与实际淘汰相符？
- 后验一致性概率：MCMC 样本中产生正确淘汰的比例
- 稳定性分析：跨赛季泛化测试（非传统 CV）

术语：
- 使用"可信区间"（贝叶斯），而非"置信区间"（频率）
"""

from __future__ import annotations

# 数据结构
try:
    from .data_structures import (
        ContestantInfo,
        MCMCConfig,
        MCMCDiagnostics,
        SeasonWeekData,
    )
    from .priors import (
        DEFAULT_INDUSTRY_PRIOR,
        DEFAULT_PRIOR_MEAN,
        PRIOR_STD,
        estimate_industry_priors_from_data,
    )
    from .sampler import AdaptiveMCMCSampler
    from .validation import (
        analyze_anomalies_for_paper,
        analyze_judges_save_preference,
        compute_bottom2_accuracy,
        compute_detailed_uncertainty,
        compute_finale_ranking_accuracy,
        compute_posterior_consistency,
        compute_reconstruction_accuracy,
        compute_uncertainty_metrics,
        evaluate_constraint_satisfaction,
        evaluate_finale_ranking,
        run_sensitivity_analysis,
        run_stability_analysis,
    )
    from .voting import (
        check_elimination_constraint,
        check_finale_constraint,
        compute_combined_rank,
        compute_combined_score_percentage,
        get_voting_method,
    )
except ImportError:
    # 当作为脚本运行时使用绝对导入
    from data_structures import (
        ContestantInfo,
        MCMCConfig,
        MCMCDiagnostics,
        SeasonWeekData,
    )
    from priors import (
        DEFAULT_INDUSTRY_PRIOR,
        DEFAULT_PRIOR_MEAN,
        PRIOR_STD,
        estimate_industry_priors_from_data,
    )
    from sampler import AdaptiveMCMCSampler
    from validation import (
        analyze_anomalies_for_paper,
        analyze_judges_save_preference,
        compute_bottom2_accuracy,
        compute_detailed_uncertainty,
        compute_finale_ranking_accuracy,
        compute_posterior_consistency,
        compute_reconstruction_accuracy,
        compute_uncertainty_metrics,
        evaluate_constraint_satisfaction,
        evaluate_finale_ranking,
        run_sensitivity_analysis,
        run_stability_analysis,
    )
    from voting import (
        check_elimination_constraint,
        check_finale_constraint,
        compute_combined_rank,
        compute_combined_score_percentage,
        get_voting_method,
    )

__all__ = [
    # 数据结构
    "SeasonWeekData",
    "ContestantInfo",
    "MCMCConfig",
    "MCMCDiagnostics",
    # 先验
    "DEFAULT_INDUSTRY_PRIOR",
    "DEFAULT_PRIOR_MEAN",
    "PRIOR_STD",
    "estimate_industry_priors_from_data",
    # 采样器
    "AdaptiveMCMCSampler",
    # 投票方法
    "get_voting_method",
    "compute_combined_score_percentage",
    "compute_combined_rank",
    "check_elimination_constraint",
    "check_finale_constraint",
    # 验证
    "evaluate_constraint_satisfaction",
    "evaluate_finale_ranking",
    "compute_uncertainty_metrics",
    "compute_reconstruction_accuracy",
    "compute_posterior_consistency",
    "run_stability_analysis",
    "analyze_anomalies_for_paper",
    "run_sensitivity_analysis",
    # 新增验证函数
    "compute_bottom2_accuracy",
    "compute_finale_ranking_accuracy",
    "compute_detailed_uncertainty",
    "analyze_judges_save_preference",
]
