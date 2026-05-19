"""
Task 4: 渐进技术公平系统 (PTFS)。

提出并验证新的投票组合系统，平衡技术公平性与观众参与度。
"""

from .data_loader import (
    ContestantWeekData,
    SeasonData,
    build_season_data,
    get_controversy_cases,
    load_raw_data,
)
from .evaluation import (
    ControversyMetrics,
    EngagementMetrics,
    SystemMetrics,
    TechnicalFairnessMetrics,
    compare_systems,
    evaluate_system,
)
from .model import (
    PercentageBasedSystem,
    PTFSParams,
    PTFSSystem,
    RankBasedSystem,
    SeasonResult,
    WeekResult,
    simulate_all_seasons,
    simulate_season,
)
from .optimization import (
    OptimizationResult,
    evaluate_params,
    fine_tune_around_best,
    grid_search,
    run_sensitivity_analysis,
)

__all__ = [
    # Data
    "ContestantWeekData",
    "SeasonData",
    "build_season_data",
    "get_controversy_cases",
    "load_raw_data",
    # Model
    "PTFSParams",
    "PTFSSystem",
    "RankBasedSystem",
    "PercentageBasedSystem",
    "SeasonResult",
    "WeekResult",
    "simulate_season",
    "simulate_all_seasons",
    # Evaluation
    "TechnicalFairnessMetrics",
    "EngagementMetrics",
    "ControversyMetrics",
    "SystemMetrics",
    "evaluate_system",
    "compare_systems",
    # Optimization
    "OptimizationResult",
    "grid_search",
    "fine_tune_around_best",
    "evaluate_params",
    "run_sensitivity_analysis",
]
