"""
数据驱动的先验估计。

从历史数据中学习行业特定的先验分布。
"""

from __future__ import annotations

import numpy as np

try:
    from .data_structures import ContestantInfo, SeasonWeekData
except ImportError:
    from data_structures import ContestantInfo, SeasonWeekData

# 默认先验 (用于无数据时的 fallback)
DEFAULT_INDUSTRY_PRIOR = {
    "Actor/Actress": 0.70,
    "Singer/Rapper": 0.75,
    "Musician": 0.70,
    "TV Personality": 0.65,
    "Comedian": 0.60,
    "Model": 0.55,
    "Social media personality": 0.60,
    "Social Media Personality": 0.60,
    "Radio Personality": 0.45,
    "Athlete": 0.55,
    "Beauty Pagent": 0.50,
    "Magician": 0.45,
    "Producer": 0.45,
    "Fashion Designer": 0.40,
    "Fitness Instructor": 0.40,
    "Sports Broadcaster": 0.40,
    "Entrepreneur": 0.35,
    "Motivational Speaker": 0.35,
    "Journalist": 0.30,
    "News Anchor": 0.30,
    "Politician": 0.25,
    "Military": 0.30,
    "Astronaut": 0.35,
    "Conservationist": 0.30,
    "Racing Driver": 0.35,
    "Con artist": 0.20,
}


def estimate_industry_priors_from_data(
    contestant_info: dict[str, ContestantInfo],
    season_week_data: list[SeasonWeekData],
) -> tuple[dict[str, float], dict]:
    """
    从数据中估计行业先验（初步分析）。

    方法：计算每个行业选手的平均存活周数比例作为先验。

    Returns
    -------
    tuple[dict[str, float], dict]
        (行业先验字典, 估计诊断信息)
    """
    # 统计每个选手的存活周数
    contestant_weeks: dict[str, int] = {}
    max_weeks_per_season: dict[int, int] = {}

    # 构建赛季数据映射
    season_data: dict[int, list[SeasonWeekData]] = {}
    for swd in season_week_data:
        season_data.setdefault(swd.season, []).append(swd)

    # 计算每个赛季的最大周数
    for season, weeks_list in season_data.items():
        max_weeks_per_season[season] = max(swd.week for swd in weeks_list)

    # 统计每个选手出现的周数
    for swd in season_week_data:
        for cid in swd.contestant_ids:
            contestant_weeks[cid] = contestant_weeks.get(cid, 0) + 1

    # 按行业聚合
    industry_survival: dict[str, list[float]] = {}

    for cid, info in contestant_info.items():
        if cid not in contestant_weeks:
            continue

        # 提取赛季号
        season = int(cid.split("_S")[-1]) if "_S" in cid else 0
        if season == 0 or season not in max_weeks_per_season:
            continue

        # 存活比例 = 出现周数 / 该赛季最大周数
        survival_ratio = contestant_weeks[cid] / max_weeks_per_season[season]

        industry = info.industry
        if industry not in industry_survival:
            industry_survival[industry] = []
        industry_survival[industry].append(survival_ratio)

    # 计算每个行业的平均存活率
    estimated_priors: dict[str, float] = {}
    diagnostics = {"n_industries": 0, "sample_sizes": {}, "raw_means": {}}

    for industry, ratios in industry_survival.items():
        if len(ratios) >= 3:  # 至少 3 个样本才估计
            mean_ratio = np.mean(ratios)
            # 将存活率映射到 [0.2, 0.8] 范围
            estimated_priors[industry] = 0.2 + 0.6 * mean_ratio
            diagnostics["sample_sizes"][industry] = len(ratios)
            diagnostics["raw_means"][industry] = float(mean_ratio)

    diagnostics["n_industries"] = len(estimated_priors)

    # 填充缺失行业（使用默认值）
    for industry in DEFAULT_INDUSTRY_PRIOR:
        if industry not in estimated_priors:
            estimated_priors[industry] = DEFAULT_INDUSTRY_PRIOR[industry]

    return estimated_priors, diagnostics


DEFAULT_PRIOR_MEAN = 0.5
PRIOR_STD = 0.25  # 略微增大先验不确定性
