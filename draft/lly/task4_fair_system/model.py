"""
Task 4 核心模型：渐进技术公平系统 (PTFS)。

实现三种投票组合方法并进行对比：
1. Rank-based (S1-S2, S28+)
2. Percentage-based (S3-S27)
3. PTFS (新提出)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).parent))

try:
    from .data_loader import ContestantWeekData, SeasonData, get_week_data
except ImportError:
    from data_loader import ContestantWeekData, SeasonData, get_week_data


@dataclass(frozen=True)
class PTFSParams:
    """PTFS 系统参数。"""

    judge_weight_start: float = 0.40  # 初始评委权重
    judge_weight_end: float = 0.70  # 最终评委权重
    improvement_alpha: float = 0.10  # 进步系数
    protection_threshold: float = 0.25  # 技术保护门槛（Top 25% 免疫）

    def to_dict(self) -> dict:
        return {
            "judge_weight_start": self.judge_weight_start,
            "judge_weight_end": self.judge_weight_end,
            "improvement_alpha": self.improvement_alpha,
            "protection_threshold": self.protection_threshold,
        }


@dataclass
class WeekResult:
    """单周模拟结果。"""

    season: int
    week: int
    scores: dict[str, float]  # contestant_id -> combined score
    rankings: dict[str, int]  # contestant_id -> rank (1 is best)
    eliminated_id: str | None
    protected_ids: list[str]  # 受保护的选手


@dataclass
class SeasonResult:
    """单赛季模拟结果。"""

    season: int
    week_results: list[WeekResult]
    final_placements: dict[str, int]  # contestant_id -> final placement
    elimination_order: list[str]  # 淘汰顺序（先淘汰的在前）


class VotingSystem(Protocol):
    """投票系统协议。"""

    def compute_score(
        self,
        week_data: list[ContestantWeekData],
        prev_week_data: list[ContestantWeekData] | None,
        week: int,
        max_week: int,
    ) -> dict[str, float]: ...

    def get_protected_ids(
        self,
        week_data: list[ContestantWeekData],
    ) -> list[str]: ...


class RankBasedSystem:
    """排名法系统（S1-S2, S28+）。"""

    def compute_score(
        self,
        week_data: list[ContestantWeekData],
        prev_week_data: list[ContestantWeekData] | None,
        week: int,
        max_week: int,
    ) -> dict[str, float]:
        """计算排名法综合得分（排名之和，越小越好）。"""
        n = len(week_data)

        # 评委排名
        judge_sorted = sorted(week_data, key=lambda x: x.judge_score, reverse=True)
        judge_ranks = {c.contestant_id: i + 1 for i, c in enumerate(judge_sorted)}

        # 粉丝排名
        fan_sorted = sorted(week_data, key=lambda x: x.fan_share, reverse=True)
        fan_ranks = {c.contestant_id: i + 1 for i, c in enumerate(fan_sorted)}

        # 综合得分 = 排名之和（越小越好，转换为越大越好）
        scores = {}
        for c in week_data:
            rank_sum = judge_ranks[c.contestant_id] + fan_ranks[c.contestant_id]
            # 转换为越大越好：max_rank_sum - actual_rank_sum
            scores[c.contestant_id] = (2 * n + 2) - rank_sum

        return scores

    def get_protected_ids(
        self,
        week_data: list[ContestantWeekData],
    ) -> list[str]:
        """排名法没有保护机制。"""
        return []


class PercentageBasedSystem:
    """百分比法系统（S3-S27）。"""

    def compute_score(
        self,
        week_data: list[ContestantWeekData],
        prev_week_data: list[ContestantWeekData] | None,
        week: int,
        max_week: int,
    ) -> dict[str, float]:
        """计算百分比法综合得分。"""
        total_judge = sum(c.judge_score for c in week_data)
        total_fan = sum(c.fan_share for c in week_data)

        scores = {}
        for c in week_data:
            judge_pct = c.judge_score / total_judge if total_judge > 0 else 0
            fan_pct = c.fan_share / total_fan if total_fan > 0 else 0
            scores[c.contestant_id] = judge_pct + fan_pct

        return scores

    def get_protected_ids(
        self,
        week_data: list[ContestantWeekData],
    ) -> list[str]:
        """百分比法没有保护机制。"""
        return []


class PTFSSystem:
    """渐进技术公平系统 (PTFS)。"""

    def __init__(self, params: PTFSParams | None = None):
        self.params = params or PTFSParams()

    def _get_dynamic_weights(self, week: int, max_week: int) -> tuple[float, float]:
        """计算动态权重。"""
        if max_week <= 1:
            w_judge = (
                self.params.judge_weight_start + self.params.judge_weight_end
            ) / 2
        else:
            progress = (week - 1) / (max_week - 1)
            w_judge = (
                self.params.judge_weight_start
                + (self.params.judge_weight_end - self.params.judge_weight_start)
                * progress
            )
        w_fan = 1.0 - w_judge
        return w_judge, w_fan

    def _compute_improvement(
        self,
        current: ContestantWeekData,
        prev_week_data: list[ContestantWeekData] | None,
        score_range: tuple[float, float],
    ) -> float:
        """计算进步分。"""
        if prev_week_data is None:
            return 0.0

        # 找到上周数据
        prev_data = None
        for p in prev_week_data:
            if p.contestant_id == current.contestant_id:
                prev_data = p
                break

        if prev_data is None:
            return 0.0

        # 计算归一化进步分
        improvement = current.judge_score - prev_data.judge_score
        if improvement <= 0:
            return 0.0

        score_min, score_max = score_range
        if score_max <= score_min:
            return 0.0

        normalized = improvement / (score_max - score_min)
        return min(normalized, 1.0)  # 截断到 [0, 1]

    def compute_score(
        self,
        week_data: list[ContestantWeekData],
        prev_week_data: list[ContestantWeekData] | None,
        week: int,
        max_week: int,
    ) -> dict[str, float]:
        """计算 PTFS 综合得分。"""
        w_judge, w_fan = self._get_dynamic_weights(week, max_week)

        # 归一化评委分
        total_judge = sum(c.judge_score for c in week_data)
        total_fan = sum(c.fan_share for c in week_data)

        # 计算分数范围（用于归一化进步分）
        all_scores = [c.judge_score for c in week_data]
        score_range = (min(all_scores), max(all_scores))

        scores = {}
        for c in week_data:
            judge_pct = c.judge_score / total_judge if total_judge > 0 else 0
            fan_pct = c.fan_share / total_fan if total_fan > 0 else 0

            improvement = self._compute_improvement(c, prev_week_data, score_range)

            combined = (
                w_judge * judge_pct
                + w_fan * fan_pct
                + self.params.improvement_alpha * improvement
            )
            scores[c.contestant_id] = combined

        return scores

    def get_protected_ids(
        self,
        week_data: list[ContestantWeekData],
    ) -> list[str]:
        """获取受保护的选手（技术分 Top θ%）。"""
        if self.params.protection_threshold <= 0:
            return []

        n = len(week_data)
        n_protected = max(1, int(n * self.params.protection_threshold))

        # 按技术分排序
        sorted_data = sorted(week_data, key=lambda x: x.judge_score, reverse=True)
        return [c.contestant_id for c in sorted_data[:n_protected]]


def simulate_season(
    season_data: SeasonData,
    system: RankBasedSystem | PercentageBasedSystem | PTFSSystem,
) -> SeasonResult:
    """模拟单赛季比赛。

    Parameters
    ----------
    season_data : SeasonData
        赛季数据
    system : VotingSystem
        投票系统

    Returns
    -------
    SeasonResult
        模拟结果
    """
    week_results: list[WeekResult] = []
    eliminated_ids: set[str] = set()
    elimination_order: list[str] = []

    # 获取所有选手
    all_contestants = set(season_data.contestants.keys())
    active_contestants = all_contestants.copy()

    prev_week_data: list[ContestantWeekData] | None = None

    for week in season_data.weeks:
        # 获取当周活跃选手数据
        week_data = get_week_data(season_data, week)
        week_data = [c for c in week_data if c.contestant_id in active_contestants]

        if len(week_data) == 0:
            continue

        # 计算得分
        scores = system.compute_score(
            week_data, prev_week_data, week, season_data.max_week
        )

        # 计算排名
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        rankings = {cid: i + 1 for i, cid in enumerate(sorted_ids)}

        # 获取受保护选手
        protected_ids = system.get_protected_ids(week_data)

        # 确定淘汰者（非决赛周）
        eliminated_id = None
        is_finale = week == season_data.max_week

        if not is_finale and len(sorted_ids) > 1:
            # 找到得分最低且未受保护的选手
            for cid in reversed(sorted_ids):
                if cid not in protected_ids:
                    eliminated_id = cid
                    break

            # 如果所有人都受保护，则淘汰得分最低者
            if eliminated_id is None:
                eliminated_id = sorted_ids[-1]

            if eliminated_id:
                eliminated_ids.add(eliminated_id)
                active_contestants.remove(eliminated_id)
                elimination_order.append(eliminated_id)

        week_results.append(
            WeekResult(
                season=season_data.season,
                week=week,
                scores=scores,
                rankings=rankings,
                eliminated_id=eliminated_id,
                protected_ids=protected_ids,
            )
        )

        prev_week_data = week_data

    # 计算最终名次
    final_placements: dict[str, int] = {}

    # 决赛选手按最后一周得分排名
    if week_results:
        final_week = week_results[-1]
        finalists = [
            cid for cid in final_week.rankings.keys() if cid not in eliminated_ids
        ]
        finalists_sorted = sorted(
            finalists, key=lambda x: final_week.scores[x], reverse=True
        )
        for i, cid in enumerate(finalists_sorted):
            final_placements[cid] = i + 1
            elimination_order.append(cid)

    # 淘汰选手按淘汰顺序逆序排名
    n_total = len(all_contestants)
    for i, cid in enumerate(elimination_order):
        if cid not in final_placements:
            # 第一个被淘汰的排最后
            final_placements[cid] = n_total - i

    return SeasonResult(
        season=season_data.season,
        week_results=week_results,
        final_placements=final_placements,
        elimination_order=elimination_order,
    )


def simulate_all_seasons(
    seasons_data: dict[int, SeasonData],
    system: RankBasedSystem | PercentageBasedSystem | PTFSSystem,
) -> dict[int, SeasonResult]:
    """模拟所有赛季。

    Parameters
    ----------
    seasons_data : dict[int, SeasonData]
        所有赛季数据
    system : VotingSystem
        投票系统

    Returns
    -------
    dict[int, SeasonResult]
        赛季 -> 结果
    """
    results = {}
    for season, data in seasons_data.items():
        results[season] = simulate_season(data, system)
    return results


def get_historical_system(season: int) -> RankBasedSystem | PercentageBasedSystem:
    """获取历史上实际使用的系统。"""
    if season <= 2 or season >= 28:
        return RankBasedSystem()
    else:
        return PercentageBasedSystem()
