"""
Task 4 评价指标模块。

基于全数据的统计分析，展示系统间的权衡关系。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

try:
    from .data_loader import SeasonData, compute_cumulative_judge_rank
    from .model import SeasonResult
except ImportError:
    from data_loader import SeasonData, compute_cumulative_judge_rank
    from model import SeasonResult


@dataclass
class TechnicalFairnessMetrics:
    """技术公平性指标。"""

    # 核心相关性
    kendall_tau: float
    kendall_p_value: float
    spearman_rho: float
    spearman_p_value: float

    # 排名偏差分布
    mean_rank_deviation: float
    median_rank_deviation: float
    std_rank_deviation: float
    max_rank_deviation: int
    rank_deviation_90th: float

    # 技术公平性
    tech_winner_rate: float  # 技术第一夺冠率
    tech_top3_in_final_top3: float  # 技术 Top3 进入最终 Top3 的比例

    # 极端案例（全数据统计）
    tech_bottom25_in_top3_rate: float  # 技术后 25% 进入 Top 3 的比例
    tech_top25_eliminated_early_rate: float  # 技术前 25% 前半程被淘汰的比例


@dataclass
class EngagementMetrics:
    """观众参与度指标。"""

    # 悬念指标
    mean_weekly_entropy: float  # 每周得分熵（竞争激烈程度）
    std_weekly_entropy: float
    mean_score_gap: float  # 淘汰者与次低分者的平均差距

    # 爆冷指标（更有意义的定义）
    tech_top50_eliminated_rate: float  # 技术前 50% 被淘汰的比例（每周）
    tech_bottom_eliminated_rate: float  # 技术最低分被淘汰的比例（越低=越多爆冷）
    close_call_rate: float  # 前两名差距 < 5% 的比例


@dataclass
class RobustnessMetrics:
    """鲁棒性指标（跨赛季稳定性）。"""

    tau_std_across_seasons: float
    tau_min: float
    tau_max: float
    consistent_seasons_rate: float  # τ > 0.5 的赛季比例


@dataclass
class TradeoffAnalysis:
    """权衡分析。"""

    # 与基准系统的对比
    tau_vs_rank: float
    tau_vs_pct: float

    # 极端案例改善
    bottom25_improvement_vs_pct: float
    top3_retention_vs_pct: float

    # 统计显著性（配对 t 检验）
    tau_diff_significant: bool
    tau_diff_p_value: float


@dataclass
class SystemMetrics:
    """系统综合指标。"""

    system_name: str
    technical_fairness: TechnicalFairnessMetrics
    engagement: EngagementMetrics
    robustness: RobustnessMetrics
    tradeoff: TradeoffAnalysis | None = None
    season_details: list[dict] = field(default_factory=list)


def compute_technical_fairness(
    seasons_data: dict[int, SeasonData],
    results: dict[int, SeasonResult],
) -> tuple[TechnicalFairnessMetrics, list[dict]]:
    """计算技术公平性指标。"""
    all_tech_ranks: list[int] = []
    all_final_ranks: list[int] = []
    rank_deviations: list[int] = []
    tech_winner_count = 0
    total_seasons = 0

    tech_bottom25_in_top3 = 0
    total_top3 = 0
    tech_top25_eliminated_early = 0
    total_top25 = 0
    tech_top3_in_final_top3_count = 0
    total_tech_top3 = 0

    season_details = []

    for season, season_data in seasons_data.items():
        if season not in results:
            continue

        result = results[season]
        if not result.final_placements:
            continue

        total_seasons += 1
        max_week = season_data.max_week
        cumulative_scores = compute_cumulative_judge_rank(season_data, max_week)

        if not cumulative_scores:
            continue

        sorted_by_tech = sorted(
            cumulative_scores.items(), key=lambda x: x[1], reverse=True
        )
        tech_ranks = {cid: i + 1 for i, (cid, _) in enumerate(sorted_by_tech)}
        n_contestants = len(tech_ranks)

        # 技术第一是否夺冠
        tech_first = sorted_by_tech[0][0]
        simulated_winner = None
        for cid, place in result.final_placements.items():
            if place == 1:
                simulated_winner = cid
                break

        if tech_first == simulated_winner:
            tech_winner_count += 1

        tech_top3_ids = {cid for cid, _ in sorted_by_tech[:3]}
        total_tech_top3 += len(tech_top3_ids)

        season_tech_ranks = []
        season_final_ranks = []

        for cid, final_rank in result.final_placements.items():
            if cid not in tech_ranks:
                continue

            tech_rank = tech_ranks[cid]
            all_tech_ranks.append(tech_rank)
            all_final_ranks.append(final_rank)
            rank_deviations.append(abs(tech_rank - final_rank))

            season_tech_ranks.append(tech_rank)
            season_final_ranks.append(final_rank)

            if tech_rank <= 3 and final_rank <= 3:
                tech_top3_in_final_top3_count += 1

            if final_rank <= 3:
                total_top3 += 1
                if tech_rank > 0.75 * n_contestants:
                    tech_bottom25_in_top3 += 1

            if tech_rank <= max(1, 0.25 * n_contestants):
                total_top25 += 1
                if final_rank > n_contestants / 2:
                    tech_top25_eliminated_early += 1

        if len(season_tech_ranks) >= 2:
            s_tau, _ = stats.kendalltau(season_tech_ranks, season_final_ranks)
        else:
            s_tau = np.nan

        season_details.append({
            "season": season,
            "n_contestants": n_contestants,
            "kendall_tau": float(s_tau) if not np.isnan(s_tau) else None,
            "mean_deviation": float(
                np.mean([
                    abs(t - f) for t, f in zip(season_tech_ranks, season_final_ranks)
                ])
            )
            if season_tech_ranks
            else 0,
            "tech_winner": tech_first == simulated_winner,
        })

    if len(all_tech_ranks) >= 2:
        tau, tau_p = stats.kendalltau(all_tech_ranks, all_final_ranks)
        rho, rho_p = stats.spearmanr(all_tech_ranks, all_final_ranks)
    else:
        tau, tau_p, rho, rho_p = 0.0, 1.0, 0.0, 1.0

    deviations = np.array(rank_deviations) if rank_deviations else np.array([0])

    return (
        TechnicalFairnessMetrics(
            kendall_tau=float(tau),
            kendall_p_value=float(tau_p),
            spearman_rho=float(rho),
            spearman_p_value=float(rho_p),
            mean_rank_deviation=float(np.mean(deviations)),
            median_rank_deviation=float(np.median(deviations)),
            std_rank_deviation=float(np.std(deviations)),
            max_rank_deviation=int(np.max(deviations)),
            rank_deviation_90th=float(np.percentile(deviations, 90)),
            tech_winner_rate=tech_winner_count / total_seasons
            if total_seasons > 0
            else 0,
            tech_top3_in_final_top3=tech_top3_in_final_top3_count / total_tech_top3
            if total_tech_top3 > 0
            else 0,
            tech_bottom25_in_top3_rate=tech_bottom25_in_top3 / total_top3
            if total_top3 > 0
            else 0,
            tech_top25_eliminated_early_rate=tech_top25_eliminated_early / total_top25
            if total_top25 > 0
            else 0,
        ),
        season_details,
    )


def compute_engagement(
    seasons_data: dict[int, SeasonData],
    results: dict[int, SeasonResult],
) -> EngagementMetrics:
    """计算观众参与度指标。

    重新定义爆冷：基于技术分排名，而非综合得分。
    """
    weekly_entropies: list[float] = []
    score_gaps: list[float] = []
    tech_top50_eliminated = 0
    tech_bottom_eliminated = 0
    close_call_count = 0
    total_eliminations = 0

    for season, result in results.items():
        season_data = seasons_data.get(season)
        if season_data is None:
            continue

        for week_result in result.week_results:
            if week_result.eliminated_id is None:
                continue

            total_eliminations += 1
            scores = week_result.scores
            eliminated = week_result.eliminated_id
            week = week_result.week

            # 熵
            score_values = np.array(list(scores.values()))
            if score_values.sum() > 0:
                probs = score_values / score_values.sum()
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log2(probs))
                weekly_entropies.append(entropy)

            # 与次低分者的差距
            sorted_scores = sorted(scores.items(), key=lambda x: x[1])
            if len(sorted_scores) >= 2:
                gap = sorted_scores[1][1] - sorted_scores[0][1]
                score_gaps.append(gap)

                # 前两名很接近（悬念）
                total_score = sum(s for _, s in sorted_scores)
                if total_score > 0 and gap < 0.02 * total_score:
                    close_call_count += 1

            # 获取本周技术排名
            # season_data.contestants 是 dict[str, list[ContestantWeekData]]
            judge_scores = {}
            for cid, week_list in season_data.contestants.items():
                if cid in scores:
                    for w in week_list:
                        if w.week == week:
                            judge_scores[cid] = w.judge_score
                            break

            if judge_scores and eliminated in judge_scores:
                sorted_by_judge = sorted(
                    judge_scores.items(), key=lambda x: x[1], reverse=True
                )
                n = len(sorted_by_judge)
                tech_ranks = {cid: i + 1 for i, (cid, _) in enumerate(sorted_by_judge)}

                elim_tech_rank = tech_ranks.get(eliminated, n)

                # 技术最低分被淘汰
                if elim_tech_rank == n:
                    tech_bottom_eliminated += 1

                # 技术前 50% 被淘汰（爆冷）
                if elim_tech_rank <= n / 2:
                    tech_top50_eliminated += 1

    return EngagementMetrics(
        mean_weekly_entropy=float(np.mean(weekly_entropies))
        if weekly_entropies
        else 0.0,
        std_weekly_entropy=float(np.std(weekly_entropies)) if weekly_entropies else 0.0,
        mean_score_gap=float(np.mean(score_gaps)) if score_gaps else 0.0,
        tech_top50_eliminated_rate=tech_top50_eliminated / total_eliminations
        if total_eliminations > 0
        else 0.0,
        tech_bottom_eliminated_rate=tech_bottom_eliminated / total_eliminations
        if total_eliminations > 0
        else 0.0,
        close_call_rate=close_call_count / total_eliminations
        if total_eliminations > 0
        else 0.0,
    )


def compute_robustness(season_details: list[dict]) -> RobustnessMetrics:
    """计算鲁棒性指标。"""
    taus = [s["kendall_tau"] for s in season_details if s["kendall_tau"] is not None]

    if not taus:
        return RobustnessMetrics(
            tau_std_across_seasons=0.0,
            tau_min=0.0,
            tau_max=0.0,
            consistent_seasons_rate=0.0,
        )

    return RobustnessMetrics(
        tau_std_across_seasons=float(np.std(taus)),
        tau_min=float(np.min(taus)),
        tau_max=float(np.max(taus)),
        consistent_seasons_rate=sum(1 for t in taus if t > 0.5) / len(taus),
    )


def compute_tradeoff(
    system_metrics: SystemMetrics,
    rank_metrics: SystemMetrics,
    pct_metrics: SystemMetrics,
) -> TradeoffAnalysis:
    """计算权衡分析。"""
    tau_vs_rank = (
        system_metrics.technical_fairness.kendall_tau
        - rank_metrics.technical_fairness.kendall_tau
    )
    tau_vs_pct = (
        system_metrics.technical_fairness.kendall_tau
        - pct_metrics.technical_fairness.kendall_tau
    )

    bottom25_improvement = (
        pct_metrics.technical_fairness.tech_bottom25_in_top3_rate
        - system_metrics.technical_fairness.tech_bottom25_in_top3_rate
    )

    top3_retention_improvement = (
        system_metrics.technical_fairness.tech_top3_in_final_top3
        - pct_metrics.technical_fairness.tech_top3_in_final_top3
    )

    # 配对 t 检验
    system_taus = [
        s["kendall_tau"]
        for s in system_metrics.season_details
        if s["kendall_tau"] is not None
    ]
    pct_taus = [
        s["kendall_tau"]
        for s in pct_metrics.season_details
        if s["kendall_tau"] is not None
    ]

    if len(system_taus) == len(pct_taus) and len(system_taus) > 1:
        t_stat, p_value = stats.ttest_rel(system_taus, pct_taus)
        significant = p_value < 0.05
    else:
        p_value = 1.0
        significant = False

    return TradeoffAnalysis(
        tau_vs_rank=tau_vs_rank,
        tau_vs_pct=tau_vs_pct,
        bottom25_improvement_vs_pct=bottom25_improvement,
        top3_retention_vs_pct=top3_retention_improvement,
        tau_diff_significant=significant,
        tau_diff_p_value=float(p_value),
    )


def evaluate_system(
    system_name: str,
    seasons_data: dict[int, SeasonData],
    results: dict[int, SeasonResult],
) -> SystemMetrics:
    """评估单个系统。"""
    technical, season_details = compute_technical_fairness(seasons_data, results)
    engagement = compute_engagement(seasons_data, results)
    robustness = compute_robustness(season_details)

    return SystemMetrics(
        system_name=system_name,
        technical_fairness=technical,
        engagement=engagement,
        robustness=robustness,
        tradeoff=None,
        season_details=season_details,
    )


def add_tradeoff_analysis(
    system_metrics: SystemMetrics,
    rank_metrics: SystemMetrics,
    pct_metrics: SystemMetrics,
) -> SystemMetrics:
    """添加权衡分析。"""
    system_metrics.tradeoff = compute_tradeoff(
        system_metrics, rank_metrics, pct_metrics
    )
    return system_metrics


def compare_systems(systems_metrics: list[SystemMetrics]) -> dict:
    """生成系统对比报告。"""
    comparison = {
        "systems": [],
        "summary": {
            "best_technical_fairness": None,
            "best_robustness": None,
        },
    }

    best_tau = -2.0
    best_consistency = -1.0

    for metrics in systems_metrics:
        system_summary = {
            "name": metrics.system_name,
            "kendall_tau": metrics.technical_fairness.kendall_tau,
            "spearman_rho": metrics.technical_fairness.spearman_rho,
            "mean_rank_deviation": metrics.technical_fairness.mean_rank_deviation,
            "tech_winner_rate": metrics.technical_fairness.tech_winner_rate,
            "tech_top3_in_final_top3": metrics.technical_fairness.tech_top3_in_final_top3,
            "tech_bottom25_top3_rate": metrics.technical_fairness.tech_bottom25_in_top3_rate,
            "tech_top50_eliminated_rate": metrics.engagement.tech_top50_eliminated_rate,
            "tech_bottom_eliminated_rate": metrics.engagement.tech_bottom_eliminated_rate,
            "close_call_rate": metrics.engagement.close_call_rate,
            "tau_consistency": metrics.robustness.consistent_seasons_rate,
            "tau_std": metrics.robustness.tau_std_across_seasons,
        }

        if metrics.tradeoff:
            system_summary["tau_vs_pct"] = metrics.tradeoff.tau_vs_pct
            system_summary["tau_significant"] = metrics.tradeoff.tau_diff_significant
            system_summary["tau_p_value"] = metrics.tradeoff.tau_diff_p_value

        comparison["systems"].append(system_summary)

        if metrics.technical_fairness.kendall_tau > best_tau:
            best_tau = metrics.technical_fairness.kendall_tau
            comparison["summary"]["best_technical_fairness"] = metrics.system_name

        if metrics.robustness.consistent_seasons_rate > best_consistency:
            best_consistency = metrics.robustness.consistent_seasons_rate
            comparison["summary"]["best_robustness"] = metrics.system_name

    return comparison
