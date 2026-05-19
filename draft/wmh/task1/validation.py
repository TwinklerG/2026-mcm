"""
验证和评估函数。

包含重构准确率、后验一致性、稳定性分析和敏感性分析等核心验证指标。

新增指标（v2.0）：
- Bottom-2 Recall: 针对 S28+ 规则变化的软性验证指标
- Finale Ranking Accuracy: 决赛排名 Kendall's Tau 系数
- CI 宽度分布分析: 按选手/周次的不确定性量化
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

try:
    from .data_structures import ContestantInfo, MCMCConfig, SeasonWeekData
    from .priors import DEFAULT_INDUSTRY_PRIOR
    from .sampler import AdaptiveMCMCSampler
    from .voting import (
        check_elimination_constraint,
        compute_combined_rank,
        compute_combined_score_percentage,
        get_voting_method,
    )
except ImportError:
    from data_structures import ContestantInfo, MCMCConfig, SeasonWeekData
    from priors import DEFAULT_INDUSTRY_PRIOR
    from sampler import AdaptiveMCMCSampler
    from voting import (
        check_elimination_constraint,
        compute_combined_rank,
        compute_combined_score_percentage,
        get_voting_method,
    )


def evaluate_constraint_satisfaction(
    posteriors: dict[tuple[int, int], np.ndarray],
    season_week_data: list[SeasonWeekData],
) -> tuple[float, int, int, list[dict]]:
    """评估约束满足率（异常案例单独分析）。"""
    satisfied = 0
    total = 0
    anomalies = []

    for swd in season_week_data:
        if not swd.eliminated_ids:
            continue

        key = (swd.season, swd.week)
        if key not in posteriors:
            continue

        fan_shares = posteriors[key]
        eliminated_indices = [
            swd.contestant_ids.index(eid)
            for eid in swd.eliminated_ids
            if eid in swd.contestant_ids
        ]

        if not eliminated_indices:
            continue

        is_sat, violation, details = check_elimination_constraint(
            fan_shares, swd.judge_scores, eliminated_indices, swd.season
        )

        if is_sat:
            satisfied += 1
        else:
            anomalies.append({
                "season": swd.season,
                "week": swd.week,
                "eliminated": swd.eliminated_ids,
                "type": "ANOMALY",
                "violation": violation,
                "details": details,
            })
        total += 1

    rate = satisfied / total if total > 0 else 0.0
    return rate, satisfied, total, anomalies


def evaluate_finale_ranking(
    posteriors: dict[tuple[int, int], np.ndarray],
    season_week_data: list[SeasonWeekData],
) -> tuple[float, int, int, int, int]:
    """评估决赛排名。"""
    winner_correct = 0
    winner_total = 0
    top3_correct = 0
    top3_total = 0

    for swd in season_week_data:
        if not swd.is_finale or not swd.final_placements:
            continue

        key = (swd.season, swd.week)
        if key not in posteriors:
            continue

        fan_shares = posteriors[key]
        method = get_voting_method(swd.season)
        fan_shares_norm = fan_shares / fan_shares.sum()

        if method == "percentage":
            combined = compute_combined_score_percentage(
                swd.judge_scores, fan_shares_norm
            )
            pred_ranking = list(np.argsort(-combined))
        else:
            combined_ranks = compute_combined_rank(swd.judge_scores, fan_shares_norm)
            pred_ranking = list(np.argsort(combined_ranks))

        actual_ranking = []
        for i, cid in enumerate(swd.contestant_ids):
            if cid in swd.final_placements:
                actual_ranking.append((i, swd.final_placements[cid]))
        actual_ranking.sort(key=lambda x: x[1])
        actual_order = [x[0] for x in actual_ranking]

        if actual_order:
            winner_total += 1
            if pred_ranking[0] == actual_order[0]:
                winner_correct += 1

        for i in range(min(3, len(actual_order))):
            top3_total += 1
            if i < len(pred_ranking) and pred_ranking[i] == actual_order[i]:
                top3_correct += 1

    winner_acc = winner_correct / winner_total if winner_total > 0 else 0.0
    return winner_acc, winner_correct, winner_total, top3_correct, top3_total


def compute_bottom2_accuracy(
    posteriors: dict[tuple[int, int], np.ndarray],
    season_week_data: list[SeasonWeekData],
) -> dict:
    """
    Bottom-2 Recall：针对 S28+ 规则变化的软性验证指标。

    从第 28 季起，DWTS 使用 "Judges' Save" 规则：
    1. 计算 Total_Score 找出 Bottom 2
    2. 评委从 Bottom 2 中选择淘汰谁

    对于多人淘汰的情况（如双淘汰周），检查预测的 Bottom-k 是否包含所有被淘汰者。

    这意味着只要模型能把被淘汰者圈定在 Bottom k+1 里（考虑评委选择的灵活性），
    就算数学上成功了——评委最后的选择是主观的。

    Parameters
    ----------
    posteriors : dict
        后验分布 {(season, week): fan_shares}
    season_week_data : list
        所有周次数据

    Returns
    -------
    dict
        包含 strict_accuracy、bottom2_recall、按赛季的分解
    """
    results = {
        "overall": {"strict_correct": 0, "bottom2_correct": 0, "total": 0},
        "pre_s28": {"strict_correct": 0, "bottom2_correct": 0, "total": 0},  # S1-S27
        "post_s28": {"strict_correct": 0, "bottom2_correct": 0, "total": 0},  # S28+
        "per_season": {},
        "cases": [],
    }

    for swd in season_week_data:
        # 跳过无淘汰周和决赛
        if not swd.eliminated_ids or swd.is_finale:
            continue

        key = (swd.season, swd.week)
        if key not in posteriors:
            continue

        fan_shares = posteriors[key]
        method = get_voting_method(swd.season)
        k = len(swd.eliminated_ids)  # 被淘汰人数

        # 计算综合得分
        fan_shares_norm = fan_shares / fan_shares.sum()
        if method == "percentage":
            combined = compute_combined_score_percentage(
                swd.judge_scores, fan_shares_norm
            )
            # 得分越低越危险
            sorted_indices = list(np.argsort(combined))
        else:
            combined_ranks = compute_combined_rank(swd.judge_scores, fan_shares_norm)
            # 排名越高越危险
            sorted_indices = list(np.argsort(-combined_ranks))

        # 预测的最低 k 名和 Bottom-2（或 Bottom-(k+1) 对于多淘汰）
        pred_bottom_k = set(sorted_indices[:k])
        check_size = max(2, k + 1)  # 至少检查 Bottom-2，多人淘汰时检查 k+1
        pred_bottom_flexible = set(sorted_indices[:check_size])

        # 实际被淘汰者
        eliminated_indices = set(
            swd.contestant_ids.index(eid)
            for eid in swd.eliminated_ids
            if eid in swd.contestant_ids
        )

        if not eliminated_indices:
            continue

        # Strict Correct：预测的 Bottom-k 完全匹配被淘汰的 k 人
        is_strict = pred_bottom_k == eliminated_indices

        # Bottom-2 Correct（软性）：所有被淘汰者都在预测的 Bottom-(k+1) 中
        is_bottom2 = eliminated_indices.issubset(pred_bottom_flexible)

        # 更新统计
        results["overall"]["total"] += 1
        if is_strict:
            results["overall"]["strict_correct"] += 1
        if is_bottom2:
            results["overall"]["bottom2_correct"] += 1

        # 按规则时期分类
        period = "pre_s28" if swd.season < 28 else "post_s28"
        results[period]["total"] += 1
        if is_strict:
            results[period]["strict_correct"] += 1
        if is_bottom2:
            results[period]["bottom2_correct"] += 1

        # 按赛季分类
        s = swd.season
        if s not in results["per_season"]:
            results["per_season"][s] = {
                "strict_correct": 0,
                "bottom2_correct": 0,
                "total": 0,
            }
        results["per_season"][s]["total"] += 1
        if is_strict:
            results["per_season"][s]["strict_correct"] += 1
        if is_bottom2:
            results["per_season"][s]["bottom2_correct"] += 1

        # 记录案例
        results["cases"].append({
            "season": swd.season,
            "week": swd.week,
            "n_eliminated": k,
            "eliminated": swd.eliminated_ids,
            "pred_bottom_k": [swd.contestant_ids[i] for i in pred_bottom_k],
            "pred_bottom_flexible": [
                swd.contestant_ids[i] for i in list(pred_bottom_flexible)[:check_size]
            ],
            "is_strict": is_strict,
            "is_bottom2": is_bottom2,
        })

    # 计算准确率
    for key in ["overall", "pre_s28", "post_s28"]:
        total = results[key]["total"]
        if total > 0:
            results[key]["strict_accuracy"] = results[key]["strict_correct"] / total
            results[key]["bottom2_recall"] = results[key]["bottom2_correct"] / total
        else:
            results[key]["strict_accuracy"] = 0.0
            results[key]["bottom2_recall"] = 0.0

    for s in results["per_season"]:
        total = results["per_season"][s]["total"]
        if total > 0:
            results["per_season"][s]["strict_accuracy"] = (
                results["per_season"][s]["strict_correct"] / total
            )
            results["per_season"][s]["bottom2_recall"] = (
                results["per_season"][s]["bottom2_correct"] / total
            )

    return results


def compute_finale_ranking_accuracy(
    posteriors: dict[tuple[int, int], np.ndarray],
    season_week_data: list[SeasonWeekData],
) -> dict:
    """
    决赛排名准确率：验证模型硬实力的最佳场所。

    决赛通常没有 "评委救人" 环节，纯拼分。
    使用 Kendall's Tau 系数衡量排名相关性。

    Parameters
    ----------
    posteriors : dict
        后验分布 {(season, week): fan_shares}
    season_week_data : list
        所有周次数据

    Returns
    -------
    dict
        包含 winner_accuracy、top3_accuracy、kendall_tau 等
    """
    results = {
        "winner_correct": 0,
        "winner_total": 0,
        "top3_position_correct": 0,
        "top3_total": 0,
        "kendall_tau_values": [],
        "per_season": {},
        "cases": [],
    }

    for swd in season_week_data:
        if not swd.is_finale or not swd.final_placements:
            continue

        key = (swd.season, swd.week)
        if key not in posteriors:
            continue

        fan_shares = posteriors[key]
        method = get_voting_method(swd.season)
        fan_shares_norm = fan_shares / fan_shares.sum()

        # 计算预测排名
        if method == "percentage":
            combined = compute_combined_score_percentage(
                swd.judge_scores, fan_shares_norm
            )
            pred_ranking_indices = list(np.argsort(-combined))  # 高分优先
        else:
            combined_ranks = compute_combined_rank(swd.judge_scores, fan_shares_norm)
            pred_ranking_indices = list(np.argsort(combined_ranks))  # 低排名优先

        # 实际排名
        actual_ranking = []
        for i, cid in enumerate(swd.contestant_ids):
            if cid in swd.final_placements:
                actual_ranking.append((i, swd.final_placements[cid]))
        actual_ranking.sort(key=lambda x: x[1])
        actual_order = [x[0] for x in actual_ranking]

        if not actual_order:
            continue

        # 冠军预测
        results["winner_total"] += 1
        winner_correct = pred_ranking_indices[0] == actual_order[0]
        if winner_correct:
            results["winner_correct"] += 1

        # Top-3 位置准确率
        for i in range(min(3, len(actual_order))):
            results["top3_total"] += 1
            if (
                i < len(pred_ranking_indices)
                and pred_ranking_indices[i] == actual_order[i]
            ):
                results["top3_position_correct"] += 1

        # Kendall's Tau 相关系数
        n_finalists = min(len(pred_ranking_indices), len(actual_order))
        if n_finalists >= 2:
            pred_subset = pred_ranking_indices[:n_finalists]
            actual_subset = actual_order[:n_finalists]

            # 将索引转换为排名
            pred_ranks = [
                pred_subset.index(idx) + 1 if idx in pred_subset else n_finalists + 1
                for idx in actual_subset
            ]
            actual_ranks = list(range(1, n_finalists + 1))

            tau, p_value = stats.kendalltau(pred_ranks, actual_ranks)
            results["kendall_tau_values"].append({
                "season": swd.season,
                "tau": float(tau) if not np.isnan(tau) else 0.0,
                "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
            })

        # 按赛季记录
        results["per_season"][swd.season] = {
            "winner_correct": winner_correct,
            "pred_ranking": [swd.contestant_ids[i] for i in pred_ranking_indices[:3]],
            "actual_ranking": [swd.contestant_ids[i] for i in actual_order[:3]],
        }

        results["cases"].append({
            "season": swd.season,
            "week": swd.week,
            "pred_winner": swd.contestant_ids[pred_ranking_indices[0]],
            "actual_winner": swd.contestant_ids[actual_order[0]],
            "winner_correct": winner_correct,
            "pred_top3": [swd.contestant_ids[i] for i in pred_ranking_indices[:3]],
            "actual_top3": [swd.contestant_ids[i] for i in actual_order[:3]],
        })

    # 计算汇总统计
    if results["winner_total"] > 0:
        results["winner_accuracy"] = results["winner_correct"] / results["winner_total"]
    else:
        results["winner_accuracy"] = 0.0

    if results["top3_total"] > 0:
        results["top3_position_accuracy"] = (
            results["top3_position_correct"] / results["top3_total"]
        )
    else:
        results["top3_position_accuracy"] = 0.0

    if results["kendall_tau_values"]:
        tau_values = [v["tau"] for v in results["kendall_tau_values"]]
        results["mean_kendall_tau"] = float(np.mean(tau_values))
        results["median_kendall_tau"] = float(np.median(tau_values))
    else:
        results["mean_kendall_tau"] = 0.0
        results["median_kendall_tau"] = 0.0

    return results


def compute_detailed_uncertainty(
    samples: dict[int, list[dict[int, np.ndarray]]],
    seasons_data: dict[int, list[SeasonWeekData]],
    contestant_info: dict[str, ContestantInfo] | None = None,
) -> dict:
    """
    详细的不确定性分析：按选手/周次的 CI 宽度分布。

    回答问题：is that certainty always the same for each contestant/week?

    Parameters
    ----------
    samples : dict
        MCMC 样本 {season: [sample_dicts]}
    seasons_data : dict
        赛季数据 {season: [SeasonWeekData]}
    contestant_info : dict, optional
        选手信息，用于按行业分类

    Returns
    -------
    dict
        包含 CI 宽度分布、按周/选手类型的分析
    """
    uncertainty_data = []

    for season, season_samples in samples.items():
        if not season_samples:
            continue

        weeks_data = seasons_data.get(season, [])
        for swd in weeks_data:
            week = swd.week
            stacked = np.stack([s[week] for s in season_samples])

            for i, cid in enumerate(swd.contestant_ids):
                cri_lower = np.percentile(stacked[:, i], 2.5)
                cri_upper = np.percentile(stacked[:, i], 97.5)
                ci_width = cri_upper - cri_lower
                mean_share = np.mean(stacked[:, i])
                std_share = np.std(stacked[:, i])

                # 获取选手信息
                industry = "Unknown"
                if contestant_info and cid in contestant_info:
                    industry = contestant_info[cid].industry

                uncertainty_data.append({
                    "season": season,
                    "week": week,
                    "contestant_id": cid,
                    "industry": industry,
                    "mean_share": float(mean_share),
                    "std_share": float(std_share),
                    "ci_lower": float(cri_lower),
                    "ci_upper": float(cri_upper),
                    "ci_width": float(ci_width),
                    "is_eliminated": cid in swd.eliminated_ids,
                    "n_contestants": len(swd.contestant_ids),
                })

    if not uncertainty_data:
        return {"error": "No data available"}

    # 汇总统计
    ci_widths = [d["ci_width"] for d in uncertainty_data]
    results = {
        "overall": {
            "mean_ci_width": float(np.mean(ci_widths)),
            "median_ci_width": float(np.median(ci_widths)),
            "std_ci_width": float(np.std(ci_widths)),
            "q25_ci_width": float(np.percentile(ci_widths, 25)),
            "q75_ci_width": float(np.percentile(ci_widths, 75)),
            "min_ci_width": float(np.min(ci_widths)),
            "max_ci_width": float(np.max(ci_widths)),
        },
        "by_week_position": {},
        "by_industry": {},
        "by_elimination_status": {},
        "controversial_contestants": [],  # 高不确定性
        "stable_contestants": [],  # 低不确定性
        "raw_data": uncertainty_data,
    }

    # 按周位置分析（早期 vs 后期）
    early_weeks = [d for d in uncertainty_data if d["week"] <= 4]
    late_weeks = [d for d in uncertainty_data if d["week"] > 4]

    if early_weeks:
        results["by_week_position"]["early"] = {
            "mean_ci_width": float(np.mean([d["ci_width"] for d in early_weeks])),
            "count": len(early_weeks),
        }
    if late_weeks:
        results["by_week_position"]["late"] = {
            "mean_ci_width": float(np.mean([d["ci_width"] for d in late_weeks])),
            "count": len(late_weeks),
        }

    # 按行业分析
    industries = sorted(set(d["industry"] for d in uncertainty_data))
    for industry in industries:
        industry_data = [d for d in uncertainty_data if d["industry"] == industry]
        if industry_data:
            results["by_industry"][industry] = {
                "mean_ci_width": float(np.mean([d["ci_width"] for d in industry_data])),
                "std_ci_width": float(np.std([d["ci_width"] for d in industry_data])),
                "count": len(industry_data),
            }

    # 按淘汰状态分析
    eliminated = [d for d in uncertainty_data if d["is_eliminated"]]
    survived = [d for d in uncertainty_data if not d["is_eliminated"]]

    if eliminated:
        results["by_elimination_status"]["eliminated"] = {
            "mean_ci_width": float(np.mean([d["ci_width"] for d in eliminated])),
            "count": len(eliminated),
        }
    if survived:
        results["by_elimination_status"]["survived"] = {
            "mean_ci_width": float(np.mean([d["ci_width"] for d in survived])),
            "count": len(survived),
        }

    # 找出"争议型选手"（高不确定性）和"稳态型选手"（低不确定性）
    ci_threshold_high = np.percentile(ci_widths, 90)
    ci_threshold_low = np.percentile(ci_widths, 10)

    for d in uncertainty_data:
        if d["ci_width"] >= ci_threshold_high:
            results["controversial_contestants"].append({
                "contestant_id": d["contestant_id"],
                "season": d["season"],
                "week": d["week"],
                "ci_width": d["ci_width"],
            })
        elif d["ci_width"] <= ci_threshold_low:
            results["stable_contestants"].append({
                "contestant_id": d["contestant_id"],
                "season": d["season"],
                "week": d["week"],
                "ci_width": d["ci_width"],
            })

    # 限制输出数量
    results["controversial_contestants"] = sorted(
        results["controversial_contestants"], key=lambda x: -x["ci_width"]
    )[:20]
    results["stable_contestants"] = sorted(
        results["stable_contestants"], key=lambda x: x["ci_width"]
    )[:20]

    return results


def compute_uncertainty_metrics(
    samples: dict[int, list[dict[int, np.ndarray]]],
    seasons_data: dict[int, list[SeasonWeekData]],
) -> dict:
    """
    计算后验不确定性指标。

    使用可信区间（贝叶斯），而非置信区间。
    """
    uncertainty = {}

    for season, season_samples in samples.items():
        if not season_samples:
            continue

        weeks_data = seasons_data[season]
        for swd in weeks_data:
            week = swd.week
            stacked = np.stack([s[week] for s in season_samples])

            mean = np.mean(stacked, axis=0)
            std = np.std(stacked, axis=0)
            cri_lower = np.percentile(stacked, 2.5, axis=0)  # Credible Interval
            cri_upper = np.percentile(stacked, 97.5, axis=0)

            for i, cid in enumerate(swd.contestant_ids):
                uncertainty[(season, week, cid)] = {
                    "mean": float(mean[i]),
                    "std": float(std[i]),
                    "credible_interval_lower": float(cri_lower[i]),
                    "credible_interval_upper": float(cri_upper[i]),
                    "credible_interval_width": float(cri_upper[i] - cri_lower[i]),
                }

    return uncertainty


def compute_reconstruction_accuracy(
    posteriors: dict[tuple[int, int], np.ndarray],
    season_week_data: list[SeasonWeekData],
) -> dict:
    """
    重构准确率：核心验证指标。

    对于每个淘汰周，检查估计的最低 k 个综合得分
    是否对应实际被淘汰的 k 个选手。

    支持三种情况：
    1. 无人淘汰 - 跳过该周
    2. 单人淘汰 - 检查预测的最低得分者是否是被淘汰者
    3. 多人淘汰 - 检查预测的最低 k 名是否完全匹配被淘汰的 k 人

    这是核心指标，因为：
    1. 不存在粉丝投票的真实值（无法使用 Spearman）
    2. 我们只能根据已知结果（淘汰）进行验证
    3. 直接测试模型的目的：解释淘汰

    Returns
    -------
    dict
        包含准确率、每赛季细分和案例分析。
    """
    total = 0
    correct = 0
    partial_correct = 0  # 部分正确（多人淘汰时至少预测对一个）
    cases = []

    for swd in season_week_data:
        # 无人淘汰的周次，跳过
        if not swd.eliminated_ids:
            continue

        key = (swd.season, swd.week)
        if key not in posteriors:
            continue

        fan_shares = posteriors[key]
        method = get_voting_method(swd.season)
        k = len(swd.eliminated_ids)  # 被淘汰人数

        # 计算综合得分
        fan_shares_norm = fan_shares / fan_shares.sum()
        if method == "percentage":
            combined = compute_combined_score_percentage(
                swd.judge_scores, fan_shares_norm
            )
            # 预测的最低 k 个得分对应的索引
            pred_bottom_k_idx = set(np.argsort(combined)[:k].tolist())
        else:  # rank or rank_bottom2
            combined_ranks = compute_combined_rank(swd.judge_scores, fan_shares_norm)
            # 最高排名 = 最差，预测的最低 k 名
            pred_bottom_k_idx = set(np.argsort(-combined_ranks)[:k].tolist())

        # 实际被淘汰者索引
        eliminated_indices = set(
            swd.contestant_ids.index(eid)
            for eid in swd.eliminated_ids
            if eid in swd.contestant_ids
        )

        # 完全匹配：预测的底部 k 名 = 实际被淘汰的 k 人
        is_correct = pred_bottom_k_idx == eliminated_indices

        # 部分匹配：预测的底部 k 名中至少有一个是实际被淘汰者
        is_partial = len(pred_bottom_k_idx & eliminated_indices) > 0

        total += 1
        if is_correct:
            correct += 1
        if is_partial:
            partial_correct += 1

        cases.append({
            "season": swd.season,
            "week": swd.week,
            "n_eliminated": k,
            "predicted_bottom_k": [swd.contestant_ids[i] for i in pred_bottom_k_idx],
            "actual_eliminated": swd.eliminated_ids,
            "is_correct": is_correct,
            "is_partial": is_partial,
            "overlap": len(pred_bottom_k_idx & eliminated_indices),
        })

    accuracy = correct / total if total > 0 else 0.0
    partial_accuracy = partial_correct / total if total > 0 else 0.0

    # 每赛季细分
    season_breakdown = {}
    for case in cases:
        s = case["season"]
        season_breakdown.setdefault(s, {"correct": 0, "partial": 0, "total": 0})
        season_breakdown[s]["total"] += 1
        if case["is_correct"]:
            season_breakdown[s]["correct"] += 1
        if case["is_partial"]:
            season_breakdown[s]["partial"] += 1

    for s in season_breakdown:
        total_s = season_breakdown[s]["total"]
        season_breakdown[s]["accuracy"] = (
            season_breakdown[s]["correct"] / total_s if total_s > 0 else 0.0
        )
        season_breakdown[s]["partial_accuracy"] = (
            season_breakdown[s]["partial"] / total_s if total_s > 0 else 0.0
        )

    return {
        "reconstruction_accuracy": accuracy,
        "partial_accuracy": partial_accuracy,
        "correct": correct,
        "partial_correct": partial_correct,
        "total": total,
        "per_season": season_breakdown,
        "cases": cases,
    }


def compute_posterior_consistency(
    samples: dict[int, list[dict[int, np.ndarray]]],
    seasons_data: dict[int, list[SeasonWeekData]],
) -> dict:
    """
    后验一致性概率：基于 MCMC 的验证。

    对于每次淘汰，计算 MCMC 样本中
    正确预测被淘汰选手的比例。

    高一致性 (>80%) 表明：
    - 模型对淘汰机制很有信心
    - 该案例的后验不确定性低

    低一致性 (<50%) 表明：
    - 高不确定性；多个选手可能被淘汰
    - 可能是"势均力敌"或异常

    Returns
    -------
    dict
        每集的一致性和汇总统计。
    """
    results = []

    for season, season_samples in samples.items():
        if not season_samples:
            continue

        weeks_data = seasons_data.get(season, [])
        for swd in weeks_data:
            if not swd.eliminated_ids:
                continue

            week = swd.week
            method = get_voting_method(swd.season)

            eliminated_indices = [
                swd.contestant_ids.index(eid)
                for eid in swd.eliminated_ids
                if eid in swd.contestant_ids
            ]

            if not eliminated_indices:
                continue

            # 统计有多少样本正确预测了淘汰
            n_correct = 0
            n_total = len(season_samples)

            for sample in season_samples:
                fan_shares = sample[week]
                fan_shares_norm = fan_shares / fan_shares.sum()

                if method == "percentage":
                    combined = compute_combined_score_percentage(
                        swd.judge_scores, fan_shares_norm
                    )
                    pred_lowest = int(np.argmin(combined))
                else:
                    combined_ranks = compute_combined_rank(
                        swd.judge_scores, fan_shares_norm
                    )
                    pred_lowest = int(np.argmax(combined_ranks))

                if pred_lowest in eliminated_indices:
                    n_correct += 1

            consistency = n_correct / n_total if n_total > 0 else 0.0
            results.append({
                "season": season,
                "week": week,
                "consistency": consistency,
                "eliminated": swd.eliminated_ids,
                "n_samples": n_total,
            })

    # 汇总统计
    consistencies = [r["consistency"] for r in results]
    return {
        "episodes": results,
        "mean_consistency": float(np.mean(consistencies)) if consistencies else 0.0,
        "std_consistency": float(np.std(consistencies)) if consistencies else 0.0,
        "min_consistency": float(min(consistencies)) if consistencies else 0.0,
        "high_confidence_episodes": sum(1 for c in consistencies if c > 0.8),
        "uncertain_episodes": sum(1 for c in consistencies if c < 0.5),
        "n_episodes": len(results),
    }


def run_stability_analysis(
    season_week_data: list[SeasonWeekData],
    contestant_info: dict[str, ContestantInfo],
    config: MCMCConfig,
    train_seasons_end: int = 20,
    cache_dir: Path | None = None,
) -> dict:
    """
    稳定性分析：跨赛季泛化测试。

    这不是传统的交叉验证（会破坏时间结构）。
    而是：
    1. 在第 1-20 季（旧数据）上训练
    2. 在第 21+ 季（新数据）上测试

    如果模型泛化良好，学到的先验应该能迁移。
    这测试粉丝行为模式是否随时间稳定。

    Parameters
    ----------
    season_week_data : list[SeasonWeekData]
        所有数据。
    contestant_info : dict[str, ContestantInfo]
        选手元数据。
    config : MCMCConfig
        MCMC 配置。
    train_seasons_end : int
        训练集包含的最后赛季（默认：20）。
    cache_dir : Path | None
        缓存目录（可选）。

    Returns
    -------
    dict
        训练集和测试集性能，以及泛化间隙。
    """
    # 划分数据
    train_data = [swd for swd in season_week_data if swd.season <= train_seasons_end]
    test_data = [swd for swd in season_week_data if swd.season > train_seasons_end]

    # 过滤选手信息
    train_contestants = set()
    for swd in train_data:
        train_contestants.update(swd.contestant_ids)
    train_contestant_info = {
        cid: info for cid, info in contestant_info.items() if cid in train_contestants
    }

    test_contestants = set()
    for swd in test_data:
        test_contestants.update(swd.contestant_ids)
    test_contestant_info = {
        cid: info for cid, info in contestant_info.items() if cid in test_contestants
    }

    print(
        f"  稳定性分析：在第 1-{train_seasons_end} 季上训练，在第 {train_seasons_end + 1}+ 季上测试"
    )
    print(f"    训练集: {len(train_data)} 个周次")
    print(f"    测试集: {len(test_data)} 个周次")

    # 在旧赛季上训练
    print("  在历史赛季上训练...")
    train_sampler = AdaptiveMCMCSampler(
        train_data,
        train_contestant_info,
        config,
        cache_dir,
        use_data_driven_priors=True,
    )
    train_posteriors, train_diagnostics, _ = train_sampler.sample_all_seasons(
        verbose=False
    )
    train_recon = compute_reconstruction_accuracy(train_posteriors, train_data)

    # 在新赛季上测试（使用从训练中学到的先验）
    print("  在近期赛季上测试...")
    test_sampler = AdaptiveMCMCSampler(
        test_data, test_contestant_info, config, cache_dir, use_data_driven_priors=True
    )
    # 迁移学到的先验
    test_sampler.industry_priors = train_sampler.industry_priors.copy()
    test_posteriors, test_diagnostics, _ = test_sampler.sample_all_seasons(
        verbose=False
    )
    test_recon = compute_reconstruction_accuracy(test_posteriors, test_data)

    generalization_gap = (
        train_recon["reconstruction_accuracy"] - test_recon["reconstruction_accuracy"]
    )

    return {
        "train_seasons": f"1-{train_seasons_end}",
        "test_seasons": f"{train_seasons_end + 1}+",
        "train_accuracy": train_recon["reconstruction_accuracy"],
        "test_accuracy": test_recon["reconstruction_accuracy"],
        "generalization_gap": generalization_gap,
        "train_details": train_recon,
        "test_details": test_recon,
    }


def analyze_anomalies_for_paper(anomalies: list[dict]) -> dict:
    """为论文准备异常案例分析。"""
    if not anomalies:
        return {
            "n_anomalies": 0,
            "cases": [],
        }

    analysis = {
        "n_anomalies": len(anomalies),
        "cases": [],
    }

    for case in anomalies:
        case_info = {
            "season": case["season"],
            "week": case["week"],
            "eliminated": case.get("eliminated", []),
            "anomaly_type": case.get("anomaly_type", "UNKNOWN"),
        }

        if "analysis" in case:
            case_info["analysis"] = case["analysis"]

        analysis["cases"].append(case_info)

    return analysis


def run_sensitivity_analysis(
    season_week_data: list[SeasonWeekData],
    contestant_info: dict[str, ContestantInfo],
    base_config: MCMCConfig,
    cache_dir: Path | None = None,
) -> dict:
    """
    敏感性分析：测试先验选择如何影响结果。

    如果结果对先验不敏感，则数据主导推断（良好）。

    注意：重构准确率是离散硬指标，对先验变化天然不敏感（约束主导）。
    因此我们额外计算未淘汰者（Survivors）后验均值的差异（MAD），
    这是更敏感的连续指标。
    """
    results = {}

    # 1. 数据驱动的先验（复用主流程缓存）
    print("  使用数据驱动的先验进行测试...")
    sampler_dd = AdaptiveMCMCSampler(
        season_week_data,
        contestant_info,
        base_config,
        cache_dir,  # 启用缓存，复用主流程结果
        use_data_driven_priors=True,
    )
    posteriors_dd, diagnostics_dd, _ = sampler_dd.sample_all_seasons(
        verbose=False,
        use_cache=True,  # 启用缓存读取
    )
    recon_dd = compute_reconstruction_accuracy(posteriors_dd, season_week_data)
    results["data_driven_priors"] = {
        "reconstruction_accuracy": recon_dd["reconstruction_accuracy"],
    }

    # 2. 均匀先验（使用独立缓存，避免重复计算）
    print("  使用均匀先验进行测试...")
    uniform_priors = {k: 0.5 for k in DEFAULT_INDUSTRY_PRIOR}
    sampler_uniform = AdaptiveMCMCSampler(
        season_week_data,
        contestant_info,
        base_config,
        cache_dir,  # 启用缓存，但先验哈希不同，会创建独立缓存文件
        use_data_driven_priors=False,
    )
    sampler_uniform.industry_priors = uniform_priors
    sampler_uniform.prior_means = {cid: 0.5 for cid in contestant_info}
    # 重新计算先验哈希以确保缓存键正确
    sampler_uniform.prior_hash = sampler_uniform._compute_prior_hash()
    posteriors_uniform, _, _ = sampler_uniform.sample_all_seasons(
        verbose=False,
        use_cache=True,  # 启用缓存
    )
    recon_uniform = compute_reconstruction_accuracy(
        posteriors_uniform, season_week_data
    )
    results["uniform_priors"] = {
        "reconstruction_accuracy": recon_uniform["reconstruction_accuracy"],
    }

    # 分析差异（离散指标）
    diff = abs(
        recon_dd["reconstruction_accuracy"] - recon_uniform["reconstruction_accuracy"]
    )

    # 计算未淘汰者后验均值差异（MAD）- 连续指标，更敏感
    survivor_mad_list = []
    survivor_kl_list = []

    for swd in season_week_data:
        key = (swd.season, swd.week)
        if key not in posteriors_dd or key not in posteriors_uniform:
            continue

        shares_dd = posteriors_dd[key]
        shares_uniform = posteriors_uniform[key]

        # 识别未淘汰者
        eliminated_indices = set(
            swd.contestant_ids.index(eid)
            for eid in swd.eliminated_ids
            if eid in swd.contestant_ids
        )
        survivor_indices = [
            i for i in range(len(swd.contestant_ids)) if i not in eliminated_indices
        ]

        if len(survivor_indices) == 0:
            continue

        # 归一化
        shares_dd_norm = shares_dd / shares_dd.sum()
        shares_uniform_norm = shares_uniform / shares_uniform.sum()

        # 计算未淘汰者的 MAD
        survivor_dd = shares_dd_norm[survivor_indices]
        survivor_uniform = shares_uniform_norm[survivor_indices]
        mad = np.mean(np.abs(survivor_dd - survivor_uniform))
        survivor_mad_list.append(mad)

        # 计算 KL 散度（对称化）
        eps = 1e-10
        p = np.clip(survivor_dd, eps, 1 - eps)
        q = np.clip(survivor_uniform, eps, 1 - eps)
        p = p / p.sum()
        q = q / q.sum()
        kl_pq = np.sum(p * np.log(p / q))
        kl_qp = np.sum(q * np.log(q / p))
        symmetric_kl = (kl_pq + kl_qp) / 2
        survivor_kl_list.append(symmetric_kl)

    results["sensitivity"] = {
        "accuracy_difference": diff,
        "survivor_mad_mean": float(np.mean(survivor_mad_list))
        if survivor_mad_list
        else 0.0,
        "survivor_mad_std": float(np.std(survivor_mad_list))
        if survivor_mad_list
        else 0.0,
        "survivor_kl_mean": float(np.mean(survivor_kl_list))
        if survivor_kl_list
        else 0.0,
        "survivor_kl_std": float(np.std(survivor_kl_list)) if survivor_kl_list else 0.0,
        "n_episodes_compared": len(survivor_mad_list),
    }

    return results


def analyze_judges_save_preference(
    posteriors: dict[tuple[int, int], np.ndarray],
    season_week_data: list[SeasonWeekData],
) -> dict:
    """
    分析 S28+ 中 Judges' Save 规则下评委的偏好模式。

    当模型识别出 Bottom 2 后，评委从中选择淘汰谁。
    本函数统计评委的选择偏好，检验假设：
    - 评委是否倾向于救 Judge Score 更高的选手？

    Parameters
    ----------
    posteriors : dict
        后验分布 {(season, week): fan_shares}
    season_week_data : list
        所有周次数据

    Returns
    -------
    dict
        评委偏好分析结果
    """
    results = {
        "n_judges_save_episodes": 0,
        "saved_higher_judge_score": 0,
        "saved_higher_fan_share": 0,
        "saved_higher_combined": 0,
        "cases": [],
    }

    for swd in season_week_data:
        # 只分析 S28+ 的非决赛周
        if swd.season < 28 or swd.is_finale:
            continue

        # 只分析单人淘汰（典型的 Judges' Save 场景）
        if len(swd.eliminated_ids) != 1:
            continue

        key = (swd.season, swd.week)
        if key not in posteriors:
            continue

        fan_shares = posteriors[key]
        fan_shares_norm = fan_shares / fan_shares.sum()

        # 计算综合排名（S28+ 使用排名法）
        combined_ranks = compute_combined_rank(swd.judge_scores, fan_shares_norm)

        # 找出 Bottom 2
        sorted_indices = list(np.argsort(-combined_ranks))  # 排名越高越危险
        if len(sorted_indices) < 2:
            continue

        bottom2_indices = sorted_indices[:2]
        bottom2_ids = [swd.contestant_ids[i] for i in bottom2_indices]

        # 确定被淘汰者和被救者
        eliminated_id = swd.eliminated_ids[0]
        if eliminated_id not in bottom2_ids:
            # 如果被淘汰者不在预测的 Bottom 2 中，跳过
            continue

        saved_id = [cid for cid in bottom2_ids if cid != eliminated_id][0]

        eliminated_idx = swd.contestant_ids.index(eliminated_id)
        saved_idx = swd.contestant_ids.index(saved_id)

        eliminated_judge_score = swd.judge_scores[eliminated_idx]
        saved_judge_score = swd.judge_scores[saved_idx]
        eliminated_fan_share = fan_shares_norm[eliminated_idx]
        saved_fan_share = fan_shares_norm[saved_idx]
        eliminated_combined = combined_ranks[eliminated_idx]
        saved_combined = combined_ranks[saved_idx]

        results["n_judges_save_episodes"] += 1

        # 检验假设
        if saved_judge_score > eliminated_judge_score:
            results["saved_higher_judge_score"] += 1
        if saved_fan_share > eliminated_fan_share:
            results["saved_higher_fan_share"] += 1
        if saved_combined < eliminated_combined:  # 排名越低越好
            results["saved_higher_combined"] += 1

        results["cases"].append({
            "season": swd.season,
            "week": swd.week,
            "bottom2": bottom2_ids,
            "eliminated": eliminated_id,
            "saved": saved_id,
            "eliminated_judge_score": float(eliminated_judge_score),
            "saved_judge_score": float(saved_judge_score),
            "eliminated_fan_share": float(eliminated_fan_share),
            "saved_fan_share": float(saved_fan_share),
            "judge_saved_higher_scorer": saved_judge_score > eliminated_judge_score,
        })

    # 计算比例
    n = results["n_judges_save_episodes"]
    if n > 0:
        results["pct_saved_higher_judge_score"] = (
            results["saved_higher_judge_score"] / n
        )
        results["pct_saved_higher_fan_share"] = results["saved_higher_fan_share"] / n
        results["pct_saved_higher_combined"] = results["saved_higher_combined"] / n

        # 二项检验：评委是否显著倾向于救高分者？
        from scipy.stats import binomtest

        binom_result = binomtest(
            results["saved_higher_judge_score"], n, p=0.5, alternative="greater"
        )
        results["binomial_test_p_value"] = float(binom_result.pvalue)
        results["judges_prefer_technicians"] = binom_result.pvalue < 0.05
    else:
        results["pct_saved_higher_judge_score"] = 0.0
        results["pct_saved_higher_fan_share"] = 0.0
        results["pct_saved_higher_combined"] = 0.0
        results["binomial_test_p_value"] = 1.0
        results["judges_prefer_technicians"] = False

    # 生成结论
    if n > 0:
        pct = results["pct_saved_higher_judge_score"] * 100
        if results["judges_prefer_technicians"]:
            results["conclusion"] = (
                f"在 {n} 次 Judges' Save 中，评委 {pct:.0f}% 的情况下救了 Judge Score 更高的选手 "
                f"(p={results['binomial_test_p_value']:.4f})。评委显著倾向于保全技术流选手。"
            )
        else:
            results["conclusion"] = (
                f"在 {n} 次 Judges' Save 中，评委 {pct:.0f}% 的情况下救了 Judge Score 更高的选手 "
                f"(p={results['binomial_test_p_value']:.4f})。未发现显著偏好。"
            )
    else:
        results["conclusion"] = "没有足够的 Judges' Save 案例进行分析。"

    return results
