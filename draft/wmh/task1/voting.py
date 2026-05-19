"""
投票方法和约束检查。

实现不同赛季的投票规则和淘汰约束验证。
支持无人淘汰和多人淘汰的情况。
"""

from __future__ import annotations

import numpy as np


def get_voting_method(season: int) -> str:
    """根据赛季返回投票方法。"""
    if season <= 2:
        return "rank"
    elif season <= 27:
        return "percentage"
    else:
        return "rank_bottom2"


def compute_combined_score_percentage(
    judge_scores: np.ndarray, fan_shares: np.ndarray
) -> np.ndarray:
    """百分比方法计算综合得分。"""
    judge_pct = judge_scores / judge_scores.sum()
    return judge_pct + fan_shares


def compute_combined_rank(
    judge_scores: np.ndarray, fan_shares: np.ndarray
) -> np.ndarray:
    """排名方法计算综合排名 (值越小越好)。"""
    n = len(judge_scores)
    judge_ranks = n - np.argsort(np.argsort(judge_scores))
    fan_ranks = n - np.argsort(np.argsort(fan_shares))
    return judge_ranks + fan_ranks


def check_elimination_constraint(
    fan_shares: np.ndarray,
    judge_scores: np.ndarray,
    eliminated_indices: list[int],
    season: int,
) -> tuple[bool, float, dict]:
    """
    检查淘汰约束。

    支持三种情况：
    1. 无人淘汰 (eliminated_indices 为空) - 直接返回 True
    2. 单人淘汰 - 检查被淘汰者是否有最低综合得分
    3. 多人淘汰 - 检查所有被淘汰者是否都在最低的 k 个综合得分中

    Parameters
    ----------
    fan_shares : np.ndarray
        粉丝投票份额（未归一化）
    judge_scores : np.ndarray
        评委总分
    eliminated_indices : list[int]
        被淘汰选手的索引列表
    season : int
        赛季编号

    Returns
    -------
    tuple[bool, float, dict]
        (是否满足约束, 违反程度, 详细信息)
    """
    # 无人淘汰的周次，约束自动满足
    if not eliminated_indices:
        return True, 0.0, {"no_elimination": True}

    n = len(judge_scores)
    k = len(eliminated_indices)  # 被淘汰人数
    method = get_voting_method(season)
    fan_shares_norm = fan_shares / fan_shares.sum()

    details = {
        "method": method,
        "n_contestants": n,
        "n_eliminated": k,
        "eliminated_indices": eliminated_indices,
    }

    if method == "percentage":
        combined = compute_combined_score_percentage(judge_scores, fan_shares_norm)
        # 找到最低的 k 个综合得分对应的索引
        bottom_k_idx = set(np.argsort(combined)[:k].tolist())
        eliminated_set = set(eliminated_indices)

        # 检查所有被淘汰者是否都在最低 k 名中
        is_satisfied = eliminated_set == bottom_k_idx

        # 计算违反程度
        elim_scores = [combined[i] for i in eliminated_indices]
        scores_others = [combined[i] for i in range(n) if i not in eliminated_indices]
        min_other_score = min(scores_others) if scores_others else min(elim_scores)
        max_elim_score = max(elim_scores)
        violation = max(0, max_elim_score - min_other_score)

        details["bottom_k"] = list(bottom_k_idx)
        details["margin"] = float(min_other_score - max_elim_score)
        return is_satisfied, violation, details

    elif method == "rank_bottom2":
        # S28+ 规则：底部两人由评委选择淘汰
        combined_ranks = compute_combined_rank(judge_scores, fan_shares_norm)
        # 如果淘汰人数 > 2，需要检查 bottom k+1 或特殊处理
        check_size = max(2, k)
        bottom_idx = set(np.argsort(-combined_ranks)[:check_size].tolist())

        # 至少有一个被淘汰者在 bottom 中即可（评委可以选择）
        is_satisfied = any(idx in bottom_idx for idx in eliminated_indices)
        details["bottom_check"] = list(bottom_idx)

        if is_satisfied:
            return True, 0.0, details
        else:
            elim_ranks = [combined_ranks[i] for i in eliminated_indices]
            threshold_rank = sorted(combined_ranks, reverse=True)[check_size - 1]
            violation = max(0, threshold_rank - max(elim_ranks))
            return False, violation, details

    else:  # rank (S1-S2)
        combined_ranks = compute_combined_rank(judge_scores, fan_shares_norm)
        # 最高的 k 个排名分数对应最后 k 名
        bottom_k_idx = set(np.argsort(-combined_ranks)[:k].tolist())
        eliminated_set = set(eliminated_indices)

        is_satisfied = eliminated_set == bottom_k_idx
        details["bottom_k"] = list(bottom_k_idx)
        return is_satisfied, 0.0, details


def check_finale_constraint(
    fan_shares: np.ndarray,
    judge_scores: np.ndarray,
    final_placements: dict[str, int],
    contestant_ids: list[str],
    season: int,
) -> tuple[bool, float, dict]:
    """检查决赛排名约束。"""
    if not final_placements:
        return True, 0.0, {}

    method = get_voting_method(season)
    fan_shares_norm = fan_shares / fan_shares.sum()

    if method == "percentage":
        combined = compute_combined_score_percentage(judge_scores, fan_shares_norm)
        pred_ranking = np.argsort(-combined)
    else:
        combined_ranks = compute_combined_rank(judge_scores, fan_shares_norm)
        pred_ranking = np.argsort(combined_ranks)

    actual_ranking = []
    for cid in contestant_ids:
        if cid in final_placements:
            actual_ranking.append((contestant_ids.index(cid), final_placements[cid]))
    actual_ranking.sort(key=lambda x: x[1])
    actual_order = [x[0] for x in actual_ranking]

    n_check = min(3, len(actual_order))
    violation = sum(
        (n_check - i)
        for i in range(n_check)
        if i < len(pred_ranking) and pred_ranking[i] != actual_order[i]
    )

    return (
        violation == 0,
        violation,
        {"pred": list(pred_ranking[:3]), "actual": actual_order[:3]},
    )
