"""
Task 4 主运行脚本 - 渐进技术公平系统 (PTFS)。

基于全数据统计分析，展示系统权衡关系。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    build_season_data,
    load_raw_data,
)
from evaluation import (
    add_tradeoff_analysis,
    compare_systems,
    evaluate_system,
)
from model import (
    PercentageBasedSystem,
    PTFSParams,
    PTFSSystem,
    RankBasedSystem,
    simulate_all_seasons,
)
from visualization import generate_all_plots

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURES_DIR = SCRIPT_DIR / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def _to_json_serializable(obj, precision: int = 6):
    """递归转换为 JSON 可序列化格式。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v, precision) for k, v in sorted(obj.items())}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_serializable(i, precision) for i in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(float(obj), precision)
    elif hasattr(obj, "item"):
        return _to_json_serializable(obj.item(), precision)
    elif hasattr(obj, "to_dict"):
        return _to_json_serializable(obj.to_dict(), precision)
    else:
        return obj


def objective_score(metrics, weights: dict) -> float:
    """计算综合目标得分。

    Parameters
    ----------
    metrics : SystemMetrics
    weights : dict
        各指标的权重配置

    Returns
    -------
    float
        综合得分
    """
    score = 0.0

    # 技术公平性（正向）
    score += weights.get("tau", 0.4) * metrics.technical_fairness.kendall_tau

    # 技术第一夺冠率（正向）
    score += (
        weights.get("tech_winner", 0.15) * metrics.technical_fairness.tech_winner_rate
    )

    # 技术 Top3 保留率（正向）
    score += (
        weights.get("top3_retain", 0.1)
        * metrics.technical_fairness.tech_top3_in_final_top3
    )

    # 极端案例：技术后25%进Top3（负向，越低越好）
    score -= (
        weights.get("bottom25_penalty", 0.15)
        * metrics.technical_fairness.tech_bottom25_in_top3_rate
    )

    # 参与度：技术高分被淘汰率（轻微正向，保持一定悬念）
    score += weights.get("upset", 0.05) * metrics.engagement.tech_top50_eliminated_rate

    # 鲁棒性：一致性（正向）
    score += (
        weights.get("consistency", 0.15) * metrics.robustness.consistent_seasons_rate
    )

    return score


def grid_search_ptfs(
    seasons_data: dict,
    objective_weights: dict,
    verbose: bool = False,
) -> tuple[PTFSParams, float, dict]:
    """网格搜索 PTFS 参数。"""
    param_grid = {
        "judge_weight_start": [0.35, 0.40, 0.45, 0.50],
        "judge_weight_end": [0.60, 0.65, 0.70, 0.75, 0.80],
        "improvement_alpha": [0.0, 0.05, 0.10, 0.15],
        "protection_threshold": [0.0, 0.10, 0.15, 0.20],
    }

    best_params = None
    best_score = -float("inf")
    all_results = []

    total = (
        len(param_grid["judge_weight_start"])
        * len(param_grid["judge_weight_end"])
        * len(param_grid["improvement_alpha"])
        * len(param_grid["protection_threshold"])
    )

    count = 0
    for jws in param_grid["judge_weight_start"]:
        for jwe in param_grid["judge_weight_end"]:
            if jwe < jws:
                continue
            for alpha in param_grid["improvement_alpha"]:
                for prot in param_grid["protection_threshold"]:
                    count += 1

                    params = PTFSParams(
                        judge_weight_start=jws,
                        judge_weight_end=jwe,
                        improvement_alpha=alpha,
                        protection_threshold=prot,
                    )

                    system = PTFSSystem(params)
                    results = simulate_all_seasons(seasons_data, system)
                    metrics = evaluate_system("PTFS", seasons_data, results)
                    score = objective_score(metrics, objective_weights)

                    all_results.append({
                        "params": params.to_dict(),
                        "score": score,
                        "tau": metrics.technical_fairness.kendall_tau,
                        "tech_winner": metrics.technical_fairness.tech_winner_rate,
                    })

                    if score > best_score:
                        best_score = score
                        best_params = params

                    if verbose and count % 50 == 0:
                        print(f"    进度: {count}/{total} (当前最优: {best_score:.4f})")

    sensitivity = {
        "judge_weight_start": {},
        "judge_weight_end": {},
        "improvement_alpha": {},
        "protection_threshold": {},
    }

    for key in sensitivity.keys():
        for res in all_results:
            val = res["params"][key]
            if val not in sensitivity[key]:
                sensitivity[key][val] = []
            sensitivity[key][val].append(res["score"])

        for val in sensitivity[key]:
            scores = sensitivity[key][val]
            sensitivity[key][val] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "n": len(scores),
            }

    return best_params, best_score, sensitivity


def main() -> None:
    """主函数：执行 Task 4 分析。"""
    np.random.seed(42)
    random.seed(42)

    print("=" * 70)
    print("Task 4: 渐进技术公平系统 (PTFS) - 修订版")
    print("重点: 全数据统计分析，展示权衡关系")
    print("=" * 70)

    # Step 1: 加载数据
    print("\n[Step 1/6] 加载数据...")
    contestants, week_summary, fan_shares = load_raw_data()
    seasons_data = build_season_data(contestants, week_summary, fan_shares)
    print(f"  加载了 {len(seasons_data)} 个赛季")

    # Step 2: 评估基准系统
    print("\n[Step 2/6] 评估基准系统...")

    print("  Rank-based 系统...")
    rank_system = RankBasedSystem()
    rank_results = simulate_all_seasons(seasons_data, rank_system)
    rank_metrics = evaluate_system("Rank-based", seasons_data, rank_results)

    print("  Percentage-based 系统...")
    pct_system = PercentageBasedSystem()
    pct_results = simulate_all_seasons(seasons_data, pct_system)
    pct_metrics = evaluate_system("Percentage-based", seasons_data, pct_results)

    print("\n  基准系统结果:")
    print(f"    Rank-based:")
    print(f"      Kendall's τ: {rank_metrics.technical_fairness.kendall_tau:.4f}")
    print(
        f"      Tech Winner Rate: {rank_metrics.technical_fairness.tech_winner_rate:.2%}"
    )
    print(
        f"      Tech Bottom25 → Top3: {rank_metrics.technical_fairness.tech_bottom25_in_top3_rate:.2%}"
    )
    print(f"    Percentage-based:")
    print(f"      Kendall's τ: {pct_metrics.technical_fairness.kendall_tau:.4f}")
    print(
        f"      Tech Winner Rate: {pct_metrics.technical_fairness.tech_winner_rate:.2%}"
    )
    print(
        f"      Tech Bottom25 → Top3: {pct_metrics.technical_fairness.tech_bottom25_in_top3_rate:.2%}"
    )

    # Step 3: 网格搜索优化 PTFS
    print("\n[Step 3/6] 网格搜索优化 PTFS...")
    objective_weights = {
        "tau": 0.40,
        "tech_winner": 0.15,
        "top3_retain": 0.10,
        "bottom25_penalty": 0.15,
        "upset": 0.05,
        "consistency": 0.15,
    }

    best_params, best_score, sensitivity = grid_search_ptfs(
        seasons_data, objective_weights, verbose=True
    )

    print(f"\n  最优参数:")
    print(f"    judge_weight_start: {best_params.judge_weight_start:.2f}")
    print(f"    judge_weight_end: {best_params.judge_weight_end:.2f}")
    print(f"    improvement_alpha: {best_params.improvement_alpha:.2f}")
    print(f"    protection_threshold: {best_params.protection_threshold:.2f}")
    print(f"    综合得分: {best_score:.4f}")

    # Step 4: 评估 PTFS 系统
    print("\n[Step 4/6] 评估 PTFS 系统...")
    ptfs_system = PTFSSystem(best_params)
    ptfs_results = simulate_all_seasons(seasons_data, ptfs_system)
    ptfs_metrics = evaluate_system("PTFS", seasons_data, ptfs_results)

    # 添加权衡分析
    ptfs_metrics = add_tradeoff_analysis(ptfs_metrics, rank_metrics, pct_metrics)

    # Step 5: 系统对比
    print("\n[Step 5/6] 系统对比分析...")
    all_metrics = [rank_metrics, pct_metrics, ptfs_metrics]
    comparison = compare_systems(all_metrics)

    print("\n" + "=" * 80)
    print("系统对比结果（全数据统计）:")
    print("=" * 80)
    print(f"{'指标':<30} {'Rank-based':>15} {'Percentage':>15} {'PTFS':>15}")
    print("-" * 80)

    # 技术公平性
    print(
        f"{'Kendall τ':<30} "
        f"{rank_metrics.technical_fairness.kendall_tau:>15.4f} "
        f"{pct_metrics.technical_fairness.kendall_tau:>15.4f} "
        f"{ptfs_metrics.technical_fairness.kendall_tau:>15.4f}"
    )
    print(
        f"{'Spearman ρ':<30} "
        f"{rank_metrics.technical_fairness.spearman_rho:>15.4f} "
        f"{pct_metrics.technical_fairness.spearman_rho:>15.4f} "
        f"{ptfs_metrics.technical_fairness.spearman_rho:>15.4f}"
    )
    print(
        f"{'Tech Winner Rate':<30} "
        f"{rank_metrics.technical_fairness.tech_winner_rate:>15.2%} "
        f"{pct_metrics.technical_fairness.tech_winner_rate:>15.2%} "
        f"{ptfs_metrics.technical_fairness.tech_winner_rate:>15.2%}"
    )
    print(
        f"{'Tech Top3 → Final Top3':<30} "
        f"{rank_metrics.technical_fairness.tech_top3_in_final_top3:>15.2%} "
        f"{pct_metrics.technical_fairness.tech_top3_in_final_top3:>15.2%} "
        f"{ptfs_metrics.technical_fairness.tech_top3_in_final_top3:>15.2%}"
    )
    print(
        f"{'Tech Bottom25 → Top3 (↓better)':<30} "
        f"{rank_metrics.technical_fairness.tech_bottom25_in_top3_rate:>15.2%} "
        f"{pct_metrics.technical_fairness.tech_bottom25_in_top3_rate:>15.2%} "
        f"{ptfs_metrics.technical_fairness.tech_bottom25_in_top3_rate:>15.2%}"
    )
    print(
        f"{'Mean Rank Deviation (↓better)':<30} "
        f"{rank_metrics.technical_fairness.mean_rank_deviation:>15.2f} "
        f"{pct_metrics.technical_fairness.mean_rank_deviation:>15.2f} "
        f"{ptfs_metrics.technical_fairness.mean_rank_deviation:>15.2f}"
    )
    print(
        f"{'90th% Rank Deviation':<30} "
        f"{rank_metrics.technical_fairness.rank_deviation_90th:>15.1f} "
        f"{pct_metrics.technical_fairness.rank_deviation_90th:>15.1f} "
        f"{ptfs_metrics.technical_fairness.rank_deviation_90th:>15.1f}"
    )

    print("-" * 80)
    # 参与度
    print(
        f"{'Tech Top50 Eliminated (Upset)':<30} "
        f"{rank_metrics.engagement.tech_top50_eliminated_rate:>15.2%} "
        f"{pct_metrics.engagement.tech_top50_eliminated_rate:>15.2%} "
        f"{ptfs_metrics.engagement.tech_top50_eliminated_rate:>15.2%}"
    )
    print(
        f"{'Tech Lowest Eliminated':<30} "
        f"{rank_metrics.engagement.tech_bottom_eliminated_rate:>15.2%} "
        f"{pct_metrics.engagement.tech_bottom_eliminated_rate:>15.2%} "
        f"{ptfs_metrics.engagement.tech_bottom_eliminated_rate:>15.2%}"
    )
    print(
        f"{'Close Call Rate':<30} "
        f"{rank_metrics.engagement.close_call_rate:>15.2%} "
        f"{pct_metrics.engagement.close_call_rate:>15.2%} "
        f"{ptfs_metrics.engagement.close_call_rate:>15.2%}"
    )

    print("-" * 80)
    # 鲁棒性
    print(
        f"{'τ Std Across Seasons':<30} "
        f"{rank_metrics.robustness.tau_std_across_seasons:>15.4f} "
        f"{pct_metrics.robustness.tau_std_across_seasons:>15.4f} "
        f"{ptfs_metrics.robustness.tau_std_across_seasons:>15.4f}"
    )
    print(
        f"{'Consistent Seasons (τ>0.5)':<30} "
        f"{rank_metrics.robustness.consistent_seasons_rate:>15.2%} "
        f"{pct_metrics.robustness.consistent_seasons_rate:>15.2%} "
        f"{ptfs_metrics.robustness.consistent_seasons_rate:>15.2%}"
    )

    print("=" * 80)

    # 权衡分析
    if ptfs_metrics.tradeoff:
        print("\n权衡分析:")
        print("-" * 50)
        print(f"  PTFS vs Percentage-based:")
        print(f"    τ 变化: {ptfs_metrics.tradeoff.tau_vs_pct:+.4f}")
        print(
            f"    统计显著: {'是' if ptfs_metrics.tradeoff.tau_diff_significant else '否'} "
            f"(p={ptfs_metrics.tradeoff.tau_diff_p_value:.4f})"
        )
        print(
            f"    Top3 保留率改善: {ptfs_metrics.tradeoff.top3_retention_vs_pct:+.2%}"
        )
        print(
            f"    极端案例改善 (Bottom25→Top3): {ptfs_metrics.tradeoff.bottom25_improvement_vs_pct:+.2%}"
        )

    # Step 6: 保存结果
    print("\n[Step 6/6] 保存结果...")

    # 保存参数
    with open(OUTPUT_DIR / "optimal_params.json", "w") as f:
        json.dump(
            _to_json_serializable({
                "params": best_params.to_dict(),
                "score": best_score,
                "objective_weights": objective_weights,
            }),
            f,
            indent=2,
        )

    # 保存对比结果
    with open(OUTPUT_DIR / "system_comparison.json", "w") as f:
        json.dump(_to_json_serializable(comparison), f, indent=2)

    # 保存详细指标
    detailed = {
        "rank_based": {
            "technical_fairness": {
                "kendall_tau": rank_metrics.technical_fairness.kendall_tau,
                "spearman_rho": rank_metrics.technical_fairness.spearman_rho,
                "mean_rank_deviation": rank_metrics.technical_fairness.mean_rank_deviation,
                "tech_winner_rate": rank_metrics.technical_fairness.tech_winner_rate,
                "tech_top3_in_final_top3": rank_metrics.technical_fairness.tech_top3_in_final_top3,
                "tech_bottom25_in_top3_rate": rank_metrics.technical_fairness.tech_bottom25_in_top3_rate,
            },
            "engagement": {
                "tech_top50_eliminated_rate": rank_metrics.engagement.tech_top50_eliminated_rate,
                "tech_bottom_eliminated_rate": rank_metrics.engagement.tech_bottom_eliminated_rate,
                "close_call_rate": rank_metrics.engagement.close_call_rate,
            },
            "robustness": {
                "tau_std": rank_metrics.robustness.tau_std_across_seasons,
                "consistent_rate": rank_metrics.robustness.consistent_seasons_rate,
            },
        },
        "percentage_based": {
            "technical_fairness": {
                "kendall_tau": pct_metrics.technical_fairness.kendall_tau,
                "spearman_rho": pct_metrics.technical_fairness.spearman_rho,
                "mean_rank_deviation": pct_metrics.technical_fairness.mean_rank_deviation,
                "tech_winner_rate": pct_metrics.technical_fairness.tech_winner_rate,
                "tech_top3_in_final_top3": pct_metrics.technical_fairness.tech_top3_in_final_top3,
                "tech_bottom25_in_top3_rate": pct_metrics.technical_fairness.tech_bottom25_in_top3_rate,
            },
            "engagement": {
                "tech_top50_eliminated_rate": pct_metrics.engagement.tech_top50_eliminated_rate,
                "tech_bottom_eliminated_rate": pct_metrics.engagement.tech_bottom_eliminated_rate,
                "close_call_rate": pct_metrics.engagement.close_call_rate,
            },
            "robustness": {
                "tau_std": pct_metrics.robustness.tau_std_across_seasons,
                "consistent_rate": pct_metrics.robustness.consistent_seasons_rate,
            },
        },
        "ptfs": {
            "technical_fairness": {
                "kendall_tau": ptfs_metrics.technical_fairness.kendall_tau,
                "spearman_rho": ptfs_metrics.technical_fairness.spearman_rho,
                "mean_rank_deviation": ptfs_metrics.technical_fairness.mean_rank_deviation,
                "tech_winner_rate": ptfs_metrics.technical_fairness.tech_winner_rate,
                "tech_top3_in_final_top3": ptfs_metrics.technical_fairness.tech_top3_in_final_top3,
                "tech_bottom25_in_top3_rate": ptfs_metrics.technical_fairness.tech_bottom25_in_top3_rate,
            },
            "engagement": {
                "tech_top50_eliminated_rate": ptfs_metrics.engagement.tech_top50_eliminated_rate,
                "tech_bottom_eliminated_rate": ptfs_metrics.engagement.tech_bottom_eliminated_rate,
                "close_call_rate": ptfs_metrics.engagement.close_call_rate,
            },
            "robustness": {
                "tau_std": ptfs_metrics.robustness.tau_std_across_seasons,
                "consistent_rate": ptfs_metrics.robustness.consistent_seasons_rate,
            },
            "tradeoff": {
                "tau_vs_pct": ptfs_metrics.tradeoff.tau_vs_pct
                if ptfs_metrics.tradeoff
                else None,
                "tau_significant": ptfs_metrics.tradeoff.tau_diff_significant
                if ptfs_metrics.tradeoff
                else None,
                "p_value": ptfs_metrics.tradeoff.tau_diff_p_value
                if ptfs_metrics.tradeoff
                else None,
                "bottom25_improvement": ptfs_metrics.tradeoff.bottom25_improvement_vs_pct
                if ptfs_metrics.tradeoff
                else None,
            },
        },
    }
    with open(OUTPUT_DIR / "detailed_metrics.json", "w") as f:
        json.dump(_to_json_serializable(detailed), f, indent=2)

    # 保存敏感性分析
    with open(OUTPUT_DIR / "sensitivity_analysis.json", "w") as f:
        json.dump(_to_json_serializable(sensitivity), f, indent=2)

    # 保存赛季详情
    season_details = {
        "rank_based": rank_metrics.season_details,
        "percentage_based": pct_metrics.season_details,
        "ptfs": ptfs_metrics.season_details,
    }
    with open(OUTPUT_DIR / "season_details.json", "w") as f:
        json.dump(_to_json_serializable(season_details), f, indent=2)

    # 生成图表
    print("  生成可视化图表...")
    generate_all_plots(
        all_metrics,
        sensitivity,
        best_params,
        FIGURES_DIR,
        seasons_data,
        {"rank": rank_results, "pct": pct_results, "ptfs": ptfs_results},
    )

    print("\n" + "=" * 70)
    print("Task 4 完成!")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  图表目录: {FIGURES_DIR}")
    print("=" * 70)

    # 关键结论
    print("\n关键发现:")
    print("-" * 70)
    print("1. PTFS 系统特点:")
    print(f"   - Kendall's τ: {ptfs_metrics.technical_fairness.kendall_tau:.4f}")
    print(
        f"   - 相对 Percentage 变化: {ptfs_metrics.tradeoff.tau_vs_pct:+.4f}"
        if ptfs_metrics.tradeoff
        else ""
    )
    print(
        f"   - 统计显著性: p = {ptfs_metrics.tradeoff.tau_diff_p_value:.4f}"
        if ptfs_metrics.tradeoff
        else ""
    )

    print("\n2. 权衡分析:")
    print(
        f"   - 极端案例改善 (Bottom25→Top3): {ptfs_metrics.tradeoff.bottom25_improvement_vs_pct:+.2%}"
        if ptfs_metrics.tradeoff
        else ""
    )
    print(
        f"   - Top3 保留率改善: {ptfs_metrics.tradeoff.top3_retention_vs_pct:+.2%}"
        if ptfs_metrics.tradeoff
        else ""
    )

    print("\n3. 系统稳定性:")
    print(
        f"   - 一致性赛季比例 (τ>0.5): {ptfs_metrics.robustness.consistent_seasons_rate:.2%}"
    )
    print(f"   - τ 跨赛季标准差: {ptfs_metrics.robustness.tau_std_across_seasons:.4f}")


if __name__ == "__main__":
    main()
