"""
Task 4 参数优化模块。

使用网格搜索和敏感性分析优化 PTFS 参数。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    from .data_loader import SeasonData
    from .evaluation import (
        SystemMetrics,
        compute_composite_score,
        compute_controversy_correction,
        compute_engagement,
        compute_technical_fairness,
    )
    from .model import PTFSParams, PTFSSystem, simulate_all_seasons
except ImportError:
    from data_loader import SeasonData
    from evaluation import (
        SystemMetrics,
        compute_composite_score,
        compute_controversy_correction,
        compute_engagement,
        compute_technical_fairness,
    )
    from model import PTFSParams, PTFSSystem, simulate_all_seasons


@dataclass
class OptimizationResult:
    """优化结果。"""

    best_params: PTFSParams
    best_score: float
    all_results: list[tuple[PTFSParams, float]]
    sensitivity_analysis: dict


def grid_search(
    seasons_data: dict[int, SeasonData],
    param_ranges: dict[str, list[float]] | None = None,
    objective_weights: dict[str, float] | None = None,
    verbose: bool = True,
) -> OptimizationResult:
    """网格搜索优化参数。

    Parameters
    ----------
    seasons_data : dict[int, SeasonData]
        赛季数据
    param_ranges : dict[str, list[float]], optional
        参数搜索范围
    objective_weights : dict[str, float], optional
        目标函数权重
    verbose : bool
        是否打印进度

    Returns
    -------
    OptimizationResult
        优化结果
    """
    if param_ranges is None:
        param_ranges = {
            "judge_weight_start": [0.35, 0.40, 0.45, 0.50],
            "judge_weight_end": [0.55, 0.60, 0.65, 0.70, 0.75],
            "improvement_alpha": [0.0, 0.05, 0.10, 0.15, 0.20],
            "protection_threshold": [0.0, 0.15, 0.20, 0.25, 0.30],
        }

    # 生成参数组合
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]
    combinations = list(product(*param_values))

    if verbose:
        print(f"  搜索空间: {len(combinations)} 种参数组合")

    all_results: list[tuple[PTFSParams, float]] = []
    best_params: PTFSParams | None = None
    best_score = -np.inf

    for i, combo in enumerate(combinations):
        params_dict = dict(zip(param_names, combo))

        # 确保 weight_start < weight_end
        if params_dict["judge_weight_start"] >= params_dict["judge_weight_end"]:
            continue

        params = PTFSParams(**params_dict)
        system = PTFSSystem(params)

        # 模拟所有赛季
        results = simulate_all_seasons(seasons_data, system)

        # 计算指标
        technical = compute_technical_fairness(seasons_data, results)
        engagement = compute_engagement(seasons_data, results)
        controversy = compute_controversy_correction(seasons_data, results)
        score = compute_composite_score(
            technical, engagement, controversy, objective_weights
        )

        all_results.append((params, score))

        if score > best_score:
            best_score = score
            best_params = params

        if verbose and (i + 1) % 50 == 0:
            print(f"    进度: {i + 1}/{len(combinations)}, 当前最优: {best_score:.4f}")

    # 敏感性分析
    sensitivity = run_sensitivity_analysis(seasons_data, best_params, objective_weights)

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=all_results,
        sensitivity_analysis=sensitivity,
    )


def run_sensitivity_analysis(
    seasons_data: dict[int, SeasonData],
    base_params: PTFSParams,
    objective_weights: dict[str, float] | None = None,
    perturbation: float = 0.1,
) -> dict:
    """敏感性分析。

    Parameters
    ----------
    seasons_data : dict[int, SeasonData]
        赛季数据
    base_params : PTFSParams
        基准参数
    objective_weights : dict[str, float], optional
        目标函数权重
    perturbation : float
        扰动幅度

    Returns
    -------
    dict
        敏感性分析结果
    """
    base_dict = base_params.to_dict()
    base_system = PTFSSystem(base_params)
    base_results = simulate_all_seasons(seasons_data, base_system)
    base_technical = compute_technical_fairness(seasons_data, base_results)
    base_engagement = compute_engagement(seasons_data, base_results)
    base_controversy = compute_controversy_correction(seasons_data, base_results)
    base_score = compute_composite_score(
        base_technical, base_engagement, base_controversy, objective_weights
    )

    sensitivity = {
        "base_score": base_score,
        "parameters": {},
    }

    for param_name, base_value in base_dict.items():
        param_sensitivity = {
            "base_value": base_value,
            "perturbations": [],
        }

        # 正负扰动
        for direction in [-1, 1]:
            perturbed_value = base_value * (1 + direction * perturbation)

            # 限制范围
            if param_name in ["judge_weight_start", "judge_weight_end"]:
                perturbed_value = max(0.1, min(0.9, perturbed_value))
            elif param_name == "improvement_alpha":
                perturbed_value = max(0.0, min(0.5, perturbed_value))
            elif param_name == "protection_threshold":
                perturbed_value = max(0.0, min(0.5, perturbed_value))

            perturbed_dict = base_dict.copy()
            perturbed_dict[param_name] = perturbed_value

            # 确保有效组合
            if (
                perturbed_dict["judge_weight_start"]
                >= perturbed_dict["judge_weight_end"]
            ):
                continue

            perturbed_params = PTFSParams(**perturbed_dict)
            perturbed_system = PTFSSystem(perturbed_params)
            perturbed_results = simulate_all_seasons(seasons_data, perturbed_system)

            technical = compute_technical_fairness(seasons_data, perturbed_results)
            engagement = compute_engagement(seasons_data, perturbed_results)
            controversy = compute_controversy_correction(
                seasons_data, perturbed_results
            )
            perturbed_score = compute_composite_score(
                technical, engagement, controversy, objective_weights
            )

            score_change = perturbed_score - base_score
            relative_change = score_change / base_score if base_score != 0 else 0

            param_sensitivity["perturbations"].append({
                "direction": "+" if direction > 0 else "-",
                "perturbed_value": perturbed_value,
                "score": perturbed_score,
                "score_change": score_change,
                "relative_change": relative_change,
            })

        # 计算敏感度
        if param_sensitivity["perturbations"]:
            changes = [p["relative_change"] for p in param_sensitivity["perturbations"]]
            param_sensitivity["sensitivity_index"] = np.std(changes)
        else:
            param_sensitivity["sensitivity_index"] = 0.0

        sensitivity["parameters"][param_name] = param_sensitivity

    # 排序参数敏感度
    sorted_params = sorted(
        sensitivity["parameters"].items(),
        key=lambda x: x[1]["sensitivity_index"],
        reverse=True,
    )
    sensitivity["ranked_parameters"] = [
        {"name": name, "sensitivity": data["sensitivity_index"]}
        for name, data in sorted_params
    ]

    return sensitivity


def evaluate_params(
    params: PTFSParams,
    seasons_data: dict[int, SeasonData],
    objective_weights: dict[str, float] | None = None,
) -> SystemMetrics:
    """评估指定参数。

    Parameters
    ----------
    params : PTFSParams
        参数
    seasons_data : dict[int, SeasonData]
        赛季数据
    objective_weights : dict[str, float], optional
        目标函数权重

    Returns
    -------
    SystemMetrics
        系统指标
    """
    system = PTFSSystem(params)
    results = simulate_all_seasons(seasons_data, system)

    technical = compute_technical_fairness(seasons_data, results)
    engagement = compute_engagement(seasons_data, results)
    controversy = compute_controversy_correction(seasons_data, results)
    composite = compute_composite_score(
        technical, engagement, controversy, objective_weights
    )

    return SystemMetrics(
        system_name=f"PTFS({params.judge_weight_start:.2f}-{params.judge_weight_end:.2f})",
        technical_fairness=technical,
        engagement=engagement,
        controversy=controversy,
        composite_score=composite,
    )


def fine_tune_around_best(
    seasons_data: dict[int, SeasonData],
    initial_params: PTFSParams,
    objective_weights: dict[str, float] | None = None,
    step_size: float = 0.02,
    max_iterations: int = 20,
    verbose: bool = True,
) -> OptimizationResult:
    """在最优参数附近精细调优。

    Parameters
    ----------
    seasons_data : dict[int, SeasonData]
        赛季数据
    initial_params : PTFSParams
        初始参数
    objective_weights : dict[str, float], optional
        目标函数权重
    step_size : float
        步长
    max_iterations : int
        最大迭代次数
    verbose : bool
        是否打印进度

    Returns
    -------
    OptimizationResult
        优化结果
    """
    current_params = initial_params
    current_score = evaluate_params(
        current_params, seasons_data, objective_weights
    ).composite_score

    all_results = [(current_params, current_score)]

    for iteration in range(max_iterations):
        improved = False
        current_dict = current_params.to_dict()

        for param_name in current_dict.keys():
            for direction in [-1, 1]:
                new_dict = current_dict.copy()
                new_dict[param_name] = current_dict[param_name] + direction * step_size

                # 边界检查
                if param_name in ["judge_weight_start", "judge_weight_end"]:
                    new_dict[param_name] = max(0.2, min(0.8, new_dict[param_name]))
                elif param_name == "improvement_alpha":
                    new_dict[param_name] = max(0.0, min(0.4, new_dict[param_name]))
                elif param_name == "protection_threshold":
                    new_dict[param_name] = max(0.0, min(0.4, new_dict[param_name]))

                # 约束检查
                if new_dict["judge_weight_start"] >= new_dict["judge_weight_end"]:
                    continue

                new_params = PTFSParams(**new_dict)
                new_score = evaluate_params(
                    new_params, seasons_data, objective_weights
                ).composite_score

                all_results.append((new_params, new_score))

                if new_score > current_score:
                    current_params = new_params
                    current_score = new_score
                    current_dict = new_dict
                    improved = True
                    if verbose:
                        print(
                            f"    迭代 {iteration + 1}: "
                            f"改进 {param_name} -> {new_score:.4f}"
                        )
                    break

            if improved:
                break

        if not improved:
            if verbose:
                print(f"    迭代 {iteration + 1}: 收敛")
            break

    sensitivity = run_sensitivity_analysis(
        seasons_data, current_params, objective_weights
    )

    return OptimizationResult(
        best_params=current_params,
        best_score=current_score,
        all_results=all_results,
        sensitivity_analysis=sensitivity,
    )
