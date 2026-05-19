"""
敏感性分析模块
==============
"""

from typing import Callable

import numpy as np
import polars as pl


def sensitivity_analysis(
    model_func: Callable,
    base_params: dict[str, float],
    param_to_analyze: str,
    *,
    variation_range: tuple[float, float] = (-0.3, 0.3),
    n_steps: int = 20,
    output_name: str = "output",
) -> pl.DataFrame:
    """
    单参数敏感性分析

    Parameters
    ----------
    model_func : Callable
        模型函数，接受参数字典，返回输出值
    base_params : dict
        基准参数值
    param_to_analyze : str
        要分析的参数名
    variation_range : tuple, default (-0.3, 0.3)
        变化范围（相对于基准值的比例）
    n_steps : int, default 20
        步数
    output_name : str, default "output"
        输出变量名

    Returns
    -------
    pl.DataFrame
        敏感性分析结果

    Examples
    --------
    >>> def my_model(params):
    ...     return params["a"] * params["b"] + params["c"]
    >>>
    >>> base = {"a": 1.0, "b": 2.0, "c": 3.0}
    >>> result = sensitivity_analysis(my_model, base, "a", variation_range=(-0.5, 0.5))
    """
    base_value = base_params[param_to_analyze]
    base_output = model_func(base_params)

    # 计算变化范围
    min_mult = 1 + variation_range[0]
    max_mult = 1 + variation_range[1]
    multipliers = np.linspace(min_mult, max_mult, n_steps)

    results = []

    for mult in multipliers:
        # 修改参数
        params = base_params.copy()
        params[param_to_analyze] = base_value * mult

        # 计算输出
        output = model_func(params)

        # 计算变化率
        param_change_pct = (mult - 1) * 100
        output_change_pct = (
            (output - base_output) / base_output * 100 if base_output != 0 else 0
        )

        results.append(
            {
                "param_name": param_to_analyze,
                "param_value": params[param_to_analyze],
                "param_change_pct": param_change_pct,
                output_name: output,
                f"{output_name}_change_pct": output_change_pct,
                "sensitivity": output_change_pct / param_change_pct
                if param_change_pct != 0
                else 0,
            }
        )

    return pl.DataFrame(results)


def parameter_sweep(
    model_func: Callable,
    param_ranges: dict[str, tuple[float, float, int]],
    *,
    output_name: str = "output",
) -> pl.DataFrame:
    """
    多参数网格搜索

    Parameters
    ----------
    model_func : Callable
        模型函数
    param_ranges : dict
        参数范围，格式为 {参数名: (最小值, 最大值, 步数)}
    output_name : str, default "output"
        输出变量名

    Returns
    -------
    pl.DataFrame
        参数组合和对应输出

    Examples
    --------
    >>> def my_model(params):
    ...     return params["a"] ** 2 + params["b"]
    >>>
    >>> ranges = {"a": (0, 10, 11), "b": (0, 5, 6)}
    >>> result = parameter_sweep(my_model, ranges)
    """
    from itertools import product

    # 生成参数值
    param_names = list(param_ranges.keys())
    param_values = [
        np.linspace(rng[0], rng[1], rng[2]) for rng in param_ranges.values()
    ]

    results = []

    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        output = model_func(params)

        result = params.copy()
        result[output_name] = output
        results.append(result)

    return pl.DataFrame(results)


def tornado_diagram_data(
    model_func: Callable,
    base_params: dict[str, float],
    params_to_analyze: list[str] | None = None,
    *,
    variation: float = 0.2,
) -> pl.DataFrame:
    """
    龙卷风图数据（用于可视化敏感性）

    Parameters
    ----------
    model_func : Callable
        模型函数
    base_params : dict
        基准参数值
    params_to_analyze : list[str], optional
        要分析的参数列表，None 表示所有
    variation : float, default 0.2
        变化比例（± 20%）

    Returns
    -------
    pl.DataFrame
        龙卷风图数据

    Examples
    --------
    >>> tornado_data = tornado_diagram_data(my_model, base_params)
    >>> # 可用 bar_plot 绘制龙卷风图
    """
    if params_to_analyze is None:
        params_to_analyze = list(base_params.keys())

    base_output = model_func(base_params)
    results = []

    for param in params_to_analyze:
        base_value = base_params[param]

        # 低值情况
        params_low = base_params.copy()
        params_low[param] = base_value * (1 - variation)
        output_low = model_func(params_low)

        # 高值情况
        params_high = base_params.copy()
        params_high[param] = base_value * (1 + variation)
        output_high = model_func(params_high)

        # 计算影响范围
        swing = abs(output_high - output_low)

        results.append(
            {
                "parameter": param,
                "base_value": base_value,
                "low_value": params_low[param],
                "high_value": params_high[param],
                "output_at_low": output_low,
                "output_at_high": output_high,
                "output_base": base_output,
                "swing": swing,
                "sensitivity_index": swing / base_output if base_output != 0 else 0,
            }
        )

    return pl.DataFrame(results).sort("swing", descending=True)


def monte_carlo_sensitivity(
    model_func: Callable,
    param_distributions: dict[str, tuple[str, tuple]],
    *,
    n_samples: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    蒙特卡洛敏感性分析（全局敏感性）

    Parameters
    ----------
    model_func : Callable
        模型函数
    param_distributions : dict
        参数分布，格式为 {参数名: (分布类型, 参数)}
        支持的分布："uniform", "normal", "triangular"
    n_samples : int, default 1000
        采样次数
    random_state : int, default 42
        随机种子

    Returns
    -------
    dict
        包含敏感性指标和采样结果

    Examples
    --------
    >>> distributions = {
    ...     "a": ("uniform", (0.8, 1.2)),      # 均匀分布 [0.8, 1.2]
    ...     "b": ("normal", (10, 2)),          # 正态分布 μ=10，σ=2
    ...     "c": ("triangular", (5, 10, 15)),  # 三角分布 min=5，mode=10，max=15
    ... }
    >>> result = monte_carlo_sensitivity(my_model, distributions)
    """
    np.random.seed(random_state)

    param_names = list(param_distributions.keys())
    samples = {}

    # 采样
    for name, (dist_type, params) in param_distributions.items():
        if dist_type == "uniform":
            samples[name] = np.random.uniform(params[0], params[1], n_samples)
        elif dist_type == "normal":
            samples[name] = np.random.normal(params[0], params[1], n_samples)
        elif dist_type == "triangular":
            samples[name] = np.random.triangular(
                params[0], params[1], params[2], n_samples
            )

    # 运行模型
    outputs = []
    for i in range(n_samples):
        param_values = {name: samples[name][i] for name in param_names}
        outputs.append(model_func(param_values))

    outputs = np.array(outputs)

    # 计算相关系数（Pearson）作为敏感性指标
    sensitivity = {}
    for name in param_names:
        corr = np.corrcoef(samples[name], outputs)[0, 1]
        sensitivity[name] = {
            "correlation": float(corr),
            "rank_correlation": float(
                np.corrcoef(
                    np.argsort(samples[name]).argsort(), np.argsort(outputs).argsort()
                )[0, 1]
            ),
        }

    # 构建采样 DataFrame
    sample_df = pl.DataFrame(samples).with_columns(pl.Series("output", outputs))

    return {
        "sensitivity": sensitivity,
        "output_stats": {
            "mean": float(outputs.mean()),
            "std": float(outputs.std()),
            "min": float(outputs.min()),
            "max": float(outputs.max()),
            "p5": float(np.percentile(outputs, 5)),
            "p95": float(np.percentile(outputs, 95)),
        },
        "samples": sample_df,
        "n_samples": n_samples,
    }
