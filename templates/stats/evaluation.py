"""
评价模型模块（AHP/TOPSIS/熵权法）
================================
"""

import numpy as np
import polars as pl


def ahp_weight(
    comparison_matrix: np.ndarray | list[list[float]],
    *,
    criteria_names: list[str] | None = None,
    check_consistency: bool = True,
) -> dict:
    """
    层次分析法（AHP）计算权重

    Parameters
    ----------
    comparison_matrix : np.ndarray | list
        判断矩阵，满足 a[i][j] = 1/a[j][i]
        使用 1-9 标度：1= 同等重要，3= 稍重要，5= 明显重要，7= 强烈重要，9= 极端重要
    criteria_names : list[str], optional
        指标名称列表
    check_consistency : bool, default True
        是否检验一致性

    Returns
    -------
    dict
        包含权重、一致性比率等

    Examples
    --------
    >>> # 3个指标的判断矩阵
    >>> matrix = [
    ...     [1,   3,   5],
    ...     [1/3, 1,   3],
    ...     [1/5, 1/3, 1]
    ... ]
    >>> result = ahp_weight(matrix, criteria_names=["GDP", "Population", "Athletes"])
    >>> print(result["weights"])
    """
    A = np.array(comparison_matrix, dtype=float)
    n = A.shape[0]

    if criteria_names is None:
        criteria_names = [f"C{i + 1}" for i in range(n)]

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # 找最大特征值
    max_idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_idx].real

    # 对应的特征向量作为权重
    weights = eigenvectors[:, max_idx].real
    weights = weights / weights.sum()  # 归一化

    # 一致性检验
    CI = (lambda_max - n) / (n - 1)

    # 平均随机一致性指标
    RI_table = {
        1: 0,
        2: 0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }
    RI = RI_table.get(n, 1.49)

    CR = CI / RI if RI != 0 else 0

    result = {
        "weights": dict(zip(criteria_names, weights)),
        "weights_array": weights,
        "criteria": criteria_names,
        "lambda_max": float(lambda_max),
        "CI": float(CI),
        "RI": float(RI),
        "CR": float(CR),
        "is_consistent": CR < 0.1,
    }

    if check_consistency and CR >= 0.1:
        result["warning"] = (
            f"Consistency ratio CR={CR:.4f} >= 0.1, matrix needs revision"
        )

    return result


def entropy_weight(
    df: pl.DataFrame,
    columns: list[str],
    *,
    normalize: bool = True,
) -> dict:
    """
    熵权法计算客观权重

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    columns : list[str]
        参与计算的指标列
    normalize : bool, default True
        是否先归一化数据

    Returns
    -------
    dict
        包含各指标权重

    Examples
    --------
    >>> result = entropy_weight(df, ["gdp", "population", "athletes"])
    >>> print(result["weights"])
    """
    # 提取数据
    data = df.select(columns).drop_nulls().to_numpy()
    m, n = data.shape  # m: 样本数, n: 指标数

    # 归一化（Min-Max）
    if normalize:
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        range_vals = data_max - data_min
        range_vals[range_vals == 0] = 1  # 避免除零
        data = (data - data_min) / range_vals

    # 避免 log(0)
    data = np.clip(data, 1e-10, None)

    # 计算比重
    p = data / data.sum(axis=0)

    # 计算熵值
    k = 1 / np.log(m)
    e = -k * (p * np.log(p)).sum(axis=0)

    # 计算差异系数
    d = 1 - e

    # 计算权重
    weights = d / d.sum()

    return {
        "weights": dict(zip(columns, weights)),
        "weights_array": weights,
        "criteria": columns,
        "entropy": dict(zip(columns, e)),
        "diversity": dict(zip(columns, d)),
    }


def topsis(
    df: pl.DataFrame,
    columns: list[str],
    weights: list[float] | dict[str, float] | None = None,
    *,
    beneficial: list[str] | None = None,
    cost: list[str] | None = None,
    id_column: str | None = None,
) -> pl.DataFrame:
    """
    TOPSIS 多准则决策分析

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    columns : list[str]
        评价指标列
    weights : list[float] | dict | None
        指标权重，None 表示等权
    beneficial : list[str], optional
        正向指标（越大越好），默认所有指标
    cost : list[str], optional
        负向指标（越小越好）
    id_column : str, optional
        ID 列名，用于标识方案

    Returns
    -------
    pl.DataFrame
        添加了 TOPSIS 得分和排名的数据框

    Examples
    --------
    >>> # 对国家进行综合评价
    >>> result = topsis(
    ...     df,
    ...     columns=["gdp", "population", "medals"],
    ...     weights={"gdp": 0.3, "population": 0.3, "medals": 0.4},
    ...     cost=["population"]  # 人口是成本型指标
    ... )
    >>> print(result.sort("topsis_rank"))
    """
    # 准备数据
    data = df.select(columns).to_numpy().astype(float)
    m, n = data.shape

    # 处理权重
    if weights is None:
        w = np.ones(n) / n
    elif isinstance(weights, dict):
        w = np.array([weights[c] for c in columns])
    else:
        w = np.array(weights)

    # 正向/负向指标
    if beneficial is None and cost is None:
        beneficial = columns
        cost = []
    elif beneficial is None:
        beneficial = [c for c in columns if c not in cost]
    elif cost is None:
        cost = [c for c in columns if c not in beneficial]

    # 向量归一化
    norm = np.sqrt((data**2).sum(axis=0))
    norm[norm == 0] = 1
    data_norm = data / norm

    # 加权
    data_weighted = data_norm * w

    # 计算正理想解和负理想解
    ideal_pos = np.zeros(n)
    ideal_neg = np.zeros(n)

    for i, col in enumerate(columns):
        if col in beneficial:
            ideal_pos[i] = data_weighted[:, i].max()
            ideal_neg[i] = data_weighted[:, i].min()
        else:  # cost
            ideal_pos[i] = data_weighted[:, i].min()
            ideal_neg[i] = data_weighted[:, i].max()

    # 计算距离
    dist_pos = np.sqrt(((data_weighted - ideal_pos) ** 2).sum(axis=1))
    dist_neg = np.sqrt(((data_weighted - ideal_neg) ** 2).sum(axis=1))

    # 计算得分（相对接近度）
    scores = dist_neg / (dist_pos + dist_neg + 1e-10)

    # 排名
    ranks = scores.argsort()[::-1].argsort() + 1

    # 添加结果
    result = df.with_columns(
        [
            pl.Series("topsis_score", scores),
            pl.Series("topsis_rank", ranks),
            pl.Series("dist_to_ideal", dist_pos),
            pl.Series("dist_to_anti_ideal", dist_neg),
        ]
    )

    return result


def combine_weights(
    subjective: dict[str, float],
    objective: dict[str, float],
    *,
    alpha: float = 0.5,
) -> dict[str, float]:
    """
    组合主客观权重

    Parameters
    ----------
    subjective : dict
        主观权重（如 AHP）
    objective : dict
        客观权重（如熵权法）
    alpha : float, default 0.5
        主观权重的比例（0-1）

    Returns
    -------
    dict
        组合权重

    Examples
    --------
    >>> ahp_w = ahp_weight(matrix)["weights"]
    >>> entropy_w = entropy_weight(df, columns)["weights"]
    >>> combined = combine_weights(ahp_w, entropy_w, alpha=0.6)
    """
    combined = {}
    all_keys = set(subjective.keys()) | set(objective.keys())

    for key in all_keys:
        sub_w = subjective.get(key, 0)
        obj_w = objective.get(key, 0)
        combined[key] = alpha * sub_w + (1 - alpha) * obj_w

    # 重新归一化
    total = sum(combined.values())
    if total > 0:
        combined = {k: v / total for k, v in combined.items()}

    return combined
