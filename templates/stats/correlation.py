"""
相关分析模块
============
"""

from typing import Literal

import numpy as np
import polars as pl
from scipy import stats


def correlation_test(
    df: pl.DataFrame,
    x: str,
    y: str,
    *,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> dict:
    """
    计算相关系数并进行显著性检验

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    x, y : str
        要计算相关的两列名
    method : str, default "pearson"
        相关系数类型："pearson", "spearman", "kendall"

    Returns
    -------
    dict
        包含相关系数、p 值、置信区间等

    Examples
    --------
    >>> result = correlation_test(df, "gdp", "medals", method="spearman")
    >>> print(f"r = {result['correlation']:.3f}, p = {result['p_value']:.4f}")
    """
    # 获取数据并去除缺失值
    data = df.select([x, y]).drop_nulls()
    x_vals = data[x].to_numpy()
    y_vals = data[y].to_numpy()

    n = len(x_vals)

    if method == "pearson":
        r, p = stats.pearsonr(x_vals, y_vals)
        # 计算置信区间（Fisher z-transform）
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_ci = stats.norm.ppf([0.025, 0.975])
        ci = np.tanh(z + z_ci * se)
    elif method == "spearman":
        r, p = stats.spearmanr(x_vals, y_vals)
        ci = [np.nan, np.nan]  # Spearman CI 需要 bootstrap
    elif method == "kendall":
        r, p = stats.kendalltau(x_vals, y_vals)
        ci = [np.nan, np.nan]

    # 相关强度判断
    abs_r = abs(r)
    if abs_r >= 0.8:
        strength = "very strong"
    elif abs_r >= 0.6:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.2:
        strength = "weak"
    else:
        strength = "very weak"

    return {
        "x": x,
        "y": y,
        "method": method,
        "correlation": float(r),
        "p_value": float(p),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "n": n,
        "strength": strength,
        "significant": p < 0.05,
        "interpretation": f"{strength} {'positive' if r > 0 else 'negative'} correlation",
    }


def partial_correlation(
    df: pl.DataFrame,
    x: str,
    y: str,
    control: list[str],
) -> dict:
    """
    计算偏相关系数（控制其他变量的影响）

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    x, y : str
        要计算偏相关的两个变量
    control : list[str]
        要控制的变量列表

    Returns
    -------
    dict
        偏相关系数和 p 值

    Examples
    --------
    >>> # 控制 GDP 后，人口和奖牌的偏相关
    >>> result = partial_correlation(df, "population", "medals", control=["gdp"])
    """
    from sklearn.linear_model import LinearRegression

    data = df.select([x, y] + control).drop_nulls()

    x_vals = data[x].to_numpy().reshape(-1, 1)
    y_vals = data[y].to_numpy().reshape(-1, 1)
    z_vals = data.select(control).to_numpy()

    # 计算残差
    reg_x = LinearRegression().fit(z_vals, x_vals)
    reg_y = LinearRegression().fit(z_vals, y_vals)

    x_residual = x_vals - reg_x.predict(z_vals)
    y_residual = y_vals - reg_y.predict(z_vals)

    # 计算残差的相关系数
    r, p = stats.pearsonr(x_residual.ravel(), y_residual.ravel())

    return {
        "x": x,
        "y": y,
        "control_variables": control,
        "partial_correlation": float(r),
        "p_value": float(p),
        "significant": p < 0.05,
    }


def correlation_summary(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    *,
    method: Literal["pearson", "spearman"] = "pearson",
    threshold: float = 0.5,
) -> pl.DataFrame:
    """
    批量计算相关系数并筛选显著相关对

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    columns : list[str], optional
        要分析的列，None 表示所有数值列
    method : str, default "pearson"
        相关系数类型
    threshold : float, default 0.5
        筛选阈值（|r| > threshold）

    Returns
    -------
    pl.DataFrame
        显著相关对列表

    Examples
    --------
    >>> summary = correlation_summary(df, threshold=0.6)
    >>> print(summary)
    """
    if columns is None:
        columns = [col for col in df.columns if df[col].dtype.is_numeric()]

    results = []

    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1 :]:
            result = correlation_test(df, col1, col2, method=method)
            if abs(result["correlation"]) >= threshold:
                results.append(
                    {
                        "var1": col1,
                        "var2": col2,
                        "correlation": result["correlation"],
                        "p_value": result["p_value"],
                        "significant": result["significant"],
                        "strength": result["strength"],
                    }
                )

    if not results:
        return pl.DataFrame(
            {
                "var1": [],
                "var2": [],
                "correlation": [],
                "p_value": [],
                "significant": [],
                "strength": [],
            }
        )

    return pl.DataFrame(results).sort("correlation", descending=True)
