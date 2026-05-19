"""
数据探索模块
============

提供数据统计、缺失值报告、相关性分析等功能。
"""

from typing import Literal

import numpy as np
import polars as pl


def describe_df(
    df: pl.DataFrame,
    *,
    include_categorical: bool = True,
    percentiles: list[float] | None = None,
) -> pl.DataFrame:
    """
    生成数据描述统计

    Parameters
    ----------
    df : pl.DataFrame
        输入数据框
    include_categorical : bool, default True
        是否包含分类列统计
    percentiles : list[float], optional
        要计算的百分位数，默认 [0.25, 0.5, 0.75]

    Returns
    -------
    pl.DataFrame
        统计信息数据框

    Examples
    --------
    >>> stats = describe_df(df)
    >>> print(stats)
    """
    if percentiles is None:
        percentiles = [0.25, 0.5, 0.75]

    stats_list = []

    for col in df.columns:
        col_stats = {"column": col, "dtype": str(df[col].dtype)}
        series = df[col]

        col_stats["count"] = len(series)
        col_stats["null_count"] = series.null_count()
        col_stats["null_pct"] = round(series.null_count() / len(series) * 100, 2)
        col_stats["unique_count"] = series.n_unique()

        if series.dtype.is_numeric():
            col_stats["mean"] = series.mean()
            col_stats["std"] = series.std()
            col_stats["min"] = series.min()
            col_stats["max"] = series.max()

            for p in percentiles:
                col_stats[f"p{int(p * 100)}"] = series.quantile(p)

        elif include_categorical:
            # 分类列: 最频繁值
            value_counts = series.value_counts().sort("count", descending=True)
            if len(value_counts) > 0:
                col_stats["top_value"] = str(value_counts[series.name][0])
                col_stats["top_freq"] = value_counts["count"][0]

        stats_list.append(col_stats)

    return pl.DataFrame(stats_list)


def missing_report(df: pl.DataFrame) -> pl.DataFrame:
    """
    生成缺失值报告

    Parameters
    ----------
    df : pl.DataFrame
        输入数据框

    Returns
    -------
    pl.DataFrame
        缺失值报告，包含列名、缺失数、缺失率

    Examples
    --------
    >>> report = missing_report(df)
    >>> print(report.filter(pl.col("missing_pct") > 0))
    """
    n_rows = len(df)

    report_data = []
    for col in df.columns:
        null_count = df[col].null_count()
        report_data.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": null_count,
                "missing_pct": round(null_count / n_rows * 100, 2),
                "non_missing": n_rows - null_count,
            }
        )

    return pl.DataFrame(report_data).sort("missing_pct", descending=True)


def correlation_matrix(
    df: pl.DataFrame,
    *,
    columns: list[str] | None = None,
    method: Literal["pearson", "spearman"] = "pearson",
) -> pl.DataFrame:
    """
    计算相关系数矩阵

    Parameters
    ----------
    df : pl.DataFrame
        输入数据框
    columns : list[str], optional
        要计算相关的列，None 表示所有数值列
    method : str, default "pearson"
        相关系数方法: "pearson" 或 "spearman"

    Returns
    -------
    pl.DataFrame
        相关系数矩阵

    Examples
    --------
    >>> corr = correlation_matrix(df, columns=["price", "quantity", "rating"])
    >>> corr = correlation_matrix(df, method="spearman")
    """
    # 获取数值列
    if columns is None:
        columns = [col for col in df.columns if df[col].dtype.is_numeric()]
    else:
        columns = [
            col for col in columns if col in df.columns and df[col].dtype.is_numeric()
        ]

    if len(columns) < 2:
        raise ValueError("Need at least 2 numeric columns to compute correlation")

    # 转换为 numpy 计算相关系数
    data = df.select(columns).drop_nulls().to_numpy()

    if method == "pearson":
        corr_matrix = np.corrcoef(data, rowvar=False)
    elif method == "spearman":
        from scipy import stats

        corr_matrix, _ = stats.spearmanr(data)
        if len(columns) == 2:
            corr_matrix = np.array([[1, corr_matrix], [corr_matrix, 1]])

    # 转换为 DataFrame
    corr_df = pl.DataFrame(
        {col: corr_matrix[:, i] for i, col in enumerate(columns)}
    ).with_columns(pl.Series("variable", columns))

    # 重新排列列顺序
    return corr_df.select(["variable"] + columns)


def detect_outliers(
    df: pl.DataFrame,
    *,
    columns: list[str] | None = None,
    method: Literal["iqr", "zscore"] = "iqr",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
) -> pl.DataFrame:
    """
    检测异常值

    Parameters
    ----------
    df : pl.DataFrame
        输入数据框
    columns : list[str], optional
        要检测的列，None 表示所有数值列
    method : str, default "iqr"
        检测方法
    iqr_multiplier : float, default 1.5
        IQR 乘数
    zscore_threshold : float, default 3.0
        Z-score 阈值

    Returns
    -------
    pl.DataFrame
        异常值报告，包含列名、异常值数量、异常值比例、边界值

    Examples
    --------
    >>> outliers = detect_outliers(df, method="iqr")
    >>> print(outliers)
    """
    if columns is None:
        columns = [col for col in df.columns if df[col].dtype.is_numeric()]

    report_data = []

    for col in columns:
        if col not in df.columns or not df[col].dtype.is_numeric():
            continue

        series = df[col].drop_nulls()
        n_total = len(series)

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
        else:  # zscore
            mean = series.mean()
            std = series.std()
            lower = mean - zscore_threshold * std
            upper = mean + zscore_threshold * std

        n_outliers = series.filter((series < lower) | (series > upper)).len()

        report_data.append(
            {
                "column": col,
                "method": method,
                "outlier_count": n_outliers,
                "outlier_pct": round(n_outliers / n_total * 100, 2)
                if n_total > 0
                else 0,
                "lower_bound": round(lower, 4) if lower is not None else None,
                "upper_bound": round(upper, 4) if upper is not None else None,
                "actual_min": series.min(),
                "actual_max": series.max(),
            }
        )

    return pl.DataFrame(report_data).sort("outlier_pct", descending=True)
