"""
数据预处理模块
==============

提供缺失值处理、异常值处理、标准化等功能。
"""

from typing import Literal

import polars as pl


def handle_missing(
    df: pl.DataFrame,
    *,
    strategy: Literal[
        "drop", "mean", "median", "mode", "zero", "ffill", "bfill"
    ] = "mean",
    columns: list[str] | None = None,
    threshold: float | None = None,
) -> pl.DataFrame:
    """
    处理缺失值

    Parameters
    ----------
    df : pl.DataFrame
        输入数据框
    strategy : str, default "mean"
        填充策略:
        - "drop": 删除含缺失值的行
        - "mean": 用均值填充 (仅数值列)
        - "median": 用中位数填充 (仅数值列)
        - "mode": 用众数填充
        - "zero": 用 0 填充
        - "ffill": 前向填充
        - "bfill": 后向填充
    columns : list[str], optional
        要处理的列，None 表示所有列
    threshold : float, optional
        缺失率阈值，超过此值的列将被删除

    Returns
    -------
    pl.DataFrame
        处理后的数据框

    Examples
    --------
    >>> df = handle_missing(df, strategy="mean")
    >>> df = handle_missing(df, strategy="median", columns=["price", "quantity"])
    >>> df = handle_missing(df, threshold=0.3)  # 删除缺失率超过 30% 的列
    """
    result = df.clone()

    # 首先处理高缺失率的列
    if threshold is not None:
        n_rows = len(result)
        cols_to_drop = [
            col
            for col in result.columns
            if result[col].null_count() / n_rows > threshold
        ]
        if cols_to_drop:
            result = result.drop(cols_to_drop)

    # 确定要处理的列
    if columns is None:
        columns = result.columns

    # 获取数值列
    numeric_cols = [
        col
        for col in columns
        if col in result.columns and result[col].dtype.is_numeric()
    ]

    # 根据策略处理
    if strategy == "drop":
        result = result.drop_nulls(subset=columns)

    elif strategy == "mean":
        for col in numeric_cols:
            mean_val = result[col].mean()
            if mean_val is not None:
                result = result.with_columns(pl.col(col).fill_null(mean_val))

    elif strategy == "median":
        for col in numeric_cols:
            median_val = result[col].median()
            if median_val is not None:
                result = result.with_columns(pl.col(col).fill_null(median_val))

    elif strategy == "mode":
        for col in columns:
            if col in result.columns:
                mode_val = (
                    result[col].mode().item(0) if len(result[col].mode()) > 0 else None
                )
                if mode_val is not None:
                    result = result.with_columns(pl.col(col).fill_null(mode_val))

    elif strategy == "zero":
        for col in numeric_cols:
            result = result.with_columns(pl.col(col).fill_null(0))

    elif strategy == "ffill":
        for col in columns:
            if col in result.columns:
                result = result.with_columns(pl.col(col).forward_fill())

    elif strategy == "bfill":
        for col in columns:
            if col in result.columns:
                result = result.with_columns(pl.col(col).backward_fill())

    return result


def handle_outliers(
    df: pl.DataFrame,
    *,
    method: Literal["iqr", "zscore", "clip"] = "iqr",
    columns: list[str] | None = None,
    action: Literal["remove", "clip", "nan"] = "clip",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
) -> pl.DataFrame:
    """
    处理异常值

    Parameters
    ----------
    df : pl.DataFrame
        输入数据框
    method : str, default "iqr"
        检测方法:
        - "iqr": 四分位距法 (Q1 - k*IQR, Q3 + k*IQR)
        - "zscore": Z-score 法 (|z| > threshold)
        - "clip": 直接按百分位裁剪
    columns : list[str], optional
        要处理的列，None 表示所有数值列
    action : str, default "clip"
        处理方式:
        - "remove": 删除含异常值的行
        - "clip": 将异常值裁剪到边界
        - "nan": 将异常值设为 NaN
    iqr_multiplier : float, default 1.5
        IQR 方法的乘数
    zscore_threshold : float, default 3.0
        Z-score 方法的阈值

    Returns
    -------
    pl.DataFrame
        处理后的数据框

    Examples
    --------
    >>> df = handle_outliers(df, method="iqr", action="clip")
    >>> df = handle_outliers(df, method="zscore", columns=["price"], action="remove")
    """
    result = df.clone()

    # 获取要处理的数值列
    if columns is None:
        columns = [col for col in result.columns if result[col].dtype.is_numeric()]
    else:
        columns = [
            col
            for col in columns
            if col in result.columns and result[col].dtype.is_numeric()
        ]

    for col in columns:
        series = result[col]

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr

        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            lower = mean - zscore_threshold * std
            upper = mean + zscore_threshold * std

        elif method == "clip":
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)

        # 执行处理
        if action == "clip":
            result = result.with_columns(pl.col(col).clip(lower, upper))
        elif action == "nan":
            result = result.with_columns(
                pl.when((pl.col(col) < lower) | (pl.col(col) > upper))
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
        elif action == "remove":
            result = result.filter((pl.col(col) >= lower) & (pl.col(col) <= upper))

    return result


def normalize(
    df: pl.DataFrame,
    *,
    columns: list[str] | None = None,
    method: Literal["minmax", "robust"] = "minmax",
    feature_range: tuple[float, float] = (0, 1),
) -> tuple[pl.DataFrame, dict]:
    """
    归一化数值列

    Parameters
    ----------
    df : pl.DataFrame
        输入数据框
    columns : list[str], optional
        要归一化的列，None 表示所有数值列
    method : str, default "minmax"
        归一化方法:
        - "minmax": Min-Max 归一化
        - "robust": 基于中位数和 IQR 的鲁棒归一化
    feature_range : tuple, default (0, 1)
        目标范围 (minmax 方法)

    Returns
    -------
    tuple[pl.DataFrame, dict]
        (归一化后的数据框, 归一化参数字典)

    Examples
    --------
    >>> df_norm, params = normalize(df, columns=["price", "quantity"])
    >>> # 使用 params 可以对新数据进行相同的归一化
    """
    result = df.clone()
    params = {}

    # 获取要处理的数值列
    if columns is None:
        columns = [col for col in result.columns if result[col].dtype.is_numeric()]

    min_val, max_val = feature_range

    for col in columns:
        if col not in result.columns or not result[col].dtype.is_numeric():
            continue

        series = result[col]

        if method == "minmax":
            col_min = series.min()
            col_max = series.max()

            if col_max - col_min > 0:
                result = result.with_columns(
                    (
                        (pl.col(col) - col_min)
                        / (col_max - col_min)
                        * (max_val - min_val)
                        + min_val
                    ).alias(col)
                )

            params[col] = {
                "method": "minmax",
                "min": col_min,
                "max": col_max,
                "range": feature_range,
            }

        elif method == "robust":
            median = series.median()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr > 0:
                result = result.with_columns(((pl.col(col) - median) / iqr).alias(col))

            params[col] = {"method": "robust", "median": median, "iqr": iqr}

    return result, params


def standardize(
    df: pl.DataFrame,
    *,
    columns: list[str] | None = None,
) -> tuple[pl.DataFrame, dict]:
    """
    标准化数值列 (Z-score 标准化)

    Parameters
    ----------
    df : pl.DataFrame
        输入数据框
    columns : list[str], optional
        要标准化的列，None 表示所有数值列

    Returns
    -------
    tuple[pl.DataFrame, dict]
        (标准化后的数据框, 标准化参数字典)

    Examples
    --------
    >>> df_std, params = standardize(df, columns=["price", "quantity"])
    """
    result = df.clone()
    params = {}

    if columns is None:
        columns = [col for col in result.columns if result[col].dtype.is_numeric()]

    for col in columns:
        if col not in result.columns or not result[col].dtype.is_numeric():
            continue

        series = result[col]
        mean = series.mean()
        std = series.std()

        if std > 0:
            result = result.with_columns(((pl.col(col) - mean) / std).alias(col))

        params[col] = {"mean": mean, "std": std}

    return result, params


def preprocess_pipeline(
    df: pl.DataFrame,
    *,
    drop_high_missing: float | None = 0.5,
    handle_missing_strategy: Literal[
        "drop", "mean", "median", "mode", "ffill"
    ] = "mean",
    handle_outliers_method: Literal["iqr", "zscore", "clip"] | None = "iqr",
    outlier_action: Literal["remove", "clip", "nan"] = "clip",
    normalize_method: Literal["minmax", "zscore", "robust"] | None = None,
    normalize_columns: list[str] | None = None,
) -> tuple[pl.DataFrame, dict]:
    """
    一键数据预处理流水线

    按顺序执行: 删除高缺失列 -> 处理缺失值 -> 处理异常值 -> 标准化/归一化

    Parameters
    ----------
    df : pl.DataFrame
        输入数据框
    drop_high_missing : float, optional
        删除缺失率超过此值的列，None 表示不删除
    handle_missing_strategy : str, default "mean"
        缺失值处理策略
    handle_outliers_method : str, optional
        异常值检测方法，None 表示不处理
    outlier_action : str, default "clip"
        异常值处理方式
    normalize_method : str, optional
        归一化方法，None 表示不归一化
    normalize_columns : list[str], optional
        要归一化的列

    Returns
    -------
    tuple[pl.DataFrame, dict]
        (处理后的数据框, 处理参数字典)

    Examples
    --------
    >>> df_clean, params = preprocess_pipeline(
    ...     df,
    ...     drop_high_missing=0.3,
    ...     handle_missing_strategy="median",
    ...     handle_outliers_method="iqr",
    ...     normalize_method="minmax"
    ... )
    """
    result = df.clone()
    pipeline_params = {"steps": []}

    # Step 1: 删除高缺失率列
    if drop_high_missing is not None:
        result = handle_missing(result, threshold=drop_high_missing, strategy="drop")
        pipeline_params["steps"].append(
            f"drop_high_missing(threshold={drop_high_missing})"
        )

    # Step 2: 处理缺失值
    result = handle_missing(result, strategy=handle_missing_strategy)
    pipeline_params["steps"].append(
        f"handle_missing(strategy={handle_missing_strategy})"
    )

    # Step 3: 处理异常值
    if handle_outliers_method is not None:
        result = handle_outliers(
            result, method=handle_outliers_method, action=outlier_action
        )
        pipeline_params["steps"].append(
            f"handle_outliers(method={handle_outliers_method}, action={outlier_action})"
        )

    # Step 4: 归一化/标准化
    if normalize_method is not None:
        if normalize_method == "zscore":
            result, norm_params = standardize(result, columns=normalize_columns)
        else:
            result, norm_params = normalize(
                result, columns=normalize_columns, method=normalize_method
            )
        pipeline_params["normalize_params"] = norm_params
        pipeline_params["steps"].append(f"normalize(method={normalize_method})")

    return result, pipeline_params
