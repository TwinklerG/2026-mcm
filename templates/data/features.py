"""
特征工程模块
============

提供时间特征、滞后特征、滚动特征等工程方法。
"""

from typing import Literal

import polars as pl


def create_time_features(
    df: pl.DataFrame,
    date_column: str,
    *,
    features: list[str] | None = None,
    drop_original: bool = False,
) -> pl.DataFrame:
    """
    从日期列创建时间特征

    Parameters
    ----------
    df : pl.DataFrame
        输入数据框
    date_column : str
        日期列名
    features : list[str], optional
        要创建的特征，默认全部:
        ["year", "month", "day", "weekday", "quarter", "week", "dayofyear", "is_weekend"]
    drop_original : bool, default False
        是否删除原始日期列

    Returns
    -------
    pl.DataFrame
        添加了时间特征的数据框

    Examples
    --------
    >>> df = create_time_features(df, "date", features=["year", "month", "weekday"])
    """
    if features is None:
        features = [
            "year",
            "month",
            "day",
            "weekday",
            "quarter",
            "week",
            "dayofyear",
            "is_weekend",
        ]

    result = df.clone()
    date_col = pl.col(date_column)

    feature_exprs = []

    if "year" in features:
        feature_exprs.append(date_col.dt.year().alias(f"{date_column}_year"))

    if "month" in features:
        feature_exprs.append(date_col.dt.month().alias(f"{date_column}_month"))

    if "day" in features:
        feature_exprs.append(date_col.dt.day().alias(f"{date_column}_day"))

    if "weekday" in features:
        feature_exprs.append(date_col.dt.weekday().alias(f"{date_column}_weekday"))

    if "quarter" in features:
        feature_exprs.append(date_col.dt.quarter().alias(f"{date_column}_quarter"))

    if "week" in features:
        feature_exprs.append(date_col.dt.week().alias(f"{date_column}_week"))

    if "dayofyear" in features:
        feature_exprs.append(
            date_col.dt.ordinal_day().alias(f"{date_column}_dayofyear")
        )

    if "is_weekend" in features:
        feature_exprs.append(
            (date_col.dt.weekday() >= 6).alias(f"{date_column}_is_weekend")
        )

    if feature_exprs:
        result = result.with_columns(feature_exprs)

    if drop_original:
        result = result.drop(date_column)

    return result


def create_lag_features(
    df: pl.DataFrame,
    columns: list[str],
    lags: list[int],
    *,
    group_by: str | list[str] | None = None,
    sort_by: str | None = None,
) -> pl.DataFrame:
    """
    创建滞后特征

    Parameters
    ----------
    df : pl.DataFrame
        输入数据框
    columns : list[str]
        要创建滞后特征的列
    lags : list[int]
        滞后期数列表，如 [1, 2, 3] 表示滞后 1、2、3 期
    group_by : str | list[str], optional
        分组列，用于分组内滞后
    sort_by : str, optional
        排序列（通常是时间列）

    Returns
    -------
    pl.DataFrame
        添加了滞后特征的数据框

    Examples
    --------
    >>> df = create_lag_features(df, ["sales"], lags=[1, 7, 30], sort_by="date")
    >>> df = create_lag_features(df, ["sales"], lags=[1, 2], group_by="store_id", sort_by="date")
    """
    result = df.clone()

    if sort_by is not None:
        result = result.sort(sort_by)

    lag_exprs = []
    for col in columns:
        for lag in lags:
            if group_by is not None:
                lag_expr = (
                    pl.col(col).shift(lag).over(group_by).alias(f"{col}_lag_{lag}")
                )
            else:
                lag_expr = pl.col(col).shift(lag).alias(f"{col}_lag_{lag}")
            lag_exprs.append(lag_expr)

    if lag_exprs:
        result = result.with_columns(lag_exprs)

    return result


def create_rolling_features(
    df: pl.DataFrame,
    columns: list[str],
    windows: list[int],
    *,
    functions: list[Literal["mean", "std", "min", "max", "sum", "median"]]
    | None = None,
    group_by: str | list[str] | None = None,
    sort_by: str | None = None,
    min_periods: int = 1,
) -> pl.DataFrame:
    """
    创建滚动窗口特征

    Parameters
    ----------
    df : pl.DataFrame
        输入数据框
    columns : list[str]
        要创建滚动特征的列
    windows : list[int]
        窗口大小列表，如 [7, 30] 表示 7 天和 30 天滚动
    functions : list[str], optional
        聚合函数，默认 ["mean", "std"]
    group_by : str | list[str], optional
        分组列
    sort_by : str, optional
        排序列
    min_periods : int, default 1
        最小观测数

    Returns
    -------
    pl.DataFrame
        添加了滚动特征的数据框

    Examples
    --------
    >>> df = create_rolling_features(
    ...     df, ["sales"], windows=[7, 30],
    ...     functions=["mean", "std"], sort_by="date"
    ... )
    """
    if functions is None:
        functions = ["mean", "std"]

    result = df.clone()

    if sort_by is not None:
        result = result.sort(sort_by)

    rolling_exprs = []

    for col in columns:
        for window in windows:
            for func in functions:
                col_expr = pl.col(col)

                if group_by is not None:
                    col_expr = col_expr.over(group_by)

                rolling_col = col_expr.rolling_mean(
                    window_size=window, min_periods=min_periods
                )

                if func == "mean":
                    rolling_col = col_expr.rolling_mean(
                        window_size=window, min_periods=min_periods
                    )
                elif func == "std":
                    rolling_col = col_expr.rolling_std(
                        window_size=window, min_periods=min_periods
                    )
                elif func == "min":
                    rolling_col = col_expr.rolling_min(
                        window_size=window, min_periods=min_periods
                    )
                elif func == "max":
                    rolling_col = col_expr.rolling_max(
                        window_size=window, min_periods=min_periods
                    )
                elif func == "sum":
                    rolling_col = col_expr.rolling_sum(
                        window_size=window, min_periods=min_periods
                    )
                elif func == "median":
                    rolling_col = col_expr.rolling_median(
                        window_size=window, min_periods=min_periods
                    )

                rolling_exprs.append(
                    rolling_col.alias(f"{col}_rolling_{func}_{window}")
                )

    if rolling_exprs:
        result = result.with_columns(rolling_exprs)

    return result


def encode_categorical(
    df: pl.DataFrame,
    columns: list[str],
    *,
    method: Literal["onehot", "label", "target", "frequency"] = "onehot",
    target_column: str | None = None,
    drop_original: bool = True,
) -> tuple[pl.DataFrame, dict]:
    """
    编码分类变量

    Parameters
    ----------
    df : pl.DataFrame
        输入数据框
    columns : list[str]
        要编码的分类列
    method : str, default "onehot"
        编码方法:
        - "onehot": 独热编码
        - "label": 标签编码 (0, 1, 2, ...)
        - "target": 目标编码 (用目标变量均值替换)
        - "frequency": 频率编码 (用出现频率替换)
    target_column : str, optional
        目标编码时的目标列
    drop_original : bool, default True
        是否删除原始列

    Returns
    -------
    tuple[pl.DataFrame, dict]
        (编码后的数据框, 编码映射字典)

    Examples
    --------
    >>> df, mappings = encode_categorical(df, ["category"], method="onehot")
    >>> df, mappings = encode_categorical(df, ["category"], method="target", target_column="sales")
    """
    result = df.clone()
    mappings = {}

    for col in columns:
        if col not in result.columns:
            continue

        if method == "onehot":
            # 获取唯一值
            unique_vals = result[col].unique().drop_nulls().to_list()
            mappings[col] = unique_vals

            # 创建独热编码列
            for val in unique_vals:
                result = result.with_columns(
                    (pl.col(col) == val).cast(pl.Int8).alias(f"{col}_{val}")
                )

            if drop_original:
                result = result.drop(col)

        elif method == "label":
            unique_vals = result[col].unique().drop_nulls().sort().to_list()
            label_map = {val: i for i, val in enumerate(unique_vals)}
            mappings[col] = label_map

            result = result.with_columns(
                pl.col(col)
                .replace(label_map)
                .alias(f"{col}_encoded" if not drop_original else col)
            )

            if drop_original and f"{col}_encoded" in result.columns:
                result = result.drop(col)

        elif method == "frequency":
            freq_map = (
                result.group_by(col)
                .agg(pl.len().alias("_freq"))
                .with_columns((pl.col("_freq") / len(result)).alias("_freq"))
            )
            freq_dict = dict(zip(freq_map[col].to_list(), freq_map["_freq"].to_list()))
            mappings[col] = freq_dict

            result = result.with_columns(
                pl.col(col)
                .replace(freq_dict)
                .alias(f"{col}_freq" if not drop_original else col)
            )

            if drop_original and f"{col}_freq" in result.columns:
                result = result.drop(col)

        elif method == "target":
            if target_column is None:
                raise ValueError("target_column is required for target encoding")

            target_map = result.group_by(col).agg(
                pl.col(target_column).mean().alias("_target_mean")
            )
            target_dict = dict(
                zip(target_map[col].to_list(), target_map["_target_mean"].to_list())
            )
            mappings[col] = target_dict

            result = result.with_columns(
                pl.col(col)
                .replace(target_dict)
                .alias(f"{col}_target" if not drop_original else col)
            )

            if drop_original and f"{col}_target" in result.columns:
                result = result.drop(col)

    return result, mappings


def create_interaction_features(
    df: pl.DataFrame,
    column_pairs: list[tuple[str, str]],
    *,
    operations: list[Literal["multiply", "divide", "add", "subtract"]] | None = None,
) -> pl.DataFrame:
    """
    创建交互特征

    Parameters
    ----------
    df : pl.DataFrame
        输入数据框
    column_pairs : list[tuple[str, str]]
        要创建交互的列对
    operations : list[str], optional
        运算类型，默认 ["multiply"]

    Returns
    -------
    pl.DataFrame
        添加了交互特征的数据框

    Examples
    --------
    >>> df = create_interaction_features(
    ...     df,
    ...     [("price", "quantity"), ("width", "height")],
    ...     operations=["multiply", "divide"]
    ... )
    """
    if operations is None:
        operations = ["multiply"]

    result = df.clone()

    interaction_exprs = []

    for col1, col2 in column_pairs:
        if col1 not in result.columns or col2 not in result.columns:
            continue

        if "multiply" in operations:
            interaction_exprs.append(
                (pl.col(col1) * pl.col(col2)).alias(f"{col1}_x_{col2}")
            )

        if "divide" in operations:
            interaction_exprs.append(
                (pl.col(col1) / pl.col(col2)).alias(f"{col1}_div_{col2}")
            )

        if "add" in operations:
            interaction_exprs.append(
                (pl.col(col1) + pl.col(col2)).alias(f"{col1}_plus_{col2}")
            )

        if "subtract" in operations:
            interaction_exprs.append(
                (pl.col(col1) - pl.col(col2)).alias(f"{col1}_minus_{col2}")
            )

    if interaction_exprs:
        result = result.with_columns(interaction_exprs)

    return result
