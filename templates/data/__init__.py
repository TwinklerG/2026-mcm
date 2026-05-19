"""
数据处理模块
==========================

提供高性能的数据加载、预处理和特征工程功能。

主要功能
--------
- 数据加载: load_csv, load_excel, load_parquet
- 数据预处理: handle_missing, handle_outliers, normalize
- 特征工程: create_features, encode_categorical, create_time_features
- 数据探索: describe, correlation_matrix, value_counts
"""

from .explore import (
    correlation_matrix,
    describe_df,
    detect_outliers,
    missing_report,
)
from .features import (
    create_interaction_features,
    create_lag_features,
    create_rolling_features,
    create_time_features,
    encode_categorical,
)
from .loader import load_csv, load_data, load_excel, load_parquet
from .preprocess import (
    handle_missing,
    handle_outliers,
    normalize,
    preprocess_pipeline,
    standardize,
)

__all__ = [
    # 加载
    "load_csv",
    "load_excel",
    "load_parquet",
    "load_data",
    # 预处理
    "handle_missing",
    "handle_outliers",
    "normalize",
    "standardize",
    "preprocess_pipeline",
    # 特征工程
    "create_time_features",
    "create_lag_features",
    "create_rolling_features",
    "encode_categorical",
    "create_interaction_features",
    # 探索
    "describe_df",
    "correlation_matrix",
    "missing_report",
    "detect_outliers",
]
