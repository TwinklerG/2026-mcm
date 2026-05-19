"""
假设检验模块
============
"""

from typing import Literal

import numpy as np
import polars as pl
from scipy import stats


def t_test(
    df: pl.DataFrame,
    column: str,
    group_by: str,
    *,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    equal_var: bool = True,
) -> dict:
    """
    独立样本 t 检验

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    column : str
        要比较的数值列
    group_by : str
        分组列（必须恰好有两组）
    alternative : str, default "two-sided"
        备择假设方向
    equal_var : bool, default True
        是否假设方差相等（False 使用 Welch's t-test）

    Returns
    -------
    dict
        检验结果

    Examples
    --------
    >>> # 比较东道主和非东道主的奖牌数差异
    >>> result = t_test(df, "medals", group_by="is_host")
    >>> print(f"t = {result['statistic']:.3f}, p = {result['p_value']:.4f}")
    """
    groups = df[group_by].unique().to_list()
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}")

    group1_data = (
        df.filter(pl.col(group_by) == groups[0])[column].drop_nulls().to_numpy()
    )
    group2_data = (
        df.filter(pl.col(group_by) == groups[1])[column].drop_nulls().to_numpy()
    )

    t_stat, p_value = stats.ttest_ind(
        group1_data, group2_data, equal_var=equal_var, alternative=alternative
    )

    # 效应量（Cohen's d）
    n1, n2 = len(group1_data), len(group2_data)
    pooled_std = np.sqrt(
        ((n1 - 1) * group1_data.std() ** 2 + (n2 - 1) * group2_data.std() ** 2)
        / (n1 + n2 - 2)
    )
    cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std

    return {
        "test": "Independent samples t-test" if equal_var else "Welch's t-test",
        "group1": groups[0],
        "group2": groups[1],
        "n1": n1,
        "n2": n2,
        "mean1": float(group1_data.mean()),
        "mean2": float(group2_data.mean()),
        "std1": float(group1_data.std()),
        "std2": float(group2_data.std()),
        "statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant": p_value < 0.05,
        "effect_size": "large"
        if abs(cohens_d) >= 0.8
        else "medium"
        if abs(cohens_d) >= 0.5
        else "small",
    }


def paired_t_test(
    df: pl.DataFrame,
    column1: str,
    column2: str,
    *,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> dict:
    """
    配对样本 t 检验

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    column1, column2 : str
        要比较的两列
    alternative : str, default "two-sided"
        备择假设方向

    Returns
    -------
    dict
        检验结果

    Examples
    --------
    >>> # 比较预测值和实际值
    >>> result = paired_t_test(df, "predicted", "actual")
    """
    data = df.select([column1, column2]).drop_nulls()
    vals1 = data[column1].to_numpy()
    vals2 = data[column2].to_numpy()

    t_stat, p_value = stats.ttest_rel(vals1, vals2, alternative=alternative)

    diff = vals1 - vals2
    cohens_d = diff.mean() / diff.std()

    return {
        "test": "Paired samples t-test",
        "column1": column1,
        "column2": column2,
        "n": len(vals1),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std()),
        "statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant": p_value < 0.05,
    }


def anova(
    df: pl.DataFrame,
    column: str,
    group_by: str,
) -> dict:
    """
    单因素方差分析（ANOVA）

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    column : str
        要比较的数值列
    group_by : str
        分组列

    Returns
    -------
    dict
        检验结果

    Examples
    --------
    >>> # 比较不同地区的奖牌数差异
    >>> result = anova(df, "medals", group_by="region")
    """
    groups = df[group_by].unique().to_list()
    group_data = [
        df.filter(pl.col(group_by) == g)[column].drop_nulls().to_numpy() for g in groups
    ]

    f_stat, p_value = stats.f_oneway(*group_data)

    # 计算组间和组内变异
    all_data = np.concatenate(group_data)
    grand_mean = all_data.mean()

    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in group_data)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in group_data)
    ss_total = ss_between + ss_within

    # 效应量（eta squared）
    eta_squared = ss_between / ss_total

    return {
        "test": "One-way ANOVA",
        "variable": column,
        "group_by": group_by,
        "n_groups": len(groups),
        "groups": groups,
        "group_means": {g: float(d.mean()) for g, d in zip(groups, group_data)},
        "group_stds": {g: float(d.std()) for g, d in zip(groups, group_data)},
        "group_sizes": {g: len(d) for g, d in zip(groups, group_data)},
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "eta_squared": float(eta_squared),
        "significant": p_value < 0.05,
        "effect_size": "large"
        if eta_squared >= 0.14
        else "medium"
        if eta_squared >= 0.06
        else "small",
    }


def chi_square_test(
    df: pl.DataFrame,
    column1: str,
    column2: str,
) -> dict:
    """
    卡方独立性检验

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    column1, column2 : str
        两个分类变量列

    Returns
    -------
    dict
        检验结果

    Examples
    --------
    >>> # 检验性别和获奖的独立性
    >>> result = chi_square_test(df, "gender", "won_medal")
    """
    # 使用纯 Polars 构建列联表
    cross = (
        df.group_by([column1, column2])
        .agg(pl.len().alias("count"))
        .sort([column1, column2])
    )

    # 获取唯一值
    vals1 = sorted(df[column1].unique().to_list())
    vals2 = sorted(df[column2].unique().to_list())

    # 构建 numpy 数组形式的列联表
    contingency = np.zeros((len(vals1), len(vals2)), dtype=int)
    for row in cross.iter_rows(named=True):
        i = vals1.index(row[column1])
        j = vals2.index(row[column2])
        contingency[i, j] = row["count"]

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Cramér's V
    n = contingency.sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    # 转换为字典格式（类似于 pandas crosstab 输出）
    contingency_dict = {
        str(vals2[j]): {
            str(vals1[i]): int(contingency[i, j]) for i in range(len(vals1))
        }
        for j in range(len(vals2))
    }

    return {
        "test": "Chi-square test of independence",
        "variable1": column1,
        "variable2": column2,
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": float(cramers_v),
        "significant": p_value < 0.05,
        "contingency_table": contingency_dict,
    }


def normality_test(
    df: pl.DataFrame,
    column: str,
    *,
    method: Literal["shapiro", "ks", "dagostino"] = "shapiro",
) -> dict:
    """
    正态性检验

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    column : str
        要检验的列
    method : str, default "shapiro"
        检验方法："shapiro"（Shapiro-Wilk）, "ks"（Kolmogorov-Smirnov）, "dagostino"

    Returns
    -------
    dict
        检验结果

    Examples
    --------
    >>> result = normality_test(df, "residuals")
    >>> if result["is_normal"]:
    ...     print("Data appears normally distributed")
    """
    data = df[column].drop_nulls().to_numpy()

    if method == "shapiro":
        # Shapiro-Wilk（适用于 n < 5000）
        stat, p_value = stats.shapiro(data[:5000] if len(data) > 5000 else data)
        test_name = "Shapiro-Wilk"
    elif method == "ks":
        # Kolmogorov-Smirnov
        stat, p_value = stats.kstest(data, "norm", args=(data.mean(), data.std()))
        test_name = "Kolmogorov-Smirnov"
    elif method == "dagostino":
        # D'Agostino-Pearson
        stat, p_value = stats.normaltest(data)
        test_name = "D'Agostino-Pearson"

    return {
        "test": test_name,
        "column": column,
        "n": len(data),
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_normal": p_value >= 0.05,
        "skewness": float(stats.skew(data)),
        "kurtosis": float(stats.kurtosis(data)),
    }
