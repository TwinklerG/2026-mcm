"""
格式化模块
==========
"""

from typing import Literal

import polars as pl


def format_number(
    value: float,
    *,
    precision: int = 3,
    style: Literal["decimal", "scientific", "percent", "compact"] = "decimal",
) -> str:
    """
    格式化数值

    Parameters
    ----------
    value : float
        数值
    precision : int, default 3
        精度
    style : str, default "decimal"
        格式风格

    Returns
    -------
    str
        格式化后的字符串

    Examples
    --------
    >>> format_number(1234567.89, style="compact")
    '1.23M'
    >>> format_number(0.12345, style="percent")
    '12.35%'
    """
    if value is None:
        return "-"

    if style == "decimal":
        return f"{value:.{precision}f}"

    elif style == "scientific":
        return f"{value:.{precision}e}"

    elif style == "percent":
        return f"{value * 100:.{precision - 2 if precision > 2 else 0}f}%"

    elif style == "compact":
        abs_val = abs(value)
        if abs_val >= 1e9:
            return f"{value / 1e9:.{precision - 1}f}B"
        elif abs_val >= 1e6:
            return f"{value / 1e6:.{precision - 1}f}M"
        elif abs_val >= 1e3:
            return f"{value / 1e3:.{precision - 1}f}K"
        else:
            return f"{value:.{precision}f}"

    return str(value)


def format_pvalue(
    p: float,
    *,
    threshold: float = 0.001,
    show_stars: bool = True,
) -> str:
    """
    格式化 p 值

    Parameters
    ----------
    p : float
        p 值
    threshold : float, default 0.001
        显示为 < 的阈值
    show_stars : bool, default True
        是否显示显著性星号

    Returns
    -------
    str
        格式化后的 p 值

    Examples
    --------
    >>> format_pvalue(0.0001)
    '< 0.001***'
    >>> format_pvalue(0.03)
    '0.030*'
    """
    if p is None:
        return "-"

    if p < threshold:
        result = f"< {threshold}"
    else:
        result = f"{p:.3f}"

    if show_stars:
        if p < 0.001:
            result += "***"
        elif p < 0.01:
            result += "**"
        elif p < 0.05:
            result += "*"

    return result


def format_percentage(
    value: float,
    *,
    precision: int = 1,
    show_sign: bool = False,
) -> str:
    """
    格式化百分比

    Parameters
    ----------
    value : float
        比例值 (0.1 = 10%)
    precision : int, default 1
        小数精度
    show_sign : bool, default False
        是否显示正号

    Returns
    -------
    str
        格式化后的百分比
    """
    if value is None:
        return "-"

    pct = value * 100

    if show_sign and pct > 0:
        return f"+{pct:.{precision}f}%"
    else:
        return f"{pct:.{precision}f}%"


def create_summary_table(
    data: dict[str, dict],
    *,
    row_name: str = "Metric",
    precision: int = 4,
) -> pl.DataFrame:
    """
    创建汇总表

    Parameters
    ----------
    data : dict
        数据字典，格式为 {模型名: {指标名: 值}}
    row_name : str, default "Metric"
        行名列的名称
    precision : int, default 4
        数值精度

    Returns
    -------
    pl.DataFrame
        汇总表

    Examples
    --------
    >>> data = {
    ...     "Model A": {"R²": 0.95, "RMSE": 1.23},
    ...     "Model B": {"R²": 0.92, "RMSE": 1.45},
    ... }
    >>> table = create_summary_table(data)
    """
    if not data:
        return pl.DataFrame()

    # 获取所有指标
    all_metrics = set()
    for model_data in data.values():
        all_metrics.update(model_data.keys())

    metrics = sorted(all_metrics)

    # 构建表格
    table_data = {row_name: metrics}

    for model_name, model_data in data.items():
        table_data[model_name] = [
            format_number(model_data.get(metric), precision=precision)
            for metric in metrics
        ]

    return pl.DataFrame(table_data)


def highlight_best(
    df: pl.DataFrame,
    columns: list[str],
    *,
    higher_better: list[str] | None = None,
    lower_better: list[str] | None = None,
    bold_format: str = "**{}**",
) -> pl.DataFrame:
    """
    高亮最佳值

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    columns : list[str]
        数值列
    higher_better : list[str], optional
        越高越好的列
    lower_better : list[str], optional
        越低越好的列
    bold_format : str, default "**{}**"
        加粗格式

    Returns
    -------
    pl.DataFrame
        高亮后的数据框
    """
    result = df.clone()

    if higher_better is None:
        higher_better = []
    if lower_better is None:
        lower_better = []

    for col in columns:
        if col not in result.columns:
            continue

        values = result[col]

        if col in higher_better:
            best_idx = values.arg_max()
        elif col in lower_better:
            best_idx = values.arg_min()
        else:
            continue

        # 创建高亮版本
        new_values = []
        for i, val in enumerate(values):
            if i == best_idx:
                new_values.append(bold_format.format(val))
            else:
                new_values.append(str(val))

        result = result.with_columns(pl.Series(col, new_values))

    return result
