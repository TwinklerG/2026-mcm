"""
数据加载模块
============

支持多种格式的高性能数据加载。
"""

from pathlib import Path
from typing import Any

import polars as pl


def load_csv(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    dtypes: dict[str, pl.DataType] | None = None,
    null_values: list[str] | None = None,
    skip_rows: int = 0,
    n_rows: int | None = None,
    encoding: str = "utf-8",
    separator: str = ",",
    infer_schema_length: int = 10000,
) -> pl.DataFrame:
    """
    高性能 CSV 数据加载

    Parameters
    ----------
    path : str | Path
        CSV 文件路径
    columns : list[str], optional
        要加载的列名列表，None 表示加载所有列
    dtypes : dict[str, pl.DataType], optional
        指定列的数据类型
    null_values : list[str], optional
        被视为空值的字符串列表，默认 ["", "NA", "N/A", "null", "NULL", "NaN"]
    skip_rows : int, default 0
        跳过的行数
    n_rows : int, optional
        读取的最大行数
    encoding : str, default "utf-8"
        文件编码
    separator : str, default ","
        分隔符
    infer_schema_length : int, default 10000
        用于推断数据类型的行数

    Returns
    -------
    pl.DataFrame
        加载的数据框

    Examples
    --------
    >>> df = load_csv("data.csv")
    >>> df = load_csv("data.csv", columns=["id", "value"], dtypes={"id": pl.Int64})
    """
    if null_values is None:
        null_values = ["", "NA", "N/A", "null", "NULL", "NaN", "nan", "-"]

    return pl.read_csv(
        path,
        columns=columns,
        dtypes=dtypes,
        null_values=null_values,
        skip_rows=skip_rows,
        n_rows=n_rows,
        encoding=encoding,
        separator=separator,
        infer_schema_length=infer_schema_length,
    )


def load_excel(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
    columns: list[str] | None = None,
    skip_rows: int = 0,
    n_rows: int | None = None,
) -> pl.DataFrame:
    """
    加载 Excel 文件

    Parameters
    ----------
    path : str | Path
        Excel 文件路径
    sheet_name : str | int, default 0
        工作表名称或索引
    columns : list[str], optional
        要加载的列名列表
    skip_rows : int, default 0
        跳过的行数
    n_rows : int, optional
        读取的最大行数

    Returns
    -------
    pl.DataFrame
        加载的数据框

    Examples
    --------
    >>> df = load_excel("data.xlsx", sheet_name="Sheet1")
    """
    return pl.read_excel(
        path,
        sheet_name=sheet_name,
        columns=columns,
        read_options={"skip_rows": skip_rows, "n_rows": n_rows},
    )


def load_parquet(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    n_rows: int | None = None,
) -> pl.DataFrame:
    """
    高性能 Parquet 数据加载

    Parquet 是列式存储格式，读写性能优异，推荐用于大数据集。

    Parameters
    ----------
    path : str | Path
        Parquet 文件路径
    columns : list[str], optional
        要加载的列名列表
    n_rows : int, optional
        读取的最大行数

    Returns
    -------
    pl.DataFrame
        加载的数据框

    Examples
    --------
    >>> df = load_parquet("data.parquet")
    """
    return pl.read_parquet(path, columns=columns, n_rows=n_rows)


def load_data(
    path: str | Path,
    **kwargs: Any,
) -> pl.DataFrame:
    """
    自动检测格式并加载数据

    根据文件扩展名自动选择合适的加载方法。

    Parameters
    ----------
    path : str | Path
        数据文件路径
    **kwargs
        传递给具体加载函数的参数

    Returns
    -------
    pl.DataFrame
        加载的数据框

    Examples
    --------
    >>> df = load_data("data.csv")
    >>> df = load_data("data.xlsx", sheet_name="Sheet1")
    >>> df = load_data("data.parquet")
    """
    path = Path(path)
    suffix = path.suffix.lower()

    loaders = {
        ".csv": load_csv,
        ".tsv": lambda p, **kw: load_csv(p, separator="\t", **kw),
        ".xlsx": load_excel,
        ".xls": load_excel,
        ".parquet": load_parquet,
        ".pq": load_parquet,
    }

    if suffix not in loaders:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: {list(loaders.keys())}"
        )

    return loaders[suffix](path, **kwargs)
