"""
导出模块
========

提供数据和模型导出功能。
"""

import json
from pathlib import Path
from typing import Any

import polars as pl


def to_latex_table(
    df: pl.DataFrame,
    *,
    caption: str = "",
    label: str = "",
    columns: list[str] | None = None,
    column_format: str | None = None,
    precision: int = 3,
    escape: bool = True,
) -> str:
    """
    将 DataFrame 转换为 LaTeX 表格

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    caption : str
        表格标题
    label : str
        LaTeX 标签（用于交叉引用）
    columns : list[str], optional
        要包含的列
    column_format : str, optional
        列格式，如 "lccc"
    precision : int, default 3
        小数精度
    escape : bool, default True
        是否转义特殊字符

    Returns
    -------
    str
        LaTeX 表格代码

    Examples
    --------
    >>> latex = to_latex_table(df, caption="Model Results", label="tab:results")
    >>> print(latex)
    """
    if columns is not None:
        df = df.select(columns)

    # 转换为 pandas 利用其 to_latex
    pdf = df.to_pandas()

    # 格式化数值
    for col in pdf.select_dtypes(include=["float"]).columns:
        pdf[col] = pdf[col].map(lambda x: f"{x:.{precision}f}" if pd.notna(x) else "")

    if column_format is None:
        column_format = "l" + "c" * (len(df.columns) - 1)

    # 生成 LaTeX
    import pandas as pd

    latex = pdf.to_latex(
        index=False,
        escape=escape,
        column_format=column_format,
        caption=caption if caption else None,
        label=label if label else None,
    )

    # 添加美化
    latex = latex.replace("\\toprule", "\\toprule\\midrule")

    return latex


def to_markdown_table(
    df: pl.DataFrame,
    *,
    columns: list[str] | None = None,
    precision: int = 3,
    alignment: str | list[str] | None = None,
) -> str:
    """
    将 DataFrame 转换为 Markdown 表格

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    columns : list[str], optional
        要包含的列
    precision : int, default 3
        小数精度
    alignment : str | list[str], optional
        对齐方式："l", "c", "r" 或列表

    Returns
    -------
    str
        Markdown 表格

    Examples
    --------
    >>> md = to_markdown_table(df)
    >>> print(md)
    """
    if columns is not None:
        df = df.select(columns)

    cols = df.columns
    n_cols = len(cols)

    if alignment is None:
        alignment = ["c"] * n_cols
    elif isinstance(alignment, str):
        alignment = [alignment] * n_cols

    # 构建表头
    header = "| " + " | ".join(cols) + " |"

    # 构建分隔行
    sep_map = {"l": ":---", "c": ":---:", "r": "---:"}
    separator = "| " + " | ".join(sep_map.get(a, "---") for a in alignment) + " |"

    # 构建数据行
    rows = []
    for row in df.iter_rows():
        formatted = []
        for val in row:
            if isinstance(val, float):
                formatted.append(f"{val:.{precision}f}")
            elif val is None:
                formatted.append("-")
            else:
                formatted.append(str(val))
        rows.append("| " + " | ".join(formatted) + " |")

    return "\n".join([header, separator] + rows)


def save_results(
    results: dict,
    filename: str,
    *,
    output_dir: str | Path = "outputs",
    format: str = "json",
) -> Path:
    """
    保存结果到文件

    Parameters
    ----------
    results : dict
        结果字典
    filename : str
        文件名 (不含扩展名)
    output_dir : str | Path, default "outputs"
        输出目录
    format : str, default "json"
        格式："json"

    Returns
    -------
    Path
        保存的文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{filename}.{format}"

    # 处理不可序列化的对象
    def serialize(obj):
        if isinstance(obj, pl.DataFrame):
            return obj.to_dicts()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "tolist"):  # numpy array
            return obj.tolist()
        elif hasattr(obj, "__dict__"):
            return str(obj)
        return obj

    def deep_serialize(d):
        if isinstance(d, dict):
            return {k: deep_serialize(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [deep_serialize(v) for v in d]
        else:
            return serialize(d)

    serialized = deep_serialize(results)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)

    return filepath


def export_model(
    model: Any,
    filename: str,
    *,
    output_dir: str | Path = "models",
) -> Path:
    """
    导出模型到文件

    Parameters
    ----------
    model : Any
        模型对象
    filename : str
        文件名 (不含扩展名)
    output_dir : str | Path, default "models"
        输出目录

    Returns
    -------
    Path
        保存的文件路径
    """
    import pickle

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f"{filename}.pkl"

    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    return filepath


def load_model(
    filename: str,
    *,
    model_dir: str | Path = "models",
) -> Any:
    """
    加载模型

    Parameters
    ----------
    filename : str
        文件名 (含或不含扩展名)
    model_dir : str | Path, default "models"
        模型目录

    Returns
    -------
    Any
        模型对象
    """
    import pickle

    model_dir = Path(model_dir)

    if not filename.endswith(".pkl"):
        filename = f"{filename}.pkl"

    filepath = model_dir / filename

    with open(filepath, "rb") as f:
        return pickle.load(f)
