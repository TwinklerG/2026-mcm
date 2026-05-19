"""
Typst 导出模块
==============

提供 DataFrame 到 Typst 格式的转换功能，支持表格、图片引用等。

Typst 是一种现代排版系统，语法简洁，适合学术论文写作。
参考模板：https://github.com/DawnEver/mcm-icm-typst-template
"""

from pathlib import Path

import polars as pl


def to_typst_table(
    df: pl.DataFrame,
    *,
    caption: str = "",
    label: str = "",
    columns: list[str] | None = None,
    precision: int = 3,
    column_widths: list[str] | None = None,
    alignment: str | list[str] = "center",
    highlight_header: bool = True,
    stroke: str = "0.5pt",
) -> str:
    """
    将 DataFrame 转换为 Typst 表格代码

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    caption : str
        表格标题
    label : str
        Typst 标签（用于交叉引用，如 `@tab1`）
    columns : list[str], optional
        要包含的列，默认全部
    precision : int, default 3
        浮点数精度
    column_widths : list[str], optional
        列宽设置，如 `["auto", "1fr", "2fr"]`
    alignment : str | list[str], default "center"
        对齐方式："left"、"center"、"right" 或列表
    highlight_header : bool, default True
        是否加粗表头
    stroke : str, default "0.5pt"
        边框粗细

    Returns
    -------
    str
        Typst 表格代码

    Examples
    --------
    >>> typst_code = to_typst_table(
    ...     df,
    ...     caption="模型评估结果",
    ...     label="tab:results",
    ... )
    >>> print(typst_code)
    #figure(
      table(
        columns: (auto, auto, auto),
        ...
      ),
      caption: [模型评估结果],
    )<tab:results>
    """
    if columns is not None:
        df = df.select(columns)

    cols = df.columns
    n_cols = len(cols)

    # 列宽设置
    if column_widths is None:
        widths = ", ".join(["auto"] * n_cols)
    else:
        widths = ", ".join(column_widths)

    # 对齐设置
    if isinstance(alignment, str):
        align_map = {"left": "left", "center": "center", "right": "right"}
        align = align_map.get(alignment, "center")
    else:
        align = "horizon"  # 混合对齐使用默认值

    # 构建表格内容
    lines = []
    lines.append("#figure(")
    lines.append("  table(")
    lines.append(f"    columns: ({widths}),")
    lines.append("    inset: 6pt,")
    lines.append(f"    stroke: {stroke},")
    lines.append(f"    align: {align},")

    # 表头
    if highlight_header:
        header_cells = ", ".join([f"[*{col}*]" for col in cols])
    else:
        header_cells = ", ".join([f"[{col}]" for col in cols])
    lines.append(f"    {header_cells},")

    # 数据行
    for row in df.iter_rows():
        formatted_cells = []
        for val in row:
            if val is None:
                formatted_cells.append("[-]")
            elif isinstance(val, float):
                formatted_cells.append(f"[{val:.{precision}f}]")
            elif isinstance(val, int):
                formatted_cells.append(f"[{val}]")
            else:
                # 转义特殊字符
                str_val = str(val).replace("[", r"\[").replace("]", r"\]")
                formatted_cells.append(f"[{str_val}]")
        lines.append("    " + ", ".join(formatted_cells) + ",")

    lines.append("  ),")

    # 标题
    if caption:
        lines.append(f"  caption: [{caption}],")

    lines.append(")")

    # 标签
    if label:
        # 移除可能的前缀，确保标签格式正确
        clean_label = label.replace("tab:", "").replace("tbl:", "")
        lines[-1] = lines[-1] + f"<{clean_label}>"

    return "\n".join(lines)


def to_typst_figure(
    image_path: str,
    *,
    caption: str = "",
    label: str = "",
    width: str = "80%",
) -> str:
    """
    生成 Typst 图片引用代码

    Parameters
    ----------
    image_path : str
        图片相对路径（相对于 Typst 文档）
    caption : str
        图片标题
    label : str
        Typst 标签（用于交叉引用，如 `@fig1`）
    width : str, default "80%"
        图片宽度

    Returns
    -------
    str
        Typst 图片代码

    Examples
    --------
    >>> code = to_typst_figure(
    ...     "./figures/scatter.png",
    ...     caption="散点图分析",
    ...     label="fig:scatter",
    ... )
    >>> print(code)
    #figure(
      image("./figures/scatter.png", width: 80%),
      caption: [散点图分析],
    )<scatter>
    """
    lines = []
    lines.append("#figure(")
    lines.append(f'  image("{image_path}", width: {width}),')

    if caption:
        lines.append(f"  caption: [{caption}],")

    lines.append(")")

    if label:
        clean_label = label.replace("fig:", "").replace("figure:", "")
        lines[-1] = lines[-1] + f"<{clean_label}>"

    return "\n".join(lines)


def to_typst_equation(
    equation: str,
    *,
    label: str = "",
    block: bool = True,
) -> str:
    """
    生成 Typst 公式代码

    Parameters
    ----------
    equation : str
        公式内容（Typst 数学语法）
    label : str
        公式标签（用于交叉引用）
    block : bool, default True
        是否为块级公式

    Returns
    -------
    str
        Typst 公式代码

    Examples
    --------
    >>> code = to_typst_equation("y = a x + b", label="eq:linear")
    >>> print(code)
    $ y = a x + b $<linear>
    """
    if block:
        result = f"$\n{equation}\n$"
    else:
        result = f"${equation}$"

    if label:
        clean_label = label.replace("eq:", "").replace("eqn:", "")
        result = result + f"<{clean_label}>"

    return result


def export_typst_assets(
    tables: dict[str, tuple[pl.DataFrame, str]] | None = None,
    figures: dict[str, tuple[str, str]] | None = None,
    output_file: str | Path = "typst_assets.typ",
) -> Path:
    """
    批量导出 Typst 资源文件

    将多个表格和图片引用导出到单个 Typst 文件中，方便在论文中引用。

    Parameters
    ----------
    tables : dict[str, tuple[pl.DataFrame, str]], optional
        表格字典，格式：`{标签: (数据框, 标题)}`
    figures : dict[str, tuple[str, str]], optional
        图片字典，格式：`{标签: (图片路径, 标题)}`
    output_file : str | Path, default "typst_assets.typ"
        输出文件路径

    Returns
    -------
    Path
        输出文件路径

    Examples
    --------
    >>> export_typst_assets(
    ...     tables={
    ...         "results": (results_df, "模型结果"),
    ...         "comparison": (comp_df, "方法对比"),
    ...     },
    ...     figures={
    ...         "scatter": ("./figures/scatter.png", "散点分析"),
    ...     },
    ...     output_file="assets.typ",
    ... )
    """
    output_path = Path(output_file)
    content_parts = []

    # 文件头注释
    content_parts.append("// Typst 资源文件 - 由 MCM 模板库自动生成")
    content_parts.append("// 使用方法：在主文档中使用 #include 引入")
    content_parts.append("")

    # 导出表格
    if tables:
        content_parts.append("// ===== 表格 =====")
        content_parts.append("")
        for label, (df, caption) in tables.items():
            typst_table = to_typst_table(df, caption=caption, label=label)
            content_parts.append(typst_table)
            content_parts.append("")

    # 导出图片引用
    if figures:
        content_parts.append("// ===== 图片 =====")
        content_parts.append("")
        for label, (path, caption) in figures.items():
            typst_fig = to_typst_figure(path, caption=caption, label=label)
            content_parts.append(typst_fig)
            content_parts.append("")

    # 写入文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content_parts))

    return output_path


def dataframe_to_typst_raw(
    df: pl.DataFrame,
    *,
    precision: int = 3,
) -> str:
    """
    将 DataFrame 转换为原始 Typst 表格数据（不含 figure 包装）

    适用于需要自定义表格样式的场景。

    Parameters
    ----------
    df : pl.DataFrame
        数据框
    precision : int, default 3
        浮点数精度

    Returns
    -------
    str
        Typst 表格数据
    """
    cols = df.columns
    n_cols = len(cols)

    lines = []
    lines.append("table(")
    lines.append(f"  columns: {n_cols},")

    # 表头
    header_cells = ", ".join([f"[*{col}*]" for col in cols])
    lines.append(f"  {header_cells},")

    # 数据行
    for row in df.iter_rows():
        formatted_cells = []
        for val in row:
            if val is None:
                formatted_cells.append("[-]")
            elif isinstance(val, float):
                formatted_cells.append(f"[{val:.{precision}f}]")
            else:
                formatted_cells.append(f"[{val}]")
        lines.append("  " + ", ".join(formatted_cells) + ",")

    lines.append(")")

    return "\n".join(lines)
