"""
网络数据输入输出模块
====================

提供网络数据的加载、保存和格式转换功能
"""

from pathlib import Path
from typing import Any

import networkx as nx
import polars as pl


def load_network(
    filepath: str | Path,
    *,
    source: str = "source",
    target: str = "target",
    weight: str | None = None,
    directed: bool = False,
    delimiter: str = ",",
) -> nx.Graph | nx.DiGraph:
    """
    从文件加载网络

    支持 CSV、TSV、边列表等格式

    Parameters
    ----------
    filepath : str | Path
        文件路径
    source : str, default "source"
        源节点列名
    target : str, default "target"
        目标节点列名
    weight : str, optional
        权重列名
    directed : bool, default False
        是否为有向图
    delimiter : str, default ","
        分隔符

    Returns
    -------
    nx.Graph | nx.DiGraph
        加载的网络图

    Examples
    --------
    >>> G = load_network("edges.csv", source="from", target="to", weight="weight")
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in (".csv", ".tsv", ".txt"):
        # 使用 Polars 读取
        if suffix == ".tsv":
            delimiter = "\t"

        df = pl.read_csv(filepath, separator=delimiter)

        # 创建图
        G = nx.DiGraph() if directed else nx.Graph()

        # 添加边
        if weight and weight in df.columns:
            edges = [
                (row[source], row[target], {weight: row[weight]})
                for row in df.iter_rows(named=True)
            ]
        else:
            edges = [(row[source], row[target]) for row in df.iter_rows(named=True)]

        G.add_edges_from(edges)

    elif suffix == ".gml":
        G = nx.read_gml(filepath)
        if directed and not G.is_directed():
            G = G.to_directed()

    elif suffix == ".graphml":
        G = nx.read_graphml(filepath)
        if directed and not G.is_directed():
            G = G.to_directed()

    elif suffix in (".gexf",):
        G = nx.read_gexf(filepath)
        if directed and not G.is_directed():
            G = G.to_directed()

    elif suffix == ".edgelist":
        if directed:
            G = nx.read_edgelist(
                filepath, create_using=nx.DiGraph(), delimiter=delimiter
            )
        else:
            G = nx.read_edgelist(filepath, delimiter=delimiter)

    elif suffix == ".adjlist":
        G = nx.read_adjlist(filepath)
        if directed:
            G = G.to_directed()

    else:
        raise ValueError(f"不支持的文件格式: {suffix}")

    return G


def save_network(
    G: nx.Graph | nx.DiGraph,
    filepath: str | Path,
    *,
    source: str = "source",
    target: str = "target",
    weight: str | None = "weight",
) -> None:
    """
    保存网络到文件

    Parameters
    ----------
    G : nx.Graph | nx.DiGraph
        网络图
    filepath : str | Path
        文件路径
    source : str, default "source"
        源节点列名（CSV 格式）
    target : str, default "target"
        目标节点列名（CSV 格式）
    weight : str, optional
        权重属性名

    Examples
    --------
    >>> save_network(G, "output/network.csv")
    >>> save_network(G, "output/network.gml")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        df = network_to_polars(G, source=source, target=target, weight=weight)
        df.write_csv(filepath)

    elif suffix == ".gml":
        nx.write_gml(G, filepath)

    elif suffix == ".graphml":
        nx.write_graphml(G, filepath)

    elif suffix == ".gexf":
        nx.write_gexf(G, filepath)

    elif suffix == ".edgelist":
        nx.write_edgelist(G, filepath)

    elif suffix == ".adjlist":
        nx.write_adjlist(G, filepath)

    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


def network_to_polars(
    G: nx.Graph | nx.DiGraph,
    *,
    source: str = "source",
    target: str = "target",
    weight: str | None = "weight",
    include_attributes: bool = True,
) -> pl.DataFrame:
    """
    将网络转换为 Polars DataFrame（边列表格式）

    Parameters
    ----------
    G : nx.Graph | nx.DiGraph
        网络图
    source : str, default "source"
        源节点列名
    target : str, default "target"
        目标节点列名
    weight : str, optional
        权重属性名
    include_attributes : bool, default True
        是否包含边属性

    Returns
    -------
    pl.DataFrame
        边列表 DataFrame

    Examples
    --------
    >>> df = network_to_polars(G)
    >>> print(df.head())
    """
    if include_attributes:
        edges = []
        for u, v, data in G.edges(data=True):
            edge = {source: u, target: v}
            edge.update(data)
            edges.append(edge)

        if edges:
            df = pl.DataFrame(edges)
        else:
            df = pl.DataFrame({source: [], target: []})
    else:
        edges = [(u, v) for u, v in G.edges()]
        df = pl.DataFrame(
            {
                source: [e[0] for e in edges],
                target: [e[1] for e in edges],
            }
        )

    return df


def polars_to_network(
    df: pl.DataFrame,
    *,
    source: str = "source",
    target: str = "target",
    weight: str | None = None,
    directed: bool = False,
    node_attributes: pl.DataFrame | None = None,
    node_id: str = "node",
) -> nx.Graph | nx.DiGraph:
    """
    将 Polars DataFrame 转换为网络

    Parameters
    ----------
    df : pl.DataFrame
        边列表 DataFrame
    source : str, default "source"
        源节点列名
    target : str, default "target"
        目标节点列名
    weight : str, optional
        权重列名
    directed : bool, default False
        是否为有向图
    node_attributes : pl.DataFrame, optional
        节点属性 DataFrame
    node_id : str, default "node"
        节点 ID 列名（在 node_attributes 中）

    Returns
    -------
    nx.Graph | nx.DiGraph
        转换后的网络图

    Examples
    --------
    >>> edges = pl.DataFrame({
    ...     "source": [1, 1, 2],
    ...     "target": [2, 3, 3],
    ...     "weight": [1.0, 2.0, 1.5]
    ... })
    >>> G = polars_to_network(edges, weight="weight")
    """
    G = nx.DiGraph() if directed else nx.Graph()

    # 添加边
    attr_cols = [c for c in df.columns if c not in (source, target)]

    for row in df.iter_rows(named=True):
        u, v = row[source], row[target]
        attrs = {k: row[k] for k in attr_cols if row[k] is not None}
        G.add_edge(u, v, **attrs)

    # 添加节点属性
    if node_attributes is not None:
        for row in node_attributes.iter_rows(named=True):
            node = row[node_id]
            if node in G:
                attrs = {k: v for k, v in row.items() if k != node_id and v is not None}
                G.nodes[node].update(attrs)

    return G


def adjacency_to_network(
    adj_matrix: Any,
    *,
    node_labels: list | None = None,
    directed: bool = False,
    weighted: bool = True,
) -> nx.Graph | nx.DiGraph:
    """
    将邻接矩阵转换为网络

    Parameters
    ----------
    adj_matrix : array-like
        邻接矩阵（numpy 数组或列表）
    node_labels : list, optional
        节点标签列表
    directed : bool, default False
        是否为有向图
    weighted : bool, default True
        是否为加权图

    Returns
    -------
    nx.Graph | nx.DiGraph
        转换后的网络图

    Examples
    --------
    >>> import numpy as np
    >>> adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> G = adjacency_to_network(adj, node_labels=['A', 'B', 'C'])
    """
    import numpy as np

    adj_matrix = np.array(adj_matrix)

    if directed:
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
    else:
        G = nx.from_numpy_array(adj_matrix)

    if node_labels:
        mapping = {i: label for i, label in enumerate(node_labels)}
        G = nx.relabel_nodes(G, mapping)

    if not weighted:
        for u, v in G.edges():
            G[u][v]["weight"] = 1

    return G


def network_to_adjacency(
    G: nx.Graph | nx.DiGraph,
    *,
    weight: str | None = "weight",
    nodelist: list | None = None,
) -> tuple:
    """
    将网络转换为邻接矩阵

    Parameters
    ----------
    G : nx.Graph | nx.DiGraph
        网络图
    weight : str, optional
        权重属性名
    nodelist : list, optional
        节点顺序列表

    Returns
    -------
    tuple
        (邻接矩阵, 节点列表)

    Examples
    --------
    >>> adj, nodes = network_to_adjacency(G)
    """

    if nodelist is None:
        nodelist = list(G.nodes())

    adj = nx.to_numpy_array(G, nodelist=nodelist, weight=weight)
    return adj, nodelist
