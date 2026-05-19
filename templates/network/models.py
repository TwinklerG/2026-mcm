"""
网络模型生成模块
================

提供常见网络模型的生成功能，包括：
- Erdős-Rényi (ER) 随机图
- Barabási-Albert (BA) 无标度网络
- Watts-Strogatz (WS) 小世界网络
- 随机块模型 (SBM)
"""

from typing import Sequence

import networkx as nx
import numpy as np


def generate_er_network(
    n: int,
    p: float | None = None,
    m: int | None = None,
    *,
    seed: int | None = None,
    directed: bool = False,
) -> nx.Graph | nx.DiGraph:
    """
    生成 Erdős-Rényi 随机图

    可以使用 G(n, p) 或 G(n, m) 模型

    Parameters
    ----------
    n : int
        节点数量
    p : float, optional
        边存在的概率（G(n, p) 模型）
    m : int, optional
        边的数量（G(n, m) 模型）
    seed : int, optional
        随机种子
    directed : bool, default False
        是否生成有向图

    Returns
    -------
    nx.Graph | nx.DiGraph
        生成的随机图

    Examples
    --------
    >>> G = generate_er_network(100, p=0.1)  # G(n, p) 模型
    >>> G = generate_er_network(100, m=500)  # G(n, m) 模型
    """
    if p is not None:
        return nx.erdos_renyi_graph(n, p, seed=seed, directed=directed)
    elif m is not None:
        return nx.gnm_random_graph(n, m, seed=seed, directed=directed)
    else:
        raise ValueError("必须指定 p 或 m 参数")


def generate_ba_network(
    n: int,
    m: int,
    *,
    seed: int | None = None,
) -> nx.Graph:
    """
    生成 Barabási-Albert 无标度网络

    使用优先连接机制生成，度分布服从幂律分布

    Parameters
    ----------
    n : int
        最终节点数量
    m : int
        每个新节点连接的已有节点数
    seed : int, optional
        随机种子

    Returns
    -------
    nx.Graph
        生成的无标度网络

    Examples
    --------
    >>> G = generate_ba_network(1000, 3)  # 1000 节点，每个新节点连接 3 个已有节点
    """
    return nx.barabasi_albert_graph(n, m, seed=seed)


def generate_ws_network(
    n: int,
    k: int,
    p: float,
    *,
    seed: int | None = None,
) -> nx.Graph:
    """
    生成 Watts-Strogatz 小世界网络

    从规则环形格子开始，以概率 p 重连每条边

    Parameters
    ----------
    n : int
        节点数量
    k : int
        每个节点初始连接的邻居数（必须为偶数）
    p : float
        重连概率
    seed : int, optional
        随机种子

    Returns
    -------
    nx.Graph
        生成的小世界网络

    Examples
    --------
    >>> G = generate_ws_network(100, 4, 0.3)  # 100 节点，初始度 4，重连概率 0.3
    """
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def generate_sbm_network(
    sizes: Sequence[int],
    p_matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    seed: int | None = None,
    directed: bool = False,
) -> nx.Graph | nx.DiGraph:
    """
    生成随机块模型 (Stochastic Block Model) 网络

    Parameters
    ----------
    sizes : Sequence[int]
        每个社区的节点数量
    p_matrix : array-like
        连接概率矩阵，p_matrix[i][j] 表示社区 i 到社区 j 的边概率
    seed : int, optional
        随机种子
    directed : bool, default False
        是否生成有向图

    Returns
    -------
    nx.Graph | nx.DiGraph
        生成的随机块模型网络

    Examples
    --------
    >>> # 3 个社区，各 50 个节点，组内连接概率 0.3，组间连接概率 0.01
    >>> sizes = [50, 50, 50]
    >>> p_matrix = [[0.3, 0.01, 0.01],
    ...             [0.01, 0.3, 0.01],
    ...             [0.01, 0.01, 0.3]]
    >>> G = generate_sbm_network(sizes, p_matrix)
    """
    p_matrix = np.array(p_matrix)
    return nx.stochastic_block_model(
        sizes, p_matrix.tolist(), seed=seed, directed=directed
    )


def generate_configuration_model(
    degree_sequence: Sequence[int],
    *,
    seed: int | None = None,
    create_using: type | None = None,
) -> nx.Graph:
    """
    使用配置模型生成指定度序列的随机图

    Parameters
    ----------
    degree_sequence : Sequence[int]
        目标度序列，总和必须为偶数
    seed : int, optional
        随机种子
    create_using : type, optional
        图类型

    Returns
    -------
    nx.Graph
        生成的图

    Examples
    --------
    >>> degrees = [3, 3, 3, 3, 2, 2, 2, 2]  # 度序列
    >>> G = generate_configuration_model(degrees)
    """
    return nx.configuration_model(degree_sequence, seed=seed, create_using=create_using)


def generate_powerlaw_cluster_graph(
    n: int,
    m: int,
    p: float,
    *,
    seed: int | None = None,
) -> nx.Graph:
    """
    生成幂律聚类图

    结合 BA 模型的优先连接和三角形闭合

    Parameters
    ----------
    n : int
        节点数量
    m : int
        每个新节点的边数
    p : float
        添加三角形的概率
    seed : int, optional
        随机种子

    Returns
    -------
    nx.Graph
        生成的图
    """
    return nx.powerlaw_cluster_graph(n, m, p, seed=seed)


def generate_random_geometric_graph(
    n: int,
    radius: float,
    *,
    dim: int = 2,
    seed: int | None = None,
) -> nx.Graph:
    """
    生成随机几何图

    节点随机分布在单位空间中，距离小于 radius 的节点相连

    Parameters
    ----------
    n : int
        节点数量
    radius : float
        连接阈值距离
    dim : int, default 2
        空间维度
    seed : int, optional
        随机种子

    Returns
    -------
    nx.Graph
        生成的随机几何图（节点带有 pos 属性）

    Examples
    --------
    >>> G = generate_random_geometric_graph(100, 0.2)
    >>> pos = nx.get_node_attributes(G, 'pos')  # 获取节点位置
    """
    return nx.random_geometric_graph(n, radius, dim=dim, seed=seed)


def generate_scale_free_network(
    n: int,
    *,
    alpha: float = 0.41,
    beta: float = 0.54,
    gamma: float = 0.05,
    delta_in: float = 0.2,
    delta_out: float = 0,
    seed: int | None = None,
) -> nx.DiGraph:
    """
    生成扩展的无标度有向图

    Parameters
    ----------
    n : int
        节点数量
    alpha : float
        添加新边到新节点的概率
    beta : float
        添加新边到已有节点的概率
    gamma : float
        添加新边从已有节点的概率
    delta_in : float
        入度偏好
    delta_out : float
        出度偏好
    seed : int, optional
        随机种子

    Returns
    -------
    nx.DiGraph
        生成的有向无标度网络
    """
    return nx.scale_free_graph(
        n,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta_in=delta_in,
        delta_out=delta_out,
        seed=seed,
    )
