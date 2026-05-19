"""
网络分析模块
============

提供网络中心性、社区检测、网络指标计算等功能
"""

from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx
import numpy as np
import polars as pl


@dataclass
class NetworkMetrics:
    """网络指标数据类"""

    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    avg_clustering: float
    transitivity: float
    avg_path_length: float | None
    diameter: int | None
    is_connected: bool
    n_components: int
    largest_component_size: int
    assortativity: float | None


@dataclass
class CentralityResult:
    """中心性计算结果"""

    degree: dict[Any, float]
    betweenness: dict[Any, float]
    closeness: dict[Any, float]
    eigenvector: dict[Any, float] | None
    pagerank: dict[Any, float]
    katz: dict[Any, float] | None

    def to_polars(self) -> pl.DataFrame:
        """转换为 Polars DataFrame"""
        nodes = list(self.degree.keys())
        data = {
            "node": nodes,
            "degree_centrality": [self.degree[n] for n in nodes],
            "betweenness_centrality": [self.betweenness[n] for n in nodes],
            "closeness_centrality": [self.closeness[n] for n in nodes],
            "pagerank": [self.pagerank[n] for n in nodes],
        }
        if self.eigenvector:
            data["eigenvector_centrality"] = [self.eigenvector[n] for n in nodes]
        if self.katz:
            data["katz_centrality"] = [self.katz[n] for n in nodes]
        return pl.DataFrame(data)


@dataclass
class CommunityResult:
    """社区检测结果"""

    communities: list[set]
    modularity: float
    n_communities: int
    node_to_community: dict[Any, int]

    def to_polars(self) -> pl.DataFrame:
        """转换为 Polars DataFrame"""
        nodes = list(self.node_to_community.keys())
        communities = [self.node_to_community[n] for n in nodes]
        return pl.DataFrame({"node": nodes, "community": communities})


class NetworkAnalyzer:
    """
    网络分析器

    提供全面的网络分析功能，包括中心性计算、社区检测、网络指标等

    Parameters
    ----------
    G : nx.Graph | nx.DiGraph
        NetworkX 图对象

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> analyzer = NetworkAnalyzer(G)
    >>> metrics = analyzer.compute_metrics()
    >>> centralities = analyzer.compute_centralities()
    >>> communities = analyzer.detect_communities()
    """

    def __init__(self, G: nx.Graph | nx.DiGraph):
        self.G = G
        self._is_directed = G.is_directed()

    @property
    def n_nodes(self) -> int:
        """节点数量"""
        return self.G.number_of_nodes()

    @property
    def n_edges(self) -> int:
        """边数量"""
        return self.G.number_of_edges()

    def compute_metrics(self) -> NetworkMetrics:
        """
        计算网络基本指标

        Returns
        -------
        NetworkMetrics
            包含各类网络指标的数据类
        """
        G = self.G

        # 基本指标
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G)

        # 度相关
        degrees = [d for _, d in G.degree()]
        avg_degree = np.mean(degrees) if degrees else 0.0

        # 聚类系数
        avg_clustering = nx.average_clustering(G)
        transitivity = nx.transitivity(G)

        # 连通性
        if self._is_directed:
            is_connected = nx.is_weakly_connected(G)
            components = list(nx.weakly_connected_components(G))
        else:
            is_connected = nx.is_connected(G)
            components = list(nx.connected_components(G))

        n_components = len(components)
        largest_component_size = max(len(c) for c in components) if components else 0

        # 路径相关（仅对连通图计算）
        if is_connected:
            if self._is_directed:
                largest = G.subgraph(max(nx.weakly_connected_components(G), key=len))
            else:
                largest = G
            try:
                avg_path_length = nx.average_shortest_path_length(largest)
                diameter = nx.diameter(largest)
            except nx.NetworkXError:
                avg_path_length = None
                diameter = None
        else:
            # 对最大连通分量计算
            largest = G.subgraph(max(components, key=len))
            if largest.number_of_nodes() > 1:
                try:
                    avg_path_length = nx.average_shortest_path_length(largest)
                    diameter = nx.diameter(largest)
                except nx.NetworkXError:
                    avg_path_length = None
                    diameter = None
            else:
                avg_path_length = None
                diameter = None

        # 度相关性
        try:
            assortativity = nx.degree_assortativity_coefficient(G)
        except (nx.NetworkXError, ValueError):
            assortativity = None

        return NetworkMetrics(
            n_nodes=n_nodes,
            n_edges=n_edges,
            density=density,
            avg_degree=avg_degree,
            avg_clustering=avg_clustering,
            transitivity=transitivity,
            avg_path_length=avg_path_length,
            diameter=diameter,
            is_connected=is_connected,
            n_components=n_components,
            largest_component_size=largest_component_size,
            assortativity=assortativity,
        )

    def compute_centralities(
        self,
        *,
        include_eigenvector: bool = True,
        include_katz: bool = False,
        weight: str | None = None,
    ) -> CentralityResult:
        """
        计算多种中心性指标

        Parameters
        ----------
        include_eigenvector : bool, default True
            是否计算特征向量中心性
        include_katz : bool, default False
            是否计算 Katz 中心性
        weight : str, optional
            边权重属性名

        Returns
        -------
        CentralityResult
            包含各类中心性的数据类
        """
        G = self.G

        # 度中心性
        degree = nx.degree_centrality(G)

        # 介数中心性
        betweenness = nx.betweenness_centrality(G, weight=weight)

        # 接近中心性
        closeness = nx.closeness_centrality(G)

        # PageRank
        pagerank = nx.pagerank(G, weight=weight)

        # 特征向量中心性
        eigenvector = None
        if include_eigenvector:
            try:
                eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight=weight)
            except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
                try:
                    eigenvector = nx.eigenvector_centrality_numpy(G, weight=weight)
                except Exception:
                    eigenvector = None

        # Katz 中心性
        katz = None
        if include_katz:
            try:
                katz = nx.katz_centrality(G, weight=weight)
            except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
                try:
                    katz = nx.katz_centrality_numpy(G, weight=weight)
                except Exception:
                    katz = None

        return CentralityResult(
            degree=degree,
            betweenness=betweenness,
            closeness=closeness,
            eigenvector=eigenvector,
            pagerank=pagerank,
            katz=katz,
        )

    def detect_communities(
        self,
        method: Literal[
            "louvain", "label_propagation", "girvan_newman", "greedy"
        ] = "louvain",
        *,
        resolution: float = 1.0,
        seed: int | None = None,
        n_communities: int | None = None,
    ) -> CommunityResult:
        """
        社区检测

        Parameters
        ----------
        method : str, default "louvain"
            检测方法：
            - "louvain": Louvain 算法，基于模块度优化
            - "label_propagation": 标签传播算法
            - "girvan_newman": Girvan-Newman 算法，基于边介数
            - "greedy": 贪婪模块度优化
        resolution : float, default 1.0
            Louvain 算法的分辨率参数，小于 1 倾向大社区，大于 1 倾向小社区
        seed : int, optional
            随机种子
        n_communities : int, optional
            目标社区数（仅 girvan_newman 方法使用）

        Returns
        -------
        CommunityResult
            社区检测结果
        """
        G = self.G
        if self._is_directed:
            G = G.to_undirected()

        if method == "louvain":
            communities = list(
                nx.community.louvain_communities(G, resolution=resolution, seed=seed)
            )
        elif method == "label_propagation":
            communities = list(nx.community.label_propagation_communities(G))
        elif method == "girvan_newman":
            comp = nx.community.girvan_newman(G)
            if n_communities is not None:
                for _ in range(n_communities - 1):
                    try:
                        communities = next(comp)
                    except StopIteration:
                        break
                communities = list(communities)
            else:
                communities = list(next(comp))
        elif method == "greedy":
            communities = list(nx.community.greedy_modularity_communities(G))
        else:
            raise ValueError(f"未知的社区检测方法: {method}")

        # 计算模块度
        modularity = nx.community.modularity(G, communities)

        # 构建节点到社区的映射
        node_to_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = i

        return CommunityResult(
            communities=communities,
            modularity=modularity,
            n_communities=len(communities),
            node_to_community=node_to_community,
        )

    def get_degree_distribution(self) -> pl.DataFrame:
        """
        获取度分布

        Returns
        -------
        pl.DataFrame
            包含度值和频次的 DataFrame
        """
        degrees = [d for _, d in self.G.degree()]
        unique, counts = np.unique(degrees, return_counts=True)
        return pl.DataFrame(
            {
                "degree": unique.tolist(),
                "count": counts.tolist(),
                "frequency": (counts / len(degrees)).tolist(),
            }
        )

    def get_neighbors(self, node: Any) -> list:
        """获取节点的邻居"""
        return list(self.G.neighbors(node))

    def get_shortest_path(self, source: Any, target: Any) -> list | None:
        """获取两节点间的最短路径"""
        try:
            return nx.shortest_path(self.G, source, target)
        except nx.NetworkXNoPath:
            return None

    def get_k_core(self, k: int) -> nx.Graph:
        """获取 k-核子图"""
        return nx.k_core(self.G, k=k)

    def find_bridges(self) -> list:
        """查找桥边（移除后会断开图的边）"""
        if self._is_directed:
            return []
        return list(nx.bridges(self.G))

    def find_articulation_points(self) -> list:
        """查找关节点（移除后会断开图的节点）"""
        if self._is_directed:
            return []
        return list(nx.articulation_points(self.G))


def analyze_network(G: nx.Graph | nx.DiGraph) -> dict[str, Any]:
    """
    快速网络分析

    Parameters
    ----------
    G : nx.Graph | nx.DiGraph
        NetworkX 图对象

    Returns
    -------
    dict
        包含网络指标、中心性和社区信息的字典

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> results = analyze_network(G)
    >>> print(results["metrics"])
    >>> print(results["centralities"].to_polars())
    """
    analyzer = NetworkAnalyzer(G)
    return {
        "metrics": analyzer.compute_metrics(),
        "centralities": analyzer.compute_centralities(),
        "communities": analyzer.detect_communities(),
        "degree_distribution": analyzer.get_degree_distribution(),
    }


def calculate_centralities(
    G: nx.Graph | nx.DiGraph,
    *,
    weight: str | None = None,
) -> pl.DataFrame:
    """
    计算中心性并返回 DataFrame

    Parameters
    ----------
    G : nx.Graph | nx.DiGraph
        NetworkX 图对象
    weight : str, optional
        边权重属性名

    Returns
    -------
    pl.DataFrame
        包含各类中心性的 DataFrame
    """
    analyzer = NetworkAnalyzer(G)
    return analyzer.compute_centralities(weight=weight).to_polars()


def detect_communities(
    G: nx.Graph | nx.DiGraph,
    method: str = "louvain",
    **kwargs,
) -> pl.DataFrame:
    """
    社区检测并返回 DataFrame

    Parameters
    ----------
    G : nx.Graph | nx.DiGraph
        NetworkX 图对象
    method : str, default "louvain"
        检测方法
    **kwargs
        传递给检测方法的参数

    Returns
    -------
    pl.DataFrame
        包含节点和社区标签的 DataFrame
    """
    analyzer = NetworkAnalyzer(G)
    return analyzer.detect_communities(method=method, **kwargs).to_polars()


def network_metrics(G: nx.Graph | nx.DiGraph) -> dict[str, Any]:
    """
    计算网络指标

    Parameters
    ----------
    G : nx.Graph | nx.DiGraph
        NetworkX 图对象

    Returns
    -------
    dict
        网络指标字典
    """
    analyzer = NetworkAnalyzer(G)
    metrics = analyzer.compute_metrics()
    return {
        "n_nodes": metrics.n_nodes,
        "n_edges": metrics.n_edges,
        "density": metrics.density,
        "avg_degree": metrics.avg_degree,
        "avg_clustering": metrics.avg_clustering,
        "transitivity": metrics.transitivity,
        "avg_path_length": metrics.avg_path_length,
        "diameter": metrics.diameter,
        "is_connected": metrics.is_connected,
        "n_components": metrics.n_components,
        "largest_component_size": metrics.largest_component_size,
        "assortativity": metrics.assortativity,
    }
