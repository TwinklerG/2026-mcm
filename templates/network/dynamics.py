"""
网络传播动力学模块
==================

提供网络上的传播模型，包括：
- SIR 模型（易感-感染-恢复）
- SIS 模型（易感-感染-易感）
- 级联传播模型
"""

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import polars as pl


@dataclass
class SIRResult:
    """SIR 模型模拟结果"""

    history: list[dict[str, int]]  # 每个时间步的 S, I, R 数量
    final_state: dict[Any, str]  # 每个节点的最终状态
    total_infected: int  # 总感染人数
    peak_infected: int  # 感染峰值
    peak_time: int  # 感染峰值时间

    def to_polars(self) -> pl.DataFrame:
        """转换为时间序列 DataFrame"""
        return pl.DataFrame(self.history).with_row_index("time")


@dataclass
class SISResult:
    """SIS 模型模拟结果"""

    history: list[dict[str, int]]  # 每个时间步的 S, I 数量
    final_state: dict[Any, str]  # 每个节点的最终状态
    endemic_infected: float  # 地方病平衡态的感染比例

    def to_polars(self) -> pl.DataFrame:
        """转换为时间序列 DataFrame"""
        return pl.DataFrame(self.history).with_row_index("time")


@dataclass
class CascadeResult:
    """级联传播结果"""

    activated_nodes: list[Any]  # 被激活的节点列表
    activation_time: dict[Any, int]  # 节点激活时间
    cascade_size: int  # 级联规模
    history: list[set]  # 每个时间步新激活的节点

    def to_polars(self) -> pl.DataFrame:
        """转换为 DataFrame"""
        nodes = list(self.activation_time.keys())
        times = [self.activation_time[n] for n in nodes]
        return pl.DataFrame({"node": nodes, "activation_time": times})


def sir_model(
    G: nx.Graph,
    beta: float,
    gamma: float,
    initial_infected: list | set | int = 1,
    *,
    max_steps: int = 1000,
    seed: int | None = None,
) -> SIRResult:
    """
    SIR 传播模型模拟

    S (Susceptible) -> I (Infected) -> R (Recovered)

    Parameters
    ----------
    G : nx.Graph
        网络图
    beta : float
        感染率，每条边每时间步的感染概率
    gamma : float
        恢复率，感染节点每时间步的恢复概率
    initial_infected : list | set | int, default 1
        初始感染节点，可以是：
        - 节点列表或集合
        - 整数（随机选择的初始感染节点数）
    max_steps : int, default 1000
        最大模拟步数
    seed : int, optional
        随机种子

    Returns
    -------
    SIRResult
        模拟结果

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.barabasi_albert_graph(1000, 3)
    >>> result = sir_model(G, beta=0.3, gamma=0.1, initial_infected=5)
    >>> df = result.to_polars()
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())

    # 初始化状态
    state = {node: "S" for node in nodes}

    if isinstance(initial_infected, int):
        initial_infected = rng.choice(nodes, size=initial_infected, replace=False)
    for node in initial_infected:
        state[node] = "I"

    history = []
    infected_count = sum(1 for s in state.values() if s == "I")
    susceptible_count = sum(1 for s in state.values() if s == "S")
    recovered_count = 0

    history.append({"S": susceptible_count, "I": infected_count, "R": recovered_count})

    peak_infected = infected_count
    peak_time = 0
    total_infected = infected_count

    for step in range(1, max_steps + 1):
        new_state = state.copy()

        # 感染过程
        infected_nodes = [n for n, s in state.items() if s == "I"]
        for node in infected_nodes:
            for neighbor in G.neighbors(node):
                if state[neighbor] == "S" and rng.random() < beta:
                    new_state[neighbor] = "I"
                    total_infected += 1

        # 恢复过程
        for node in infected_nodes:
            if rng.random() < gamma:
                new_state[node] = "R"

        state = new_state

        # 统计
        infected_count = sum(1 for s in state.values() if s == "I")
        susceptible_count = sum(1 for s in state.values() if s == "S")
        recovered_count = sum(1 for s in state.values() if s == "R")

        history.append(
            {
                "S": susceptible_count,
                "I": infected_count,
                "R": recovered_count,
            }
        )

        if infected_count > peak_infected:
            peak_infected = infected_count
            peak_time = step

        # 终止条件
        if infected_count == 0:
            break

    return SIRResult(
        history=history,
        final_state=state,
        total_infected=total_infected,
        peak_infected=peak_infected,
        peak_time=peak_time,
    )


def sis_model(
    G: nx.Graph,
    beta: float,
    gamma: float,
    initial_infected: list | set | int = 1,
    *,
    max_steps: int = 1000,
    seed: int | None = None,
) -> SISResult:
    """
    SIS 传播模型模拟

    S (Susceptible) -> I (Infected) -> S (Susceptible)

    Parameters
    ----------
    G : nx.Graph
        网络图
    beta : float
        感染率
    gamma : float
        恢复率
    initial_infected : list | set | int, default 1
        初始感染节点
    max_steps : int, default 1000
        最大模拟步数
    seed : int, optional
        随机种子

    Returns
    -------
    SISResult
        模拟结果

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.barabasi_albert_graph(1000, 3)
    >>> result = sis_model(G, beta=0.3, gamma=0.5, initial_infected=10)
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())

    # 初始化状态
    state = {node: "S" for node in nodes}

    if isinstance(initial_infected, int):
        initial_infected = rng.choice(nodes, size=initial_infected, replace=False)
    for node in initial_infected:
        state[node] = "I"

    history = []
    infected_count = sum(1 for s in state.values() if s == "I")
    susceptible_count = len(nodes) - infected_count

    history.append({"S": susceptible_count, "I": infected_count})

    # 记录最后几步用于计算地方病平衡态
    last_infected_counts = []

    for step in range(1, max_steps + 1):
        new_state = state.copy()

        # 感染过程
        infected_nodes = [n for n, s in state.items() if s == "I"]
        for node in infected_nodes:
            for neighbor in G.neighbors(node):
                if state[neighbor] == "S" and rng.random() < beta:
                    new_state[neighbor] = "I"

        # 恢复过程（恢复为易感状态）
        for node in infected_nodes:
            if rng.random() < gamma:
                new_state[node] = "S"

        state = new_state

        # 统计
        infected_count = sum(1 for s in state.values() if s == "I")
        susceptible_count = len(nodes) - infected_count

        history.append({"S": susceptible_count, "I": infected_count})

        # 记录最后 100 步
        if step > max_steps - 100:
            last_infected_counts.append(infected_count)

        # 如果感染清除则终止
        if infected_count == 0:
            break

    # 计算地方病平衡态
    endemic_infected = (
        np.mean(last_infected_counts) / len(nodes) if last_infected_counts else 0.0
    )

    return SISResult(
        history=history,
        final_state=state,
        endemic_infected=endemic_infected,
    )


def simulate_cascade(
    G: nx.Graph,
    seeds: list | set,
    threshold: float | dict[Any, float] = 0.5,
    *,
    max_steps: int = 100,
) -> CascadeResult:
    """
    线性阈值模型级联传播模拟

    节点在邻居中已激活比例超过阈值时被激活

    Parameters
    ----------
    G : nx.Graph
        网络图
    seeds : list | set
        初始激活（种子）节点
    threshold : float | dict, default 0.5
        激活阈值，可以是：
        - 单一数值（所有节点使用相同阈值）
        - 字典（每个节点使用不同阈值）
    max_steps : int, default 100
        最大模拟步数

    Returns
    -------
    CascadeResult
        级联传播结果

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> result = simulate_cascade(G, seeds=[0, 33], threshold=0.3)
    >>> print(f"级联规模: {result.cascade_size}")
    """
    activated = set(seeds)
    activation_time = {node: 0 for node in seeds}
    history = [set(seeds)]

    # 设置阈值
    if isinstance(threshold, (int, float)):
        thresholds = {node: threshold for node in G.nodes()}
    else:
        thresholds = threshold

    for step in range(1, max_steps + 1):
        new_activated = set()

        for node in G.nodes():
            if node in activated:
                continue

            # 计算已激活邻居的比例
            neighbors = set(G.neighbors(node))
            if not neighbors:
                continue

            activated_neighbors = neighbors & activated
            activation_ratio = len(activated_neighbors) / len(neighbors)

            if activation_ratio >= thresholds.get(node, 0.5):
                new_activated.add(node)
                activation_time[node] = step

        if not new_activated:
            break

        activated.update(new_activated)
        history.append(new_activated)

    return CascadeResult(
        activated_nodes=list(activated),
        activation_time=activation_time,
        cascade_size=len(activated),
        history=history,
    )


def independent_cascade(
    G: nx.Graph,
    seeds: list | set,
    p: float | dict = 0.1,
    *,
    max_steps: int = 100,
    seed: int | None = None,
) -> CascadeResult:
    """
    独立级联模型

    已激活节点有概率 p 激活其未激活的邻居

    Parameters
    ----------
    G : nx.Graph
        网络图
    seeds : list | set
        初始激活（种子）节点
    p : float | dict, default 0.1
        激活概率，可以是边属性名或单一数值
    max_steps : int, default 100
        最大模拟步数
    seed : int, optional
        随机种子

    Returns
    -------
    CascadeResult
        级联传播结果
    """
    rng = np.random.default_rng(seed)
    activated = set(seeds)
    activation_time = {node: 0 for node in seeds}
    history = [set(seeds)]

    newly_activated = set(seeds)

    for step in range(1, max_steps + 1):
        next_activated = set()

        for node in newly_activated:
            for neighbor in G.neighbors(node):
                if neighbor in activated:
                    continue

                # 获取激活概率
                if isinstance(p, dict):
                    prob = p.get((node, neighbor), p.get((neighbor, node), 0.1))
                elif isinstance(p, str):
                    prob = G[node][neighbor].get(p, 0.1)
                else:
                    prob = p

                if rng.random() < prob:
                    next_activated.add(neighbor)
                    activation_time[neighbor] = step

        if not next_activated:
            break

        activated.update(next_activated)
        newly_activated = next_activated
        history.append(next_activated)

    return CascadeResult(
        activated_nodes=list(activated),
        activation_time=activation_time,
        cascade_size=len(activated),
        history=history,
    )


def compute_basic_reproduction_number(
    G: nx.Graph,
    beta: float,
    gamma: float,
) -> float:
    """
    计算网络上的基本再生数 R0

    R0 = beta / gamma * <k^2> / <k>

    其中 <k> 是平均度，<k^2> 是度的二阶矩

    Parameters
    ----------
    G : nx.Graph
        网络图
    beta : float
        感染率
    gamma : float
        恢复率

    Returns
    -------
    float
        基本再生数 R0
    """
    degrees = np.array([d for _, d in G.degree()])
    mean_k = np.mean(degrees)
    mean_k2 = np.mean(degrees**2)

    if mean_k == 0:
        return 0.0

    return (beta / gamma) * (mean_k2 / mean_k)
