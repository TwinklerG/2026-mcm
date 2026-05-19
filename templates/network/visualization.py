"""
网络可视化模块
==============

提供网络可视化功能，支持 Plotly 交互式和 Matplotlib 论文级图表
"""

from typing import Literal

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from ..viz.themes import apply_theme, get_color_palette


def plot_network(
    G: nx.Graph | nx.DiGraph,
    *,
    layout: Literal[
        "spring", "circular", "kamada_kawai", "shell", "spectral", "random"
    ] = "spring",
    node_color: str | list | dict | None = None,
    node_size: str | list | dict | float = 10,
    node_labels: bool = True,
    edge_width: str | float = 0.5,
    edge_color: str = "rgba(180, 180, 180, 0.3)",
    title: str = "",
    width: int = 900,
    height: int = 700,
    colorscale: str = "Blues",
    show_colorbar: bool = True,
    theme: str = "nature",
    seed: int | None = 42,
) -> go.Figure:
    """
    绘制网络图

    Parameters
    ----------
    G : nx.Graph | nx.DiGraph
        NetworkX 图对象
    layout : str, default "spring"
        布局算法
    node_color : str | list | dict, optional
        节点颜色
    node_size : str | list | dict | float, default 10
        节点大小
    node_labels : bool, default True
        是否显示节点标签
    edge_width : str | float, default 0.5
        边宽度
    edge_color : str, default "rgba(180, 180, 180, 0.3)"
        边颜色
    title : str
        图表标题
    width, height : int
        图表尺寸
    colorscale : str, default "Blues"
        颜色映射(Plotly 内置配色)
    show_colorbar : bool, default True
        是否显示颜色条
    theme : str, default "nature"
        图表主题
    seed : int, optional
        布局随机种子

    Returns
    -------
    go.Figure
        Plotly 图表对象
    """
    # 计算布局
    layout_funcs = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "shell": nx.shell_layout,
        "spectral": nx.spectral_layout,
        "random": nx.random_layout,
    }

    pos = (
        layout_funcs[layout](G, seed=seed)
        if seed is not None
        else layout_funcs[layout](G)
    )

    # 创建边
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(
            width=edge_width if isinstance(edge_width, (int, float)) else 1.0,
            color=edge_color,
        ),
        hoverinfo="none",
        showlegend=False,
    )

    # 创建节点
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    # 处理节点颜色
    if node_color is None:
        node_colors = [G.degree(node) for node in G.nodes()]
    elif isinstance(node_color, str):
        node_colors = node_color
    elif isinstance(node_color, dict):
        node_colors = [node_color.get(node, 0) for node in G.nodes()]
    elif isinstance(node_color, list):
        node_colors = node_color
    else:
        node_colors = node_color

    # 处理节点大小
    if isinstance(node_size, (int, float)):
        node_sizes = node_size
    elif isinstance(node_size, dict):
        node_sizes = [node_size.get(node, 10) for node in G.nodes()]
    elif isinstance(node_size, str):
        if node_size == "degree":
            max_degree = max(dict(G.degree()).values())
            node_sizes = [10 + 20 * G.degree(node) / max_degree for node in G.nodes()]
        else:
            node_sizes = 10
    else:
        node_sizes = node_size

    # 节点文本
    node_text = (
        [f"Node: {node}<br>Degree: {G.degree(node)}" for node in G.nodes()]
        if node_labels
        else None
    )

    marker_dict = dict(
        size=node_sizes,
        color=node_colors,
        colorscale=colorscale,
        line=dict(width=1, color="white"),
        showscale=show_colorbar and not isinstance(node_colors, str),
        colorbar=dict(thickness=15, title="Value") if show_colorbar else None,
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=marker_dict,
        showlegend=False,
    )

    # 创建图表
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            showlegend=False,
            hovermode="closest",
            width=width,
            height=height,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
        ),
    )

    return apply_theme(fig, theme=theme, show_grid=False)


def plot_degree_distribution(
    G: nx.Graph | nx.DiGraph,
    *,
    log_scale: bool = True,
    fit_powerlaw: bool = False,
    title: str = "Degree Distribution",
    width: int = 900,
    height: int = 600,
    theme: str = "nature",
) -> go.Figure:
    """
    绘制度分布图

    Parameters
    ----------
    G : nx.Graph | nx.DiGraph
        NetworkX 图对象
    log_scale : bool, default True
        是否使用对数坐标
    fit_powerlaw : bool, default False
        是否拟合幂律分布
    title : str
        图表标题
    width, height : int
        图表尺寸
    theme : str, default "nature"
        图表主题

    Returns
    -------
    go.Figure
        Plotly 图表对象
    """
    degrees = [d for _, d in G.degree()]
    unique, counts = np.unique(degrees, return_counts=True)
    freq = counts / len(degrees)

    colors = get_color_palette("nature")
    fig = go.Figure()

    # 散点图
    fig.add_trace(
        go.Scatter(
            x=unique,
            y=freq,
            mode="markers",
            marker=dict(size=10, color=colors[0], line=dict(width=1, color="white")),
            name="Observed",
        )
    )

    # 幂律拟合
    if fit_powerlaw and len(unique) > 2:
        mask = (unique > 0) & (freq > 0)
        if mask.sum() > 1:
            log_k = np.log10(unique[mask])
            log_p = np.log10(freq[mask])
            coef = np.polyfit(log_k, log_p, 1)
            slope = coef[0]

            k_fit = np.logspace(np.log10(unique.min()), np.log10(unique.max()), 100)
            p_fit = 10 ** np.polyval(coef, np.log10(k_fit))

            fig.add_trace(
                go.Scatter(
                    x=k_fit,
                    y=p_fit,
                    mode="lines",
                    line=dict(color=colors[1], dash="dash", width=2.5),
                    name=f"Power Law (γ={-slope:.2f})",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Degree (k)",
        yaxis_title="P(k)",
        width=width,
        height=height,
    )

    if log_scale:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

    return apply_theme(fig, theme=theme)


def plot_centrality_comparison(
    G: nx.Graph | nx.DiGraph,
    *,
    centralities: list[str] | None = None,
    top_n: int = 20,
    title: str = "Centrality Comparison",
    width: int = 900,
    height: int = 600,
    theme: str = "nature",
) -> go.Figure:
    """
    对比不同中心性指标

    Parameters
    ----------
    G : nx.Graph | nx.DiGraph
        NetworkX 图对象
    centralities : list[str], optional
        要对比的中心性指标,默认 ["degree", "betweenness", "closeness"]
    top_n : int, default 20
        显示前 n 个节点
    title : str
        图表标题
    width, height : int
        图表尺寸
    theme : str, default "nature"
        图表主题

    Returns
    -------
    go.Figure
        Plotly 图表对象
    """
    if centralities is None:
        centralities = ["degree", "betweenness", "closeness"]

    all_centralities = {
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G),
        "closeness": nx.closeness_centrality(G),
        "pagerank": nx.pagerank(G),
        "eigenvector": nx.eigenvector_centrality(G, max_iter=1000)
        if nx.is_connected(G.to_undirected())
        else None,
    }

    # 获取 top_n 节点(按 degree 排序)
    degree = all_centralities["degree"]
    top_nodes = sorted(degree.keys(), key=lambda x: degree[x], reverse=True)[:top_n]

    colors = get_color_palette("nature")
    fig = go.Figure()

    for i, name in enumerate(centralities):
        if name in all_centralities and all_centralities[name] is not None:
            values = [all_centralities[name][n] for n in top_nodes]
            fig.add_trace(
                go.Bar(
                    x=[str(n) for n in top_nodes],
                    y=values,
                    name=name.capitalize(),
                    marker_color=colors[i % len(colors)],
                    marker_line=dict(width=1, color="white"),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Node",
        yaxis_title="Centrality",
        barmode="group",
        width=width,
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return apply_theme(fig, theme=theme)


def plot_community_network(
    G: nx.Graph | nx.DiGraph,
    communities: list[set] | dict,
    *,
    layout: str = "spring",
    title: str = "Community Structure",
    width: int = 900,
    height: int = 700,
    theme: str = "nature",
    seed: int | None = 42,
) -> go.Figure:
    """
    可视化社区结构

    Parameters
    ----------
    G : nx.Graph | nx.DiGraph
        NetworkX 图对象
    communities : list[set] | dict
        社区分配结果
    layout : str
        布局算法
    title : str
        图表标题
    width, height : int
        图表尺寸
    theme : str, default "nature"
        图表主题
    seed : int, optional
        布局随机种子

    Returns
    -------
    go.Figure
        Plotly 图表对象
    """
    # 转换社区格式
    if isinstance(communities, dict):
        node_to_community = communities
    else:
        node_to_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = i

    # 使用离散配色
    n_communities = len(set(node_to_community.values()))
    colors = get_color_palette("nature", n_colors=n_communities)

    # 为每个节点分配颜色
    node_colors = [colors[node_to_community.get(n, 0) % len(colors)] for n in G.nodes()]

    return plot_network(
        G,
        layout=layout,
        node_color=node_colors,
        title=title,
        width=width,
        height=height,
        show_colorbar=False,
        theme=theme,
        seed=seed,
    )


def plot_sir_dynamics(
    history: list[dict],
    *,
    title: str = "SIR Dynamics",
    width: int = 900,
    height: int = 600,
    theme: str = "nature",
) -> go.Figure:
    """
    可视化 SIR/SIS 动力学过程

    Parameters
    ----------
    history : list[dict]
        时间序列数据,每个字典包含 "S", "I", "R" 计数
    title : str
        图表标题
    width, height : int
        图表尺寸
    theme : str, default "nature"
        图表主题

    Returns
    -------
    go.Figure
        Plotly 图表对象
    """
    time_steps = list(range(len(history)))
    susceptible = [h["S"] for h in history]
    infected = [h["I"] for h in history]
    recovered = [h.get("R", 0) for h in history]

    colors = get_color_palette("nature")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time_steps,
            y=susceptible,
            mode="lines",
            name="Susceptible",
            line=dict(color=colors[2], width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_steps,
            y=infected,
            mode="lines",
            name="Infected",
            line=dict(color=colors[0], width=2.5),
        )
    )
    if any(r > 0 for r in recovered):
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=recovered,
                mode="lines",
                name="Recovered",
                line=dict(color=colors[1], width=2.5),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time Step",
        yaxis_title="Number of Individuals",
        width=width,
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return apply_theme(fig, theme=theme)


def plot_adjacency_matrix(
    G: nx.Graph | nx.DiGraph,
    *,
    nodelist: list | None = None,
    title: str = "Adjacency Matrix",
    width: int = 800,
    height: int = 700,
    colorscale: str = "Blues",
    theme: str = "nature",
) -> go.Figure:
    """
    绘制邻接矩阵热力图

    Parameters
    ----------
    G : nx.Graph | nx.DiGraph
        NetworkX 图对象
    nodelist : list, optional
        节点顺序
    title : str
        图表标题
    width, height : int
        图表尺寸
    colorscale : str
        颜色映射
    theme : str, default "nature"
        图表主题

    Returns
    -------
    go.Figure
        Plotly 图表对象
    """
    if nodelist is None:
        nodelist = list(G.nodes())

    adj_matrix = nx.to_numpy_array(G, nodelist=nodelist)

    fig = go.Figure(
        data=go.Heatmap(
            z=adj_matrix,
            x=[str(n) for n in nodelist],
            y=[str(n) for n in nodelist],
            colorscale=colorscale,
            showscale=True,
        )
    )

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis=dict(title="Node"),
        yaxis=dict(title="Node", autorange="reversed"),
    )

    return apply_theme(fig, theme=theme)
