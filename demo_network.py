"""
网络科学模块示例代码
====================

演示 templates.network 模块的完整功能
"""

# %% [markdown]
# # 网络科学分析示例
#
# 本文件演示如何使用 `templates.network` 模块进行网络科学分析，
# 包括网络生成、中心性分析、社区检测和传播模型等。

# %%
import networkx as nx
import numpy as np

from templates import network

print("✅ 网络科学模块导入成功")

# %% [markdown]
# ## 1. 网络模型生成

# %%
# 1.1 Barabási-Albert 无标度网络
print("BA 无标度网络")

G_ba = network.generate_ba_network(n=500, m=3, seed=42)
print(f"节点数: {G_ba.number_of_nodes()}")
print(f"边数: {G_ba.number_of_edges()}")
print(f"平均度: {np.mean([d for _, d in G_ba.degree()]):.2f}")

# %%
# 1.2 Erdős-Rényi 随机图
print("ER 随机图")

G_er = network.generate_er_network(n=500, p=0.02, seed=42)
print(f"节点数: {G_er.number_of_nodes()}")
print(f"边数: {G_er.number_of_edges()}")

# %%
# 1.3 Watts-Strogatz 小世界网络
print("WS 小世界网络")

G_ws = network.generate_ws_network(n=500, k=6, p=0.3, seed=42)
print(f"节点数: {G_ws.number_of_nodes()}")
print(f"边数: {G_ws.number_of_edges()}")

# %%
# 1.4 随机块模型（社区结构）
print("随机块模型（SBM）")

sizes = [100, 100, 100]  # 3 个社区，各 100 个节点
p_matrix = [
    [0.3, 0.01, 0.01],  # 组内连接概率 0.3，组间 0.01
    [0.01, 0.3, 0.01],
    [0.01, 0.01, 0.3],
]
G_sbm = network.generate_sbm_network(sizes, p_matrix, seed=42)
print(f"节点数: {G_sbm.number_of_nodes()}")
print(f"边数: {G_sbm.number_of_edges()}")

# %% [markdown]
# ## 2. 网络指标分析

# %%
# 2.1 使用 NetworkAnalyzer 进行全面分析
print("网络指标分析")

# 使用空手道俱乐部网络作为示例
G = nx.karate_club_graph()
analyzer = network.NetworkAnalyzer(G)
metrics = analyzer.compute_metrics()

print(f"节点数: {metrics.n_nodes}")
print(f"边数: {metrics.n_edges}")
print(f"密度: {metrics.density:.4f}")
print(f"平均度: {metrics.avg_degree:.2f}")
print(f"平均聚类系数: {metrics.avg_clustering:.4f}")
print(f"传递性: {metrics.transitivity:.4f}")
print(f"平均路径长度: {metrics.avg_path_length:.4f}")
print(f"直径: {metrics.diameter}")
print(f"是否连通: {metrics.is_connected}")
print(f"度相关性: {metrics.assortativity:.4f}")

# %%
# 2.2 快速网络分析
print("快速网络分析")

results = network.analyze_network(G)
print(f"社区数: {results['communities'].n_communities}")
print(f"模块度: {results['communities'].modularity:.4f}")

# 度分布
degree_dist = results["degree_distribution"]
print(f"\n度分布前 5 行:")
print(degree_dist.head())

# %% [markdown]
# ## 3. 中心性分析

# %%
# 3.1 计算多种中心性
print("中心性分析")

centralities = analyzer.compute_centralities(
    include_eigenvector=True, include_katz=True
)
df_centralities = centralities.to_polars()
print(df_centralities.head(10))

# %%
# 3.2 找出最重要的节点
print("\n最重要的节点（按 PageRank）:")
top_nodes = df_centralities.sort("pagerank", descending=True).head(5)
print(
    top_nodes.select([
        "node",
        "degree_centrality",
        "betweenness_centrality",
        "pagerank",
    ])
)

# %% [markdown]
# ## 4. 社区检测

# %%
# 4.1 Louvain 算法
print("社区检测 - Louvain")

communities = analyzer.detect_communities(method="louvain", seed=42)
print(f"社区数: {communities.n_communities}")
print(f"模块度: {communities.modularity:.4f}")
print(f"社区大小: {[len(c) for c in communities.communities]}")

# %%
# 4.2 标签传播算法
print("社区检测 - 标签传播")

communities_lp = analyzer.detect_communities(method="label_propagation")
print(f"社区数: {communities_lp.n_communities}")
print(f"模块度: {communities_lp.modularity:.4f}")

# %%
# 4.3 贪婪模块度优化
print("社区检测 - 贪婪模块度")

communities_greedy = analyzer.detect_communities(method="greedy")
print(f"社区数: {communities_greedy.n_communities}")
print(f"模块度: {communities_greedy.modularity:.4f}")

# %% [markdown]
# ## 5. 传播动力学

# %%
# 5.1 SIR 模型
print("SIR 传播模型")

# 在 BA 网络上模拟传播
G_spread = network.generate_ba_network(n=1000, m=3, seed=42)
sir_result = network.sir_model(
    G_spread,
    beta=0.15,  # 感染率
    gamma=0.1,  # 恢复率
    initial_infected=5,  # 初始感染节点数
    max_steps=200,
    seed=42,
)

print(f"总感染人数: {sir_result.total_infected}")
print(f"感染峰值: {sir_result.peak_infected}")
print(f"峰值时间: {sir_result.peak_time}")

# 转换为 DataFrame 查看时间序列
df_sir = sir_result.to_polars()
print(f"\nSIR 时间序列 (前 10 步):")
print(df_sir.head(10))

# %%
# 5.2 计算基本再生数 R0
print("基本再生数 R0")

R0 = network.compute_basic_reproduction_number(G_spread, beta=0.15, gamma=0.1)
print(f"R0 = {R0:.4f}")
print(f"理论上，R0 > 1 时疫情会爆发，R0 < 1 时疫情会消亡")

# %%
# 5.3 SIS 模型
print("SIS 传播模型")

sis_result = network.sis_model(
    G_spread,
    beta=0.1,
    gamma=0.3,
    initial_infected=10,
    max_steps=500,
    seed=42,
)

print(f"地方病平衡态感染比例: {sis_result.endemic_infected:.4f}")

# %%
# 5.4 级联传播（线性阈值模型）
print("级联传播 - 线性阈值模型")

G_cascade = nx.karate_club_graph()
cascade_result = network.simulate_cascade(
    G_cascade,
    seeds=[0, 33],  # 从两个领导者开始
    threshold=0.3,  # 激活阈值
)

print(f"级联规模: {cascade_result.cascade_size}")
print(f"被激活节点: {cascade_result.activated_nodes[:10]}...")

# %% [markdown]
# ## 6. 网络可视化

# %%
# 6.1 绘制网络图
print("网络可视化")

G = nx.karate_club_graph()

# 基础网络图
fig = network.plot_network(
    G,
    layout="spring",
    node_color="#1f77b4",
    node_size="degree",
    title="Karate Club Network",
)
fig.write_html("figures/network_basic.html")
print("✅ 基础网络图已保存")

# %%
# 6.2 度分布图
fig_degree = network.plot_degree_distribution(
    G_ba,
    log_scale=True,
    fit_powerlaw=True,
    title="BA Network Degree Distribution",
)
fig_degree.write_html("figures/degree_distribution.html")
print("✅ 度分布图已保存")

# %%
# 6.3 中心性对比图
fig_centrality = network.plot_centrality_comparison(
    G,
    centralities=["degree", "betweenness", "closeness", "pagerank"],
    top_n=15,
    title="Centrality Comparison",
)
fig_centrality.write_html("figures/centrality_comparison.html")
print("✅ 中心性对比图已保存")

# %%
# 6.4 社区结构图
communities = network.detect_communities(G, method="louvain")
node_to_community = dict(
    zip(communities["node"].to_list(), communities["community"].to_list())
)

fig_community = network.plot_community_network(
    G,
    communities=node_to_community,
    title="Community Structure",
)
fig_community.write_html("figures/community_network.html")
print("✅ 社区结构图已保存")

# %%
# 6.5 SIR 动力学图
fig_sir = network.plot_sir_dynamics(
    sir_result.history,
    title="SIR Epidemic Dynamics",
)
fig_sir.write_html("figures/sir_dynamics.html")
print("✅ SIR 动力学图已保存")

# %%
# 6.6 邻接矩阵热图
fig_adj = network.plot_adjacency_matrix(
    G,
    title="Adjacency Matrix",
)
fig_adj.write_html("figures/adjacency_matrix.html")
print("✅ 邻接矩阵热图已保存")

# %% [markdown]
# ## 7. 数据导入导出

# %%
# 7.1 网络转 DataFrame
print("数据导入导出")

df_edges = network.network_to_polars(G)
print("边列表 DataFrame:")
print(df_edges.head())

# %%
# 7.2 保存网络
network.save_network(G, "outputs/karate_club.csv")
network.save_network(G, "outputs/karate_club.gml")
print("✅ 网络已保存为 CSV 和 GML 格式")

# %%
# 7.3 加载网络
G_loaded = network.load_network("outputs/karate_club.csv")
print(f"加载的网络: {G_loaded.number_of_nodes()} 节点, {G_loaded.number_of_edges()} 边")

# %%
# 7.4 邻接矩阵转换
adj_matrix, nodes = network.network_to_adjacency(G)
print(f"邻接矩阵形状: {adj_matrix.shape}")

# 从邻接矩阵创建网络
G_from_adj = network.adjacency_to_network(adj_matrix, node_labels=nodes)
print(f"从邻接矩阵重建: {G_from_adj.number_of_nodes()} 节点")

# %% [markdown]
# ## 8. 高级功能

# %%
# 8.1 查找桥边和关节点
print("结构分析")

bridges = analyzer.find_bridges()
print(f"桥边数量: {len(bridges)}")

articulation_points = analyzer.find_articulation_points()
print(f"关节点: {articulation_points}")

# %%
# 8.2 k-核分析
k_core = analyzer.get_k_core(k=4)
print(f"4-核子图: {k_core.number_of_nodes()} 节点, {k_core.number_of_edges()} 边")

# %%
# 8.3 最短路径
path = analyzer.get_shortest_path(0, 33)
print(f"节点 0 到 33 的最短路径: {path}")
