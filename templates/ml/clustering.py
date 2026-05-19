"""
聚类模型模块
============

提供 K-Means、层次聚类、DBSCAN 等聚类方法。
"""

from typing import Literal

import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def kmeans_cluster(
    df: pl.DataFrame,
    features: list[str],
    *,
    n_clusters: int = 3,
    standardize: bool = True,
    random_state: int = 42,
    find_optimal_k: bool = False,
    k_range: tuple[int, int] = (2, 10),
) -> tuple[pl.DataFrame, dict]:
    """
    K-Means 聚类

    Parameters
    ----------
    df : pl.DataFrame
        输入数据
    features : list[str]
        用于聚类的特征列
    n_clusters : int, default 3
        聚类数量
    standardize : bool, default True
        是否标准化特征
    random_state : int, default 42
        随机种子
    find_optimal_k : bool, default False
        是否自动寻找最优 K (使用肘部法则和轮廓系数)
    k_range : tuple[int, int], default (2, 10)
        搜索 K 的范围

    Returns
    -------
    tuple[pl.DataFrame, dict]
        (添加了聚类标签的数据框, 聚类信息字典)

    Examples
    --------
    >>> df_clustered, info = kmeans_cluster(df, ["x", "y"], n_clusters=4)
    >>> print(info["silhouette_score"])
    >>>
    >>> # 自动寻找最优 K
    >>> df_clustered, info = kmeans_cluster(df, features, find_optimal_k=True)
    """
    X = df.select(features).to_numpy()

    # 处理缺失值
    mask = ~np.isnan(X).any(axis=1)
    X_clean = X[mask]

    # 标准化
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
    else:
        X_scaled = X_clean

    # 寻找最优 K
    if find_optimal_k:
        k_values = range(k_range[0], k_range[1] + 1)
        inertias = []
        silhouettes = []

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X_scaled, labels))

        # 选择轮廓系数最高的 K
        n_clusters = k_values[np.argmax(silhouettes)]

    # 最终聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels_clean = kmeans.fit_predict(X_scaled)

    # 将标签映射回原始数据
    labels = np.full(len(df), -1, dtype=int)
    labels[mask] = labels_clean

    # 添加聚类标签
    result = df.with_columns(pl.Series("cluster", labels))

    # 计算评估指标
    info = {
        "n_clusters": n_clusters,
        "inertia": kmeans.inertia_,
        "silhouette_score": silhouette_score(X_scaled, labels_clean),
        "calinski_harabasz_score": calinski_harabasz_score(X_scaled, labels_clean),
        "cluster_centers": kmeans.cluster_centers_,
        "cluster_sizes": dict(zip(*np.unique(labels_clean, return_counts=True))),
    }

    if find_optimal_k:
        info["k_search"] = {
            "k_values": list(k_values),
            "inertias": inertias,
            "silhouettes": silhouettes,
        }

    return result, info


def hierarchical_cluster(
    df: pl.DataFrame,
    features: list[str],
    *,
    n_clusters: int = 3,
    linkage: Literal["ward", "complete", "average", "single"] = "ward",
    standardize: bool = True,
) -> tuple[pl.DataFrame, dict]:
    """
    层次聚类

    Parameters
    ----------
    df : pl.DataFrame
        输入数据
    features : list[str]
        用于聚类的特征列
    n_clusters : int, default 3
        聚类数量
    linkage : str, default "ward"
        链接方式: "ward", "complete", "average", "single"
    standardize : bool, default True
        是否标准化特征

    Returns
    -------
    tuple[pl.DataFrame, dict]
        (添加了聚类标签的数据框, 聚类信息字典)

    Examples
    --------
    >>> df_clustered, info = hierarchical_cluster(df, features, n_clusters=4)
    """
    X = df.select(features).to_numpy()

    mask = ~np.isnan(X).any(axis=1)
    X_clean = X[mask]

    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
    else:
        X_scaled = X_clean

    # 聚类
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels_clean = clustering.fit_predict(X_scaled)

    labels = np.full(len(df), -1, dtype=int)
    labels[mask] = labels_clean

    result = df.with_columns(pl.Series("cluster", labels))

    info = {
        "n_clusters": n_clusters,
        "linkage": linkage,
        "silhouette_score": silhouette_score(X_scaled, labels_clean),
        "cluster_sizes": dict(zip(*np.unique(labels_clean, return_counts=True))),
    }

    return result, info


def dbscan_cluster(
    df: pl.DataFrame,
    features: list[str],
    *,
    eps: float = 0.5,
    min_samples: int = 5,
    standardize: bool = True,
) -> tuple[pl.DataFrame, dict]:
    """
    DBSCAN 密度聚类 (可检测异常点)

    Parameters
    ----------
    df : pl.DataFrame
        输入数据
    features : list[str]
        用于聚类的特征列
    eps : float, default 0.5
        邻域半径
    min_samples : int, default 5
        核心点所需最小样本数
    standardize : bool, default True
        是否标准化特征

    Returns
    -------
    tuple[pl.DataFrame, dict]
        (添加了聚类标签的数据框, 聚类信息字典)
        标签为 -1 表示异常点

    Examples
    --------
    >>> df_clustered, info = dbscan_cluster(df, features, eps=0.3)
    >>> # 查看异常点
    >>> anomalies = df_clustered.filter(pl.col("cluster") == -1)
    """
    X = df.select(features).to_numpy()

    mask = ~np.isnan(X).any(axis=1)
    X_clean = X[mask]

    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
    else:
        X_scaled = X_clean

    # 聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels_clean = clustering.fit_predict(X_scaled)

    labels = np.full(len(df), -2, dtype=int)  # -2 表示缺失
    labels[mask] = labels_clean

    result = df.with_columns(pl.Series("cluster", labels))

    # 统计信息
    n_clusters = len(set(labels_clean)) - (1 if -1 in labels_clean else 0)
    n_noise = (labels_clean == -1).sum()

    info = {
        "n_clusters": n_clusters,
        "n_noise_points": int(n_noise),
        "noise_ratio": n_noise / len(labels_clean),
        "eps": eps,
        "min_samples": min_samples,
        "cluster_sizes": dict(
            zip(*np.unique(labels_clean[labels_clean >= 0], return_counts=True))
        ),
    }

    # 如果有足够的聚类，计算轮廓系数
    if n_clusters > 1:
        non_noise_mask = labels_clean >= 0
        if non_noise_mask.sum() > n_clusters:
            info["silhouette_score"] = silhouette_score(
                X_scaled[non_noise_mask], labels_clean[non_noise_mask]
            )

    return result, info
