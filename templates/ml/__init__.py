"""
机器学习模块
============

提供预测、分类、聚类等机器学习模型的封装。

主要功能
--------
- 预测模型: RandomForest, XGBoost, Ridge, Lasso
- 分类模型: RandomForest, Logistic, SVM
- 聚类模型: KMeans, DBSCAN, 层次聚类
- 模型评估: 交叉验证, 特征重要性, SHAP分析
"""

from .classification import (
    ClassificationModel,
    train_classifier,
    train_logistic,
)
from .clustering import (
    dbscan_cluster,
    hierarchical_cluster,
    kmeans_cluster,
)
from .evaluation import (
    classification_metrics,
    cross_validate,
    feature_importance,
    regression_metrics,
)
from .regression import (
    RegressionModel,
    train_ensemble,
    train_random_forest,
    train_ridge,
    train_xgboost,
)

__all__ = [
    # 回归
    "RegressionModel",
    "train_random_forest",
    "train_xgboost",
    "train_ridge",
    "train_ensemble",
    # 分类
    "ClassificationModel",
    "train_classifier",
    "train_logistic",
    # 聚类
    "kmeans_cluster",
    "hierarchical_cluster",
    "dbscan_cluster",
    # 评估
    "cross_validate",
    "feature_importance",
    "regression_metrics",
    "classification_metrics",
]
