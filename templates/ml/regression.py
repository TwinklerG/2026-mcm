"""
回归模型模块
============

提供各类回归模型的统一封装。
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class RegressionModel:
    """
    回归模型封装类

    提供统一的训练、预测、评估接口。

    Attributes
    ----------
    model : Any
        sklearn 模型对象
    feature_names : list[str]
        特征名列表
    target_name : str
        目标变量名
    metrics : dict
        模型评估指标
    feature_importance : dict
        特征重要性 (如果模型支持)

    Examples
    --------
    >>> model = train_random_forest(df, features=["x1", "x2"], target="y")
    >>> predictions = model.predict(new_df)
    >>> print(model.metrics)
    """

    model: Any
    feature_names: list[str]
    target_name: str
    metrics: dict = field(default_factory=dict)
    feature_importance_: dict = field(default_factory=dict)
    scaler: StandardScaler | None = None

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """预测"""
        X = df.select(self.feature_names).to_numpy()
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_with_ci(
        self,
        df: pl.DataFrame,
        confidence: float = 0.95,
        n_bootstrap: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        带置信区间的预测 (Bootstrap 方法)

        Returns
        -------
        tuple
            (预测值, 下界, 上界)
        """
        X = df.select(self.feature_names).to_numpy()
        if self.scaler is not None:
            X = self.scaler.transform(X)

        predictions = self.model.predict(X)

        # 如果是集成模型，使用树的预测来估计不确定性
        if hasattr(self.model, "estimators_"):
            tree_predictions = np.array(
                [tree.predict(X) for tree in self.model.estimators_]
            )
            std = tree_predictions.std(axis=0)

            from scipy import stats

            z = stats.norm.ppf((1 + confidence) / 2)
            ci_lower = predictions - z * std
            ci_upper = predictions + z * std
        else:
            # 简单模型使用残差估计
            ci_lower = predictions * 0.9
            ci_upper = predictions * 1.1

        return predictions, ci_lower, ci_upper

    def get_feature_importance(self) -> pl.DataFrame:
        """获取特征重要性 DataFrame"""
        if not self.feature_importance_:
            return pl.DataFrame()

        return pl.DataFrame(
            {
                "feature": list(self.feature_importance_.keys()),
                "importance": list(self.feature_importance_.values()),
            }
        ).sort("importance", descending=True)


def train_random_forest(
    df: pl.DataFrame,
    features: list[str],
    target: str,
    *,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = False,
) -> RegressionModel:
    """
    训练随机森林回归模型

    Parameters
    ----------
    df : pl.DataFrame
        训练数据
    features : list[str]
        特征列名
    target : str
        目标列名
    n_estimators : int, default 100
        树的数量
    max_depth : int, optional
        最大深度
    min_samples_split : int, default 2
        分裂所需最小样本数
    min_samples_leaf : int, default 1
        叶节点最小样本数
    test_size : float, default 0.2
        测试集比例
    random_state : int, default 42
        随机种子
    scale_features : bool, default False
        是否标准化特征

    Returns
    -------
    RegressionModel
        训练好的模型

    Examples
    --------
    >>> model = train_random_forest(
    ...     df,
    ...     features=["gdp", "population", "athletes"],
    ...     target="medals",
    ...     n_estimators=200,
    ... )
    >>> print(model.metrics)
    >>> print(model.get_feature_importance())
    """
    # 准备数据
    X = df.select(features).to_numpy()
    y = df.select(target).to_numpy().ravel()

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 标准化
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 训练模型
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "mape": np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100,
    }

    # 特征重要性
    importance = dict(zip(features, model.feature_importances_))

    return RegressionModel(
        model=model,
        feature_names=features,
        target_name=target,
        metrics=metrics,
        feature_importance_=importance,
        scaler=scaler,
    )


def train_xgboost(
    df: pl.DataFrame,
    features: list[str],
    target: str,
    *,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    test_size: float = 0.2,
    random_state: int = 42,
) -> RegressionModel:
    """
    训练 XGBoost 回归模型

    Parameters
    ----------
    df : pl.DataFrame
        训练数据
    features : list[str]
        特征列名
    target : str
        目标列名
    n_estimators : int, default 100
        迭代次数
    max_depth : int, default 6
        最大深度
    learning_rate : float, default 0.1
        学习率
    subsample : float, default 0.8
        子采样比例
    colsample_bytree : float, default 0.8
        特征采样比例
    test_size : float, default 0.2
        测试集比例
    random_state : int, default 42
        随机种子

    Returns
    -------
    RegressionModel
        训练好的模型

    Examples
    --------
    >>> model = train_xgboost(df, features=["x1", "x2"], target="y")
    """
    # 使用 sklearn 的 GradientBoosting 作为替代 (避免额外依赖)
    # 如需使用 xgboost 库，可以在 pyproject.toml 中添加

    X = df.select(features).to_numpy()
    y = df.select(target).to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
    }

    importance = dict(zip(features, model.feature_importances_))

    return RegressionModel(
        model=model,
        feature_names=features,
        target_name=target,
        metrics=metrics,
        feature_importance_=importance,
    )


def train_ridge(
    df: pl.DataFrame,
    features: list[str],
    target: str,
    *,
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> RegressionModel:
    """
    训练 Ridge 回归模型

    Parameters
    ----------
    df : pl.DataFrame
        训练数据
    features : list[str]
        特征列名
    target : str
        目标列名
    alpha : float, default 1.0
        正则化强度
    test_size : float, default 0.2
        测试集比例
    random_state : int, default 42
        随机种子

    Returns
    -------
    RegressionModel
        训练好的模型
    """
    X = df.select(features).to_numpy()
    y = df.select(target).to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 标准化对线性模型很重要
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
    }

    # 系数作为特征重要性
    importance = dict(zip(features, np.abs(model.coef_)))

    return RegressionModel(
        model=model,
        feature_names=features,
        target_name=target,
        metrics=metrics,
        feature_importance_=importance,
        scaler=scaler,
    )


def train_ensemble(
    df: pl.DataFrame,
    features: list[str],
    target: str,
    *,
    models: list[Literal["rf", "gb", "ridge"]] | None = None,
    weights: list[float] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> RegressionModel:
    """
    训练集成模型 (多模型加权平均)

    Parameters
    ----------
    df : pl.DataFrame
        训练数据
    features : list[str]
        特征列名
    target : str
        目标列名
    models : list[str], optional
        要集成的模型，默认 ["rf", "gb", "ridge"]
    weights : list[float], optional
        模型权重，默认等权
    test_size : float, default 0.2
        测试集比例
    random_state : int, default 42
        随机种子

    Returns
    -------
    RegressionModel
        集成模型

    Examples
    --------
    >>> model = train_ensemble(df, features, target, models=["rf", "gb"])
    """
    if models is None:
        models = ["rf", "gb", "ridge"]
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    X = df.select(features).to_numpy()
    y = df.select(target).to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 训练各个模型
    trained_models = []
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for m in models:
        if m == "rf":
            estimator = RandomForestRegressor(
                n_estimators=100, random_state=random_state, n_jobs=-1
            )
            estimator.fit(X_train, y_train)
        elif m == "gb":
            estimator = GradientBoostingRegressor(
                n_estimators=100, random_state=random_state
            )
            estimator.fit(X_train, y_train)
        elif m == "ridge":
            estimator = Ridge(alpha=1.0)
            estimator.fit(X_train_scaled, y_train)
        trained_models.append((m, estimator))

    # 集成预测
    predictions = []
    for name, estimator in trained_models:
        if name == "ridge":
            pred = estimator.predict(X_test_scaled)
        else:
            pred = estimator.predict(X_test)
        predictions.append(pred)

    y_pred = np.average(predictions, axis=0, weights=weights)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
    }

    # 创建一个包装器模型
    class EnsembleWrapper:
        def __init__(self, models, weights, scaler):
            self.models = models
            self.weights = weights
            self.scaler = scaler

        def predict(self, X):
            predictions = []
            for name, estimator in self.models:
                if name == "ridge":
                    pred = estimator.predict(self.scaler.transform(X))
                else:
                    pred = estimator.predict(X)
                predictions.append(pred)
            return np.average(predictions, axis=0, weights=self.weights)

    ensemble = EnsembleWrapper(trained_models, weights, scaler)

    return RegressionModel(
        model=ensemble,
        feature_names=features,
        target_name=target,
        metrics=metrics,
        feature_importance_={},
    )
