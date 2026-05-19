"""
模型评估模块
============

提供交叉验证、特征重要性、评估指标等功能。
"""

from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, cross_val_score


def cross_validate(
    model: Any,
    df: pl.DataFrame,
    features: list[str],
    target: str,
    *,
    cv: int = 5,
    scoring: str | None = None,
    return_train_score: bool = False,
) -> dict:
    """
    交叉验证

    Parameters
    ----------
    model : Any
        sklearn 模型对象
    df : pl.DataFrame
        数据框
    features : list[str]
        特征列名
    target : str
        目标列名
    cv : int, default 5
        折数
    scoring : str, optional
        评分方式，默认回归用 "r2"，分类用 "accuracy"
    return_train_score : bool, default False
        是否返回训练分数

    Returns
    -------
    dict
        交叉验证结果

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> model = RandomForestRegressor()
    >>> results = cross_validate(model, df, features, target, cv=5)
    >>> print(f"Mean R²: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
    """
    X = df.select(features).to_numpy()
    y = df.select(target).to_numpy().ravel()

    # 自动检测任务类型
    if scoring is None:
        n_unique = len(np.unique(y))
        if n_unique <= 20 and np.all(y == y.astype(int)):
            scoring = "accuracy"
        else:
            scoring = "r2"

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)

    results = {
        "scores": scores.tolist(),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "cv_folds": cv,
        "scoring": scoring,
    }

    if return_train_score:
        from sklearn.model_selection import cross_validate as sklearn_cv

        cv_results = sklearn_cv(
            model, X, y, cv=kf, scoring=scoring, return_train_score=True
        )
        results["train_scores"] = cv_results["train_score"].tolist()
        results["mean_train_score"] = float(cv_results["train_score"].mean())

    return results


def feature_importance(
    model: Any,
    feature_names: list[str],
    *,
    method: Literal["builtin", "permutation"] = "builtin",
    df: pl.DataFrame | None = None,
    target: str | None = None,
    n_repeats: int = 10,
) -> pl.DataFrame:
    """
    计算特征重要性

    Parameters
    ----------
    model : Any
        训练好的模型
    feature_names : list[str]
        特征名列表
    method : str, default "builtin"
        方法: "builtin" 使用模型内置, "permutation" 使用置换重要性
    df : pl.DataFrame, optional
        置换重要性需要的数据
    target : str, optional
        置换重要性需要的目标列
    n_repeats : int, default 10
        置换重复次数

    Returns
    -------
    pl.DataFrame
        特征重要性表 (按重要性降序)

    Examples
    --------
    >>> importance = feature_importance(model.model, model.feature_names)
    >>> print(importance.head(10))
    """
    if method == "builtin":
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            if model.coef_.ndim == 1:
                importances = np.abs(model.coef_)
            else:
                importances = np.abs(model.coef_).mean(axis=0)
        else:
            raise ValueError("Model does not have feature_importances_ or coef_")

        return (
            pl.DataFrame(
                {
                    "feature": feature_names,
                    "importance": importances,
                }
            )
            .sort("importance", descending=True)
            .with_columns(
                (pl.col("importance") / pl.col("importance").sum() * 100).alias(
                    "importance_pct"
                )
            )
        )

    elif method == "permutation":
        if df is None or target is None:
            raise ValueError("df and target are required for permutation importance")

        from sklearn.inspection import permutation_importance

        X = df.select(feature_names).to_numpy()
        y = df.select(target).to_numpy().ravel()

        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=42
        )

        return pl.DataFrame(
            {
                "feature": feature_names,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort("importance", descending=True)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    include_mape: bool = True,
) -> dict:
    """
    计算回归评估指标

    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
    include_mape : bool, default True
        是否包含 MAPE

    Returns
    -------
    dict
        评估指标字典

    Examples
    --------
    >>> metrics = regression_metrics(y_test, y_pred)
    >>> print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
    """
    metrics = {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
    }

    if include_mape:
        # 避免除零
        mask = y_true != 0
        if mask.any():
            metrics["mape"] = (
                np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            )
        else:
            metrics["mape"] = np.nan

    # 额外指标
    metrics["explained_variance"] = 1 - np.var(y_true - y_pred) / np.var(y_true)

    return metrics


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list | None = None,
    average: Literal["binary", "micro", "macro", "weighted"] = "weighted",
) -> dict:
    """
    计算分类评估指标

    Parameters
    ----------
    y_true : np.ndarray
        真实标签
    y_pred : np.ndarray
        预测标签
    labels : list, optional
        类别标签
    average : str, default "weighted"
        多分类平均方式

    Returns
    -------
    dict
        评估指标字典

    Examples
    --------
    >>> metrics = classification_metrics(y_test, y_pred)
    >>> print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    """
    # 判断是否二分类
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    avg = "binary" if n_classes == 2 else average

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
    }

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm

    # 分类报告
    metrics["classification_report"] = classification_report(
        y_true, y_pred, labels=labels, zero_division=0
    )

    return metrics


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric_func: callable = r2_score,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """
    使用 Bootstrap 计算评估指标的置信区间

    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
    metric_func : callable, default r2_score
        评估指标函数
    n_bootstrap : int, default 1000
        Bootstrap 次数
    confidence : float, default 0.95
        置信水平
    random_state : int, default 42
        随机种子

    Returns
    -------
    tuple[float, float, float]
        (指标值, 下界, 上界)

    Examples
    --------
    >>> r2, lower, upper = bootstrap_confidence_interval(y_test, y_pred)
    >>> print(f"R² = {r2:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
    """
    np.random.seed(random_state)

    n = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_scores, alpha * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)
    point_estimate = metric_func(y_true, y_pred)

    return point_estimate, lower, upper
