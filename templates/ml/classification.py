"""
分类模型模块
============

提供各类分类模型的统一封装。
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class ClassificationModel:
    """
    分类模型封装类

    Attributes
    ----------
    model : Any
        sklearn 模型对象
    feature_names : list[str]
        特征名列表
    target_name : str
        目标变量名
    classes : list
        类别列表
    metrics : dict
        模型评估指标
    feature_importance : dict
        特征重要性
    """

    model: Any
    feature_names: list[str]
    target_name: str
    classes: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    feature_importance_: dict = field(default_factory=dict)
    scaler: StandardScaler | None = None

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """预测类别"""
        X = df.select(self.feature_names).to_numpy()
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """预测概率"""
        X = df.select(self.feature_names).to_numpy()
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            # SVM with probability=False
            return self.model.decision_function(X)

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


def train_classifier(
    df: pl.DataFrame,
    features: list[str],
    target: str,
    *,
    model_type: Literal["rf", "gb", "svm", "logistic"] = "rf",
    test_size: float = 0.2,
    random_state: int = 42,
    **model_params,
) -> ClassificationModel:
    """
    训练分类模型

    Parameters
    ----------
    df : pl.DataFrame
        训练数据
    features : list[str]
        特征列名
    target : str
        目标列名
    model_type : str, default "rf"
        模型类型: "rf", "gb", "svm", "logistic"
    test_size : float, default 0.2
        测试集比例
    random_state : int, default 42
        随机种子
    **model_params
        传递给模型的额外参数

    Returns
    -------
    ClassificationModel
        训练好的模型

    Examples
    --------
    >>> model = train_classifier(df, features, target, model_type="rf")
    >>> predictions = model.predict(test_df)
    >>> probabilities = model.predict_proba(test_df)
    """
    X = df.select(features).to_numpy()
    y = df.select(target).to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 对某些模型需要标准化
    scaler = None
    if model_type in ["svm", "logistic"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 创建模型
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", None),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "gb":
        model = GradientBoostingClassifier(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", 3),
            learning_rate=model_params.get("learning_rate", 0.1),
            random_state=random_state,
        )
    elif model_type == "svm":
        model = SVC(
            C=model_params.get("C", 1.0),
            kernel=model_params.get("kernel", "rbf"),
            probability=True,
            random_state=random_state,
        )
    elif model_type == "logistic":
        model = LogisticRegression(
            C=model_params.get("C", 1.0),
            max_iter=model_params.get("max_iter", 1000),
            random_state=random_state,
        )

    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    # 判断是否二分类
    n_classes = len(np.unique(y))
    average = "binary" if n_classes == 2 else "weighted"

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
    }

    # 特征重要性
    importance = {}
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(features, model.feature_importances_))
    elif hasattr(model, "coef_"):
        if model.coef_.ndim == 1:
            importance = dict(zip(features, np.abs(model.coef_)))
        else:
            importance = dict(zip(features, np.abs(model.coef_).mean(axis=0)))

    return ClassificationModel(
        model=model,
        feature_names=features,
        target_name=target,
        classes=list(np.unique(y)),
        metrics=metrics,
        feature_importance_=importance,
        scaler=scaler,
    )


def train_logistic(
    df: pl.DataFrame,
    features: list[str],
    target: str,
    *,
    C: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ClassificationModel:
    """
    训练 Logistic 回归 (适用于二分类概率预测)

    Parameters
    ----------
    df : pl.DataFrame
        训练数据
    features : list[str]
        特征列名
    target : str
        目标列名 (0/1 二分类)
    C : float, default 1.0
        正则化强度的倒数
    test_size : float, default 0.2
        测试集比例
    random_state : int, default 42
        随机种子

    Returns
    -------
    ClassificationModel
        训练好的模型

    Examples
    --------
    >>> # 预测首次获奖概率
    >>> model = train_logistic(df, features=["gdp", "population"], target="first_medal")
    >>> proba = model.predict_proba(test_df)[:, 1]  # 获取正类概率
    """
    return train_classifier(
        df,
        features,
        target,
        model_type="logistic",
        C=C,
        test_size=test_size,
        random_state=random_state,
    )
