# %% [markdown]
# # MCM 2026 Templates 模块验证
#
# 模块结构：
# - `data`：数据加载、预处理、特征工程
# - `viz`：可视化（Plotly 交互式 + Matplotlib 论文级）
# - `ml`：机器学习（回归、分类、聚类）
# - `stats`：统计分析（相关分析、假设检验、评价模型）
# - `utils`：工具（Typst 导出、格式化、计时）

# %%
# 导入所有模块
import time
from datetime import date, timedelta

import matplotlib
import numpy as np
import polars as pl

from templates import data, ml, stats, utils, viz
from templates.viz import mpl

matplotlib.use("Agg")

print("✅ 所有模块导入成功")

# %% [markdown]
# ## 1. 创建测试数据

# %%
# 设置随机种子
np.random.seed(42)

# 创建测试数据
n_samples = 200

# 生成时间序列测试数据
dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_samples)]

# 生成特征和目标变量
x1 = np.random.randn(n_samples) * 10 + 50
x2 = np.random.randn(n_samples) * 5 + 30
x3 = np.random.randn(n_samples) * 8 + 40
noise = np.random.randn(n_samples) * 2
y = 2 * x1 + 1.5 * x2 - 0.5 * x3 + noise + 10

# 添加一些缺失值
x1_list = x1.tolist()
for idx in np.random.choice(n_samples, 10, replace=False):
    x1_list[idx] = None

# 添加异常值
x2_list = x2.tolist()
for idx in np.random.choice(n_samples, 5, replace=False):
    x2_list[idx] = 100.0  # 异常值

# 创建分组变量
groups = np.random.choice(["A", "B", "C"], n_samples)
binary_group = np.random.choice(["Control", "Treatment"], n_samples)

# 创建 DataFrame
df = pl.DataFrame(
    {
        "date": dates,
        "x1": x1_list,
        "x2": x2_list,
        "x3": x3.tolist(),
        "y": y.tolist(),
        "group": groups.tolist(),
        "binary_group": binary_group.tolist(),
        "category": np.random.choice(["Cat1", "Cat2", "Cat3"], n_samples).tolist(),
    }
)

print(f"测试数据维度：{df.shape}")
print(f"x1 缺失值数：{df['x1'].null_count()}")
print("\n数据预览:")
df.head()

# %%
# 1.2 保存并加载数据
print("数据加载（load_csv / load_data）")
df.write_csv("outputs/test_data.csv")
df_loaded = data.load_csv("outputs/test_data.csv")
print(f"从 CSV 加载：{df_loaded.shape}")

# load_data 自动识别格式
df_auto = data.load_data("outputs/test_data.csv")
print(f"自动识别加载：{df_auto.shape}")

# %% [markdown]
# ## 2. 数据探索模块（data.explore）

# %%
# 2.1 数据描述统计
print("数据描述统计（describe_df）")
stats_df = data.describe_df(df)
stats_df

# %%
# 2.2 缺失值报告
print("缺失值报告（missing_report）")
missing = data.missing_report(df)
missing

# %%
# 2.3 异常值检测
print("异常值检测（detect_outliers） - IQR 方法")
outliers_iqr = data.detect_outliers(df, method="iqr")
outliers_iqr

# %%
# 2.4 相关系数矩阵
print("相关系数矩阵（correlation_matrix）")
corr_df = data.correlation_matrix(df, columns=["x1", "x2", "x3", "y"])
corr_df

# %% [markdown]
# ## 3. 数据预处理模块（data.preprocess）

# %%
# 3.1 处理缺失值
print("处理缺失值（handle_missing） - 中位数填充")
df_no_missing = data.handle_missing(df, strategy="median")
print(f"处理前缺失值：{df['x1'].null_count()}")
print(f"处理后缺失值：{df_no_missing['x1'].null_count()}")

# %%
# 3.2 处理异常值
print("处理异常值（handle_outliers） - IQR 裁剪")
df_no_outliers = data.handle_outliers(df_no_missing, method="iqr", action="clip")
print(f"x2 处理前范围：[{df['x2'].min():.2f}, {df['x2'].max():.2f}]")
print(
    f"x2 处理后范围：[{df_no_outliers['x2'].min():.2f}, {df_no_outliers['x2'].max():.2f}]"
)

# %%
# 3.3 标准化
print("标准化（standardize） - Z-score")
df_standardized, std_params = data.standardize(
    df_no_outliers, columns=["x1", "x2", "x3"]
)
print(
    f"x1 标准化后：mean={df_standardized['x1'].mean():.4f}, std={df_standardized['x1'].std():.4f}"
)

# %%
# 3.4 归一化
print("归一化（normalize） - MinMax")
df_normalized, norm_params = data.normalize(
    df_no_outliers, columns=["x1", "x2"], method="minmax"
)
print(
    f"x1 归一化后范围：[{df_normalized['x1'].min():.4f}, {df_normalized['x1'].max():.4f}]"
)

# %%
# 3.5 一站式预处理流水线
print("预处理流水线（preprocess_pipeline）")
df_clean, params = data.preprocess_pipeline(
    df,
    drop_high_missing=0.5,
    handle_missing_strategy="median",
    handle_outliers_method="iqr",
    outlier_action="clip",
)
print(f"清洗后数据维度：{df_clean.shape}")
print(f"处理参数：{list(params.keys())}")

# %% [markdown]
# ## 4. 特征工程模块（data.features）

# %%
# 4.1 时间特征
print("时间特征（create_time_features）")
df_time = data.create_time_features(
    df_clean, "date", features=["year", "month", "weekday", "is_weekend"]
)
print(f"新增列：{[c for c in df_time.columns if 'date_' in c]}")
df_time.select(
    [
        "date",
        "date_year",
        "date_month",
        "date_weekday",
        "date_is_weekend",
    ]
).head()

# %%
# 4.2 滞后特征
print("滞后特征（create_lag_features）")
df_lag = data.create_lag_features(
    df_clean, columns=["y"], lags=[1, 3, 7], sort_by="date"
)
print(f"新增列：{[c for c in df_lag.columns if 'lag' in c]}")
df_lag.select(["date", "y", "y_lag_1", "y_lag_3", "y_lag_7"]).head(10)

# %%
# 4.3 滚动特征
print("滚动特征（create_rolling_features）")
df_rolling = data.create_rolling_features(
    df_clean, columns=["y"], windows=[7], functions=["mean", "std"], sort_by="date"
)
rolling_cols = [c for c in df_rolling.columns if "rolling" in c]
print(f"新增列：{rolling_cols}")
df_rolling.select(["date", "y"] + rolling_cols).tail(10)

# %%
# 4.4 分类编码
print("分类编码（encode_categorical） - One-Hot")
df_encoded, enc_mapping = data.encode_categorical(
    df_clean, columns=["category"], method="onehot"
)
print(f"编码后新增列：{[c for c in df_encoded.columns if 'category_' in c]}")
df_encoded.select([c for c in df_encoded.columns if "category" in c]).head()

# %%
# 4.5 交互特征
print("交互特征（create_interaction_features）")
df_interact = data.create_interaction_features(
    df_clean, column_pairs=[("x1", "x2")], operations=["multiply", "divide"]
)
interact_cols = [c for c in df_interact.columns if "x1" in c and "x2" in c]
print(f"新增列：{interact_cols}")
df_interact.select(["x1", "x2"] + interact_cols).head()

# %% [markdown]
# ## 5. 可视化模块（viz）- Plotly 交互式图表

# %%
# 5.1 折线图
print("折线图（line_plot）")
fig = viz.line_plot(
    df_clean.sort("date").head(50),
    x="date",
    y="y",
    title="时间序列趋势",
    show_markers=True,
)
fig.show()

# %%
# 5.2 散点图
print("散点图（scatter_plot）")
fig = viz.scatter_plot(
    df_clean,
    x="x1",
    y="y",
    color="group",
    title="X1 vs Y 散点图",
    show_trendline=True,
)
fig.show()

# %%
# 5.3 柱状图
print("柱状图（bar_plot）")
# 创建分组统计数据
bar_data = df_clean.group_by("group").agg(pl.col("y").mean().alias("mean_y"))
fig = viz.bar_plot(
    bar_data,
    x="group",
    y="mean_y",
    title="各组平均 Y 值",
    show_values=True,
)
fig.show()

# %%
# 5.4 热力图
print("热力图（heatmap）")
corr_matrix = data.correlation_matrix(df_clean, columns=["x1", "x2", "x3", "y"])
fig = viz.heatmap(
    corr_matrix,
    title="相关系数矩阵",
    show_values=True,
)
fig.show()

# %%
# 5.5 箱线图
print("箱线图（box_plot）")
fig = viz.box_plot(
    df_clean,
    y="y",
    x="group",
    title="各组 Y 值分布",
    show_points=True,
)
fig.show()

# %%
# 5.6 直方图
print("直方图（histogram）")
fig = viz.histogram(
    df_clean,
    x="y",
    bins=30,
    title="Y 值分布",
    show_kde=True,
)
fig.show()

# %%
# 5.7 雷达图
print("雷达图（radar_plot）")
radar_data = pl.DataFrame(
    {
        "name": ["Model A", "Model B"],
        "Accuracy": [0.85, 0.78],
        "Precision": [0.82, 0.80],
        "Recall": [0.88, 0.75],
        "F1": [0.85, 0.77],
        "Speed": [0.90, 0.95],
    }
)
fig = viz.radar_plot(
    radar_data,
    categories=["Accuracy", "Precision", "Recall", "F1", "Speed"],
    name_col="name",
    title="模型性能对比",
)
fig.show()

# %%
# 5.8 子图网格
print("子图网格（subplot_grid）")
figs = [
    viz.scatter_plot(df_clean, x="x1", y="y", title="X1 vs Y"),
    viz.scatter_plot(df_clean, x="x2", y="y", title="X2 vs Y"),
    viz.histogram(df_clean, x="y", title="Y 分布"),
    viz.box_plot(df_clean, y="y", x="group", title="Y by Group"),
]
fig = viz.subplot_grid(figs, rows=2, cols=2, main_title="多图组合")
fig.show()

# %%
# 5.9 主题与配色
print("主题与配色（apply_theme / get_color_palette）")
colors = viz.get_color_palette("nature", n_colors=5)
print(f"Nature 配色：{colors}")

fig = viz.scatter_plot(df_clean, x="x1", y="y", title="自定义主题")
fig = viz.apply_theme(fig, theme="science")
fig.show()

# %% [markdown]
# ## 6. 可视化模块（viz.mpl）- Matplotlib 论文级图表

# %%
# 6.1 配置全局样式
print("配置 Matplotlib 样式（setup_style）")
mpl.setup_style(style="nature", font_size=11)
print("✅ Nature 风格已配置")

# %%
# 6.2 Matplotlib 散点图
print("Matplotlib 散点图（mpl.scatter_plot）")
fig = mpl.scatter_plot(
    df_clean,
    x="x1",
    y="y",
    title="X1 vs Y (Nature Style)",
    show_trendline=True,
    style="nature",
)

# %%
# 6.3 Matplotlib 热力图
print("Matplotlib 热力图（mpl.heatmap）")
fig = mpl.heatmap(
    corr_matrix,
    title="Correlation Matrix (Nature Style)",
    style="nature",
)

# %%
# 6.4 Matplotlib 柱状图
print("Matplotlib 柱状图（mpl.bar_plot）")
fig = mpl.bar_plot(
    bar_data,
    x="group",
    y="mean_y",
    title="Group Mean Y (Science Style)",
    style="science",
)

# %%
# 6.5 Matplotlib 直方图
print("Matplotlib 直方图（mpl.histogram）")
fig = mpl.histogram(
    df_clean,
    x="y",
    title="Y Distribution (Nature Style)",
    show_kde=True,
    style="nature",
)

# %%
# 6.6 Matplotlib 箱线图
print("Matplotlib 箱线图（mpl.box_plot）")
fig = mpl.box_plot(
    df_clean,
    y="y",
    x="group",
    title="Y by Group (Nature Style)",
    style="nature",
)

# %%
# 关闭所有图表
mpl.close_all()
print("✅ 已关闭所有 Matplotlib 图表")

# %% [markdown]
# ## 7. 机器学习模块（ml）- 回归

# %%
# 准备 ML 数据
features = ["x1", "x2", "x3"]
target = "y"

print("7.1 随机森林回归 (train_random_forest)")

rf_model = ml.train_random_forest(
    df_clean.drop_nulls(),
    features=features,
    target=target,
    n_estimators=100,
    test_size=0.2,
)
print(f"R² = {rf_model.metrics['r2']:.4f}")
print(f"RMSE = {rf_model.metrics['rmse']:.4f}")
print(f"MAE = {rf_model.metrics['mae']:.4f}")
print("\n特征重要性:")
rf_model.get_feature_importance()

# %%
# 7.2 Ridge 回归
print("Ridge 回归 (train_ridge)")

ridge_model = ml.train_ridge(
    df_clean.drop_nulls(),
    features=features,
    target=target,
    alpha=1.0,
)
print(f"R² = {ridge_model.metrics['r2']:.4f}")
print(f"RMSE = {ridge_model.metrics['rmse']:.4f}")

# %%
# 7.3 XGBoost 回归
print("XGBoost 回归 (train_xgboost)")

xgb_model = ml.train_xgboost(
    df_clean.drop_nulls(),
    features=features,
    target=target,
    n_estimators=100,
)
print(f"R² = {xgb_model.metrics['r2']:.4f}")
print(f"RMSE = {xgb_model.metrics['rmse']:.4f}")

# %%
# 7.4 集成模型
print("集成模型 (train_ensemble)")

ensemble_model = ml.train_ensemble(
    df_clean.drop_nulls(),
    features=features,
    target=target,
    models=["rf", "ridge"],
)
print(f"R² = {ensemble_model.metrics['r2']:.4f}")
print(f"RMSE = {ensemble_model.metrics['rmse']:.4f}")

# %%
# 7.5 模型预测
print("模型预测")

predictions = rf_model.predict(df_clean.drop_nulls().head(10))
actuals = df_clean.drop_nulls().head(10)["y"].to_numpy()

pred_df = pl.DataFrame(
    {
        "actual": actuals,
        "predicted": predictions,
        "error": actuals - predictions,
    }
)
pred_df

# %% [markdown]
# ## 8. 机器学习模块（ml）- 分类

# %%
# 8.1 逻辑回归分类
print("逻辑回归分类（train_logistic）")

# 创建二分类目标
df_cls = df_clean.drop_nulls().with_columns(
    (pl.col("y") > pl.col("y").median()).cast(pl.Int32).alias("label")
)

logistic_model = ml.train_logistic(
    df_cls,
    features=features,
    target="label",
)
print(f"Accuracy = {logistic_model.metrics['accuracy']:.4f}")
print(f"F1 = {logistic_model.metrics['f1']:.4f}")

# %%
# 8.2 随机森林分类
print("随机森林分类（train_classifier）")

rf_classifier = ml.train_classifier(
    df_cls,
    features=features,
    target="label",
    model_type="rf",
)
print(f"Accuracy = {rf_classifier.metrics['accuracy']:.4f}")
print(f"F1 = {rf_classifier.metrics['f1']:.4f}")

# %% [markdown]
# ## 9. 机器学习模块（ml）- 聚类

# %%
# 9.1 K-Means 聚类
print("K-Means 聚类（kmeans_cluster）")

df_clustered, kmeans_info = ml.kmeans_cluster(
    df_clean.drop_nulls(),
    features=features,
    n_clusters=3,
)
print(f"聚类数：{kmeans_info['n_clusters']}")
print(f"轮廓系数：{kmeans_info['silhouette_score']:.4f}")
print(f"各簇大小：{kmeans_info['cluster_sizes']}")

# %%
# 9.2 自动寻找最优 K
print("K-Means 自动寻优（find_optimal_k=True）")

df_opt, opt_info = ml.kmeans_cluster(
    df_clean.drop_nulls(),
    features=features,
    find_optimal_k=True,
    k_range=(2, 6),
)
print(f"最优 K：{opt_info['n_clusters']}")
print(f"轮廓系数：{opt_info['silhouette_score']:.4f}")

# %%
# 9.3 层次聚类
print("层次聚类（hierarchical_cluster）")

df_hier, hier_info = ml.hierarchical_cluster(
    df_clean.drop_nulls(),
    features=features,
    n_clusters=3,
    linkage="ward",
)
print(f"轮廓系数：{hier_info['silhouette_score']:.4f}")
print(f"各簇大小：{hier_info['cluster_sizes']}")

# %%
# 9.4 DBSCAN 聚类
print("DBSCAN 聚类（dbscan_cluster）")

df_dbscan, dbscan_info = ml.dbscan_cluster(
    df_clean.drop_nulls(),
    features=features,
    eps=0.8,
    min_samples=5,
)
print(f"聚类数：{dbscan_info['n_clusters']}")
print(f"噪声点数：{dbscan_info['n_noise_points']}")

# %% [markdown]
# ## 10. 机器学习模块（ml）- 模型评估

# %%
# 10.1 交叉验证
print("交叉验证（cross_validate）")

cv_results = ml.cross_validate(
    rf_model.model,
    df_clean.drop_nulls(),
    features=features,
    target=target,
    cv=5,
)
print(f"CV R²：{cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
print(f"各折得分：{[f'{s:.4f}' for s in cv_results['scores']]}")

# %%
# 10.2 特征重要性
print("特征重要性（feature_importance）")

importance_df = ml.feature_importance(rf_model.model, features)
importance_df

# %%
# 10.3 回归评估指标
print("回归评估指标（regression_metrics）")

y_true = df_clean.drop_nulls()["y"].to_numpy()
y_pred = rf_model.predict(df_clean.drop_nulls())
reg_metrics = ml.regression_metrics(y_true, y_pred)
print(f"R² = {reg_metrics['r2']:.4f}")
print(f"RMSE = {reg_metrics['rmse']:.4f}")
print(f"MAE = {reg_metrics['mae']:.4f}")
print(f"MAPE = {reg_metrics['mape']:.2f}%")

# %%
# 10.4 分类评估指标
print("分类评估指标（classification_metrics）")

y_true_cls = df_cls["label"].to_numpy()
y_pred_cls = logistic_model.predict(df_cls)
cls_metrics = ml.classification_metrics(y_true_cls, y_pred_cls)
print(f"Accuracy = {cls_metrics['accuracy']:.4f}")
print(f"Precision = {cls_metrics['precision']:.4f}")
print(f"Recall = {cls_metrics['recall']:.4f}")
print(f"F1 = {cls_metrics['f1']:.4f}")

# %% [markdown]
# ## 11. 统计分析模块（stats）- 相关分析

# %%
# 11.1 相关性检验
print("相关性检验（correlation_test）")

corr_result = stats.correlation_test(df_clean, "x1", "y", method="pearson")
print(f"方法：{corr_result['method']}")
print(f"相关系数：{corr_result['correlation']:.4f}")
print(f"P 值：{corr_result['p_value']:.4f}")
print(f"解释：{corr_result['interpretation']}")
print(f"显著：{'是' if corr_result['significant'] else '否'}")

# %%
# 11.2 Spearman 相关
print("Spearman 相关性检验（spearman_correlation_test）")

spearman_result = stats.correlation_test(df_clean, "x1", "y", method="spearman")
print(f"Spearman r = {spearman_result['correlation']:.4f}")
print(f"P 值：{spearman_result['p_value']:.4f}")

# %%
# 11.3 偏相关分析
print("偏相关分析（partial_correlation）")

partial_result = stats.partial_correlation(df_clean, "x1", "y", control=["x2", "x3"])
print(f"控制变量：{partial_result['control_variables']}")
print(f"偏相关系数：{partial_result['partial_correlation']:.4f}")
print(f"P 值：{partial_result['p_value']:.4f}")

# %%
# 11.4 相关性摘要
print("相关性摘要（correlation_summary）")

corr_summary = stats.correlation_summary(df_clean, columns=["x1", "x2", "x3", "y"])
corr_summary

# %% [markdown]
# ## 12. 统计分析模块（stats）- 假设检验

# %%
# 12.1 独立样本 t 检验
print("独立样本 t 检验（t_test）")

t_result = stats.t_test(df_clean, "y", group_by="binary_group")
print(f"t = {t_result['statistic']:.4f}")
print(f"p = {t_result['p_value']:.4f}")
print(f"Cohen's d = {t_result['cohens_d']:.4f}（{t_result['effect_size']}）")
print(f"显著：{'是' if t_result['significant'] else '否'}")
# 12.2 配对 t 检验
print("配对 t 检验（paired_t_test）")

# 添加配对数据
df_paired = df_clean.with_columns(
    (pl.col("y") + np.random.randn(len(df_clean)) * 5).alias("y_after")
)

paired_result = stats.paired_t_test(df_paired, "y", "y_after")
print(f"t = {paired_result['statistic']:.4f}")
print(f"p = {paired_result['p_value']:.4f}")
print(f"平均差异：{paired_result['mean_diff']:.4f}")

# %%
# 12.3 单因素方差分析（ANOVA）
print("单因素方差分析（anova）")

anova_result = stats.anova(df_clean, "y", group_by="group")
print(f"F = {anova_result['f_statistic']:.4f}")
print(f"p = {anova_result['p_value']:.4f}")
print(f"各组均值：{anova_result['group_means']}")

# %%
# 12.4 卡方检验
print("卡方检验（chi_square_test）")

chi2_result = stats.chi_square_test(df_clean, "group", "binary_group")
print(f"χ² = {chi2_result['chi2_statistic']:.4f}")
print(f"p = {chi2_result['p_value']:.4f}")
print(f"自由度 = {chi2_result['degrees_of_freedom']}")
print(f"Cramér's V = {chi2_result['cramers_v']:.4f}")

# %%
# 12.5 正态性检验
print("正态性检验（normality_test）")

normal_result = stats.normality_test(df_clean, "y")
print(f"统计量：{normal_result['statistic']:.4f}")
print(f"p = {normal_result['p_value']:.4f}")
print(f"是否正态：{'是' if normal_result['is_normal'] else '否'}")

# %% [markdown]
# ## 13. 统计分析模块（stats）- 评价模型

# %%
# 13.1 AHP 层次分析法
print("AHP 层次分析法（ahp_weight）")

# 判断矩阵
ahp_matrix = [[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]]

ahp_result = stats.ahp_weight(ahp_matrix, criteria_names=["指标A", "指标B", "指标C"])
print(f"权重：{ahp_result['weights']}")
print(f"CR = {ahp_result['CR']:.4f}")
print(f"一致性：{'通过' if ahp_result['is_consistent'] else '不通过'}")

# %%
# 13.2 熵权法
print("熵权法（entropy_weight）")

entropy_result = stats.entropy_weight(df_clean, columns=["x1", "x2", "x3"])
print(f"权重：{entropy_result['weights']}")
print(f"熵值：{entropy_result['entropy']}")

# %%
# 13.3 组合权重
print("组合权重（combine_weights）")

combined = stats.combine_weights(
    ahp_result["weights"],
    entropy_result["weights"],
    alpha=0.5,
)
print(f"组合权重：{combined}")

# %%
# 13.4 TOPSIS 综合评价
print("TOPSIS 综合评价（topsis）")

# 创建评价数据
eval_df = df_clean.drop_nulls().head(20).with_row_index("id")

topsis_result = stats.topsis(
    eval_df,
    columns=["x1", "x2", "x3"],
    weights=[0.4, 0.3, 0.3],
    beneficial=["x1", "x3"],
    cost=["x2"],
    id_column="id",
)

print("TOPSIS 评价结果（Top 10）：")
topsis_result.sort("topsis_rank").head(10).select(["id", "topsis_score", "topsis_rank"])

# %% [markdown]
# ## 14. 统计分析模块（stats）- 敏感性分析

# %%
# 14.1 单参数敏感性分析
print("单参数敏感性分析（sensitivity_analysis）")


# 定义模型函数
def model_func(params):
    return params["a"] * 2 + params["b"] * 3 + params["c"] ** 2


base_params = {"a": 1.0, "b": 2.0, "c": 3.0}

sensitivity_df = stats.sensitivity_analysis(
    model_func,
    base_params,
    param_to_analyze="a",
    variation_range=(-0.3, 0.3),
    n_steps=10,
)
sensitivity_df

# %%
# 14.2 多参数网格搜索
print("多参数网格搜索（parameter_sweep）")

sweep_df = stats.parameter_sweep(
    model_func,
    param_ranges={
        "a": (0.5, 1.5, 5),
        "b": (1.5, 2.5, 5),
        "c": (2.5, 3.5, 5),
    },
)
print(f"参数组合数：{len(sweep_df)}")
sweep_df.head(10)

# %%
# 14.3 龙卷风图数据
print("龙卷风图数据（tornado_diagram_data）")

tornado_df = stats.tornado_diagram_data(
    model_func,
    base_params,
    variation=0.2,
)
print("龙卷风图数据：")
tornado_df

# %% [markdown]
# ## 15. 工具模块（utils）- Typst 导出

# %%
# 15.1 Typst 表格
print("Typst 表格（to_typst_table）")

result_table = pl.DataFrame(
    {
        "Model": ["RF", "Ridge", "XGBoost"],
        "R²": [0.856, 0.812, 0.845],
        "RMSE": [2.34, 2.89, 2.45],
        "MAE": [1.78, 2.12, 1.89],
    }
)

typst_table = utils.to_typst_table(
    result_table,
    caption="模型评估结果",
    label="tab:results",
    precision=3,
)
print(typst_table)

# %%
# 15.2 Typst 图片引用
print("Typst 图片引用（to_typst_figure）")

typst_fig = utils.to_typst_figure(
    "./figures/scatter.png",
    caption="散点图分析",
    label="fig:scatter",
    width="80%",
)
print(typst_fig)

# %%
# 15.3 Typst 公式
print("Typst 公式（to_typst_equation）")

typst_eq = utils.to_typst_equation(
    "y = alpha x + beta",
    label="eq:linear",
)
print(typst_eq)

# %%
# 15.4 批量导出 Typst 资源
print("批量导出 Typst 资源（export_typst_assets）")

utils.export_typst_assets(
    tables={"results": (result_table, "模型评估结果")},
    figures={"scatter": ("./figures/scatter.png", "散点图分析")},
    output_file="outputs/assets.typ",
)
print("✅ 已导出到 outputs/assets.typ")

# %%
# 15.5 DataFrame 转 Typst 原始格式
print("DataFrame 转 Typst 原始格式（dataframe_to_typst_raw）")

typst_raw = utils.dataframe_to_typst_raw(result_table.head(3))
print(typst_raw)

# %% [markdown]
# ## 16. 工具模块（utils）- 格式化

# %%
# 16.1 数值格式化
print("数值格式化（format_number）")

print(f"Decimal: {utils.format_number(1234567.89, style='decimal')}")
print(f"Scientific: {utils.format_number(0.00123, style='scientific')}")
print(f"Compact: {utils.format_number(1234567.89, style='compact')}")
print(f"Percent: {utils.format_number(0.1234, style='percent')}")

# %%
# 16.2 P 值格式化
print("P 值格式化（format_pvalue）")

print(f"p=0.0001: {utils.format_pvalue(0.0001)}")
print(f"p=0.03: {utils.format_pvalue(0.03)}")
print(f"p=0.15: {utils.format_pvalue(0.15)}")

# %%
# 16.3 百分比格式化
print("百分比格式化（format_percentage）")

print(f"0.1234: {utils.format_percentage(0.1234)}")
print(f"0.8567: {utils.format_percentage(0.8567)}")

# %%
# 16.4 汇总表
print("创建摘要表格（create_summary_table）")

data = {
    "Model A": {"R²": 0.95, "RMSE": 1.23, "MAE": 0.98},
    "Model B": {"R²": 0.92, "RMSE": 1.45, "MAE": 1.15},
    "Model C": {"R²": 0.88, "RMSE": 1.67, "MAE": 1.32},
}
summary = utils.create_summary_table(data)
summary

# %% [markdown]
# ## 17. 工具模块（utils）- 计时

# %%
# 17.1 上下文管理器计时
print("上下文管理器计时（timer）")

with utils.timer("示例任务"):
    time.sleep(0.1)
    _ = [i**2 for i in range(10000)]

# %%
# 17.2 装饰器计时
print("装饰器计时（timed）")


@utils.timed
def example_func():
    time.sleep(0.05)
    return sum(i**2 for i in range(1000))


result = example_func()
print(f"计算结果: {result}")

# %%
# 17.3 Timer 类（with 上下文 + 分段计时）
print("Timer 类（with 上下文 + 分段计时）")

with utils.Timer() as t:
    time.sleep(0.05)
    t.lap("Step 1")
    time.sleep(0.05)
    t.lap("Step 2")

print(f"\n总时间：{t.total:.2f}s")
print(t.summary())

# %% [markdown]
# ## 18. 工具模块（utils）- 模型导出与加载

# %%
# 18.1 保存结果
print("保存结果 (save_results)")

results = {
    "metrics": rf_model.metrics,
    "features": features,
    "target": target,
}

utils.save_results(results, "rf_results", output_dir="outputs")
print("✅ 结果已保存到 outputs/rf_results.json")

# %%
# 18.2 导出模型
print("导出模型（export_model）")

utils.export_model(rf_model.model, "rf_model", output_dir="models")
print("✅ 模型已保存到 models/rf_model.pkl")

# %%
# 18.3 加载模型
print("加载模型（load_model）")

loaded_model = utils.load_model("rf_model", model_dir="models")
print(f"✅ 模型已加载：{type(loaded_model).__name__}")

# %% [markdown]
# ## 19. 可视化导出

# %%
# 19.1 保存 Plotly 图表
print("保存 Plotly 图表（viz.save_figure）")

fig = viz.scatter_plot(df_clean, x="x1", y="y", title="Test Scatter")
viz.save_figure(fig, "test_scatter", output_dir="figures", format="png")
print("✅ 图表已保存到 figures/test_scatter.png")

# %%
# 19.2 保存 Matplotlib 图表
print("保存 Matplotlib 图表（mpl.save_figure）")

fig = mpl.scatter_plot(df_clean, x="x1", y="y", title="Test Scatter MPL")
mpl.save_figure(fig, "test_scatter_mpl", output_dir="figures", formats=["png"])
print("✅ 图表已保存到 figures/test_scatter_mpl.png")
mpl.close_all()
