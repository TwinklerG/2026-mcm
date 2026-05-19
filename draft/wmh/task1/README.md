# 任务 1：基于贝叶斯 MCMC 的粉丝投票估计

## 模型性能总结

### 核心验证指标

| 指标                      | 数值            | 说明                                             |
| --------------------------- | ---------------- | ------------------------------------------------------- |
| **重构准确率** | 95.02% (248/261) | 估计的最低得分与实际淘汰相符       |
| **后验一致性**   | 89.97% (mean)    | MCMC 样本中产生正确淘汰的比例 |
| **约束满足率** | 99.23% (259/261) | 粉丝分布满足淘汰约束       |
| **Bottom-2 Recall** | 99.23% | S28+ 规则下的软性验证指标 |
| **冠军预测准确率** | 100% (34/34) | 决赛冠军预测准确 |
| **Kendall's τ** | 0.994 | 决赛排名相关系数 |
| **异常检测**      | 2                | 模型的异常检测能力                    |

### 规则变化影响（S28+ Judges' Save）

| 指标                | S1-S27 | S28+   | 说明                         |
| ------------------- | ------ | ------ | ---------------------------- |
| **Strict Accuracy** | 100%   | 76.79% | 精确预测被淘汰者             |
| **Bottom-2 Recall** | 100%   | 96.43% | 被淘汰者在预测的 Bottom 2 中 |

> **核心发现**：S28+ 后 Strict Accuracy 下降并非模型失败，而是反映了"Judges' Save"规则引入后的结构性变化。Bottom-2 Recall 依然保持高水平，说明模型准确识别了危险区选手。

### MCMC 诊断

| 指标                          | 数值 | 目标              |
| ------------------------------- | ----- | ------------------- |
| **平均接受率**        | 34.6% | ~35% (最优 ~23%) |
| **平均 ESS**                    | 191   | >100                |
| **最小 ESS**                     | 9     | >50                |
| **95% 可信区间宽度** | 0.097 | -                   |
| **低 ESS 赛季** | S14, S21 | 可能为多峰分布 |

### 跨赛季泛化能力（稳定性分析）

| 数据集       | 准确率 | 描述        |
| ------------- | -------- | ------------------ |
| Train (S1-20) | 100% | 历史赛季 |
| Test (S21+)   | 87.25% | 近期赛季     |
| Gap           | 12.75% | 结构性漂移 |

> **论文亮点**：模型精度的下降揭示了 DWTS 投票生态在第 20 季后发生了**结构性漂移 (Structural Drift)**，这可能与社交媒体影响力增大、评委评分标准变化有关。

## 项目结构

```
task1_bayesian_mcmc/
├── model.py             # 统一导出接口
├── data_structures.py   # 数据结构和配置类
├── priors.py           # 数据驱动的先验估计
├── sampler.py          # 自适应 MCMC 采样器
├── validation.py       # 验证和评估函数
├── voting.py           # 投票方法和约束检查
├── run_inference.py     # 带缓存的推断脚本
├── visualize.py         # 论文级可视化
├── README.md            # 本文件
├── cache/               # 缓存的 MCMC 结果（自动生成）
├── outputs/             # 结果 JSON 和 CSV
│   ├── results.json              # 主结果文件
│   ├── fan_shares.csv            # 粉丝份额估计
│   ├── bottom2_accuracy.json     # Bottom-2 指标
│   ├── finale_ranking.json       # 决赛排名指标
│   ├── detailed_uncertainty.json # 详细不确定性分析
│   └── reconstruction_details.json
└── figures/             # 生成的可视化图表
    ├── comprehensive_summary.png      # 综合总结（单页）
    ├── stacked_accuracy_by_season.png # 堆叠柱状图（核心图）
    ├── structural_drift_analysis.png  # 结构漂移分析
    ├── finale_ranking_accuracy.png    # 决赛排名准确率
    ├── detailed_ci_distribution.png   # CI 宽度分布
    ├── controversial_vs_stable.png    # 争议型 vs 稳态型选手
    ├── validation_summary.png
    ├── reconstruction_by_season.png
    ├── fan_share_distribution.png
    ├── credible_intervals.png
    └── anomaly_analysis.png
```

## 快速开始

```bash
# 完整运行（首次运行，无缓存）
cd mcm
uv run python draft/lly/task1_bayesian_mcmc/run_inference.py

# 快速迭代（使用缓存）
uv run python draft/lly/task1_bayesian_mcmc/run_inference.py --skip-sensitivity --skip-stability

# 清除缓存并重新运行
uv run python draft/lly/task1_bayesian_mcmc/run_inference.py --clear-cache

# 生成可视化图表
uv run python draft/lly/task1_bayesian_mcmc/visualize.py
```

## 新增功能

### 1. Bottom-2 Recall 指标
针对 S28+ "Judges' Save" 规则，只要被淘汰者在模型预测的 Bottom 2 中，即视为成功。

### 2. 决赛排名准确率
- **Winner Accuracy**: 冠军预测准确率
- **Top-3 Position Accuracy**: 前三名位置准确率
- **Kendall's τ**: 排名相关系数

### 3. 详细不确定性分析
- 按选手/周次的 CI 宽度分布
- 识别"争议型选手"（高不确定性）和"稳态型选手"（低不确定性）
- 按行业、淘汰状态、周位置分类分析

### 4. 低 ESS 自适应重采样
自动检测低 ESS 赛季并尝试增强采样。

### 5. 可视化升级
- **堆叠柱状图**: Strict Accuracy + Bottom-2 Only
- **结构漂移分析图**: 展示规则变化的影响
- **综合总结图**: 单页展示所有关键指标

## 方法论

### 核心思路

1. **逆问题**：从观察到的淘汰结果推断隐含的粉丝投票份额
2. **贝叶斯 MCMC**：带稀疏采样的自适应 Metropolis-Hastings 算法，优化 ESS
3. **异常检测**：无法解释的案例会被标记为异常
4. **数据驱动的先验**：基于历史数据的行业生存分析

### 验证哲学

**为什么不使用这些方法：**
- **Spearman 相关性**：不存在粉丝投票的真实值
- **传统交叉验证**：破坏了赛季的时间结构

**我们使用的方法：**
- **重构准确率**：估计的最低综合得分是否与实际淘汰相符？
- **后验一致性**：MCMC 样本中产生正确淘汰的比例
- **稳定性分析**：在 S1-20 上训练，在 S21+ 上测试（时间划分，非随机划分）
- **Bottom-2 Recall**：S28+ 规则下的软性验证（新增）
- **Finale Ranking**：决赛排名准确率和 Kendall's τ（新增）

### 关键设计决策

1. **稀疏采样（每 5 步）**：减少自相关，提高 ESS
2. **平滑权重 = 0.5**：从 1.2 降低以改善混合
3. **自适应提议标准差**：目标 ~35% 接受率
4. **缓存**：支持开发过程中的快速迭代

## 输出结果解释

### 重构准确率
这是**核心指标**。它测试模型是否能够解释为什么某个选手被淘汰：
- 95%+ = 优秀（模型几乎解释了所有淘汰）
- 80-95% = 良好（一些淘汰可能涉及特殊规则）
- <80% = 模型需要改进

### 后验一致性
显示每集的模型置信度：
- 高 (>80%)：模型对淘汰机制很有信心
- 中 (50-80%)：存在一些不确定性，竞争激烈
- 低 (<50%)：高度不确定，可能是异常

### 异常
这些**不是**模型失败。它们是以下情况：
- 标准粉丝投票无法解释该淘汰
- 可能应用了特殊规则（免疫、团队舞蹈）
- 可能存在数据错误

## 学术定位

这**不是**一个预测模型。它是一个**贝叶斯推断 + 异常检测**框架：
- 高约束满足率验证了 Plackett-Luce 假设
- 检测到的异常是一个**功能**，而非 bug
- 可信区间量化了估计不确定性

## 术语注释

- 使用 **可信区间 (Credible Interval)**（贝叶斯），而非置信区间 (Confidence Interval)（频率）
- 使用 **重构准确率 (Reconstruction Accuracy)**，而非预测准确率
- 使用 **异常检测 (Anomaly Detection)**，而非“模型失败”

## 配置

默认 MCMC 配置：
```python
MCMCConfig(
    n_warmup=3000,
    n_samples=10000,
    thinning=5,
    initial_proposal_std=0.15,
    target_acceptance=0.35,
    adapt_interval=50,
    smoothness_weight=0.5,
    random_seed=42,
)
```

### 缓存机制

**缓存键结构**：`cache_s{season}_{config_hash}_{prior_hash}.pkl`

- `config_hash`：MCMC 配置的哈希值
- `prior_hash`：先验分布的哈希值（区分数据驱动 vs 均匀先验）

**性能优化**：
- **首次运行**（建立缓存）：≈32分钟（3次完整MCMC）
- **后续运行**（读取缓存）：≈20秒（加速 96倍）
- **敏感性分析**：数据驱动先验复用主流程缓存，均匀先验使用独立缓存

**缓存失效条件**：
- 更改 MCMC 配置参数
- 更改先验分布设置
- 数据文件更新

**手动清空缓存**：
```bash
rm -rf task1_bayesian_mcmc/cache
```
