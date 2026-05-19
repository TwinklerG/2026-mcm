# Task 4: 渐进技术公平系统 (PTFS)

## 概述

PTFS (Progressive Technical Fairness System) 是一种新的 DWTS 投票组合系统，通过动态调整评委分与粉丝投票的权重，在保持观众参与度的同时提升技术公平性。

## 问题诊断

### 来自 Task 1 的发现
- **Judges' Save 失效**：90.5% 的情况下评委救了粉丝份额更高的选手
- 粉丝投票与评委评分存在系统性分歧

### 来自 Task 3 的发现
- **运动员悖论**：评委 β=-0.129（负），粉丝 β=+0.052（正）
- **职业舞伴效应**：对粉丝影响 (ICC=21.8%) 是评委 (ICC=9.5%) 的 2.3 倍
- **年龄偏差**：评委对年龄的惩罚是粉丝的 18 倍

## 模型设计

### 核心公式

$$S_{i,t} = w_J(t) \cdot \tilde{J}_{i,t} + (1 - w_J(t)) \cdot \tilde{F}_{i,t}$$

其中：
- $\tilde{J}_{i,t}$：归一化评委分
- $\tilde{F}_{i,t}$：估计的粉丝份额（来自 Task 1）
- $w_J(t)$：动态评委权重

### 动态权重

$$w_J(t) = w_J^{\text{start}} + (w_J^{\text{end}} - w_J^{\text{start}}) \cdot \frac{t-1}{T-1}$$

**最优参数**：
- $w_J^{\text{start}} = 0.45$（初始评委权重 45%）
- $w_J^{\text{end}} = 0.80$（最终评委权重 80%）

**设计理念**：
- **早期**：更重视粉丝投票，维持参与度和悬念
- **后期**：更重视评委评分，确保技术能力得到认可

---

## 评价指标

### 技术公平性
| 指标                   | 定义                       | 方向 |
| ---------------------- | -------------------------- | ---- |
| Kendall's τ            | 累积技术分与最终排名相关性 | ↑    |
| Tech Top3 → Final Top3 | 技术前三进入最终前三       | ↑    |
| Tech Lowest Eliminated | 技术最低分被淘汰比例       | ↑    |
| Mean Rank Deviation    | 平均排名偏差               | ↓    |

### 观众参与度
| 指标                  | 定义                    | 方向 |
| --------------------- | ----------------------- | ---- |
| Tech Top50 Eliminated | 技术前50%被淘汰（爆冷） | 适中 |
| Close Call Rate       | 前两名差距<2%（悬念）   | ↑    |

### 鲁棒性
| 指标                 | 定义               | 方向 |
| -------------------- | ------------------ | ---- |
| τ Std Across Seasons | 跨赛季 τ 标准差    | ↓    |
| Consistent Seasons   | τ > 0.5 的赛季比例 | ↑    |

---

## 实验结果

### 系统对比（34 赛季）

| 指标                   | Rank-based | Percentage | **PTFS**  | 最优 |
| ---------------------- | ---------- | ---------- | --------- | ---- |
| Kendall's τ            | 0.674      | 0.727      | **0.749** | PTFS |
| Spearman ρ             | 0.815      | 0.860      | **0.876** | PTFS |
| Tech Top3 → Final Top3 | 59.8%      | 71.6%      | **72.5%** | PTFS |
| Mean Rank Deviation    | 1.61       | 1.35       | **1.26**  | PTFS |
| Tech Lowest Eliminated | 41.9%      | 47.2%      | **52.8%** | PTFS |
| Close Call Rate        | 30.2%      | 66.1%      | **79.1%** | PTFS |
| Consistent Seasons     | 91.2%      | 94.1%      | **97.1%** | PTFS |

### 统计显著性

| 对比               | Δτ     | p-value   | 显著 |
| ------------------ | ------ | --------- | ---- |
| PTFS vs Percentage | +0.022 | **0.031** | ✓    |
| PTFS vs Rank-based | +0.075 | <0.01     | ✓    |

### 权衡分析

PTFS 实现了多维度的平衡改进：

| 维度       | PTFS vs Percentage  | 解读                 |
| ---------- | ------------------- | -------------------- |
| 技术公平性 | **+0.022 τ**        | 显著提升（p=0.031）  |
| 一致性     | **+2.9%**           | 97%赛季达到τ>0.5     |
| 悬念       | **+13% Close Call** | 更多激烈淘汰         |
| 公平淘汰   | **+5.6%**           | 技术最低分更易被淘汰 |

---

## 输出文件

| 文件                        | 内容           |
| --------------------------- | -------------- |
| `optimal_params.json`       | 最优参数       |
| `system_comparison.json`    | 系统对比       |
| `detailed_metrics.json`     | 详细指标       |
| `season_details.json`       | 34赛季逐一数据 |
| `sensitivity_analysis.json` | 参数敏感性     |

## 图表

| 图表                           | 说明                  |
| ------------------------------ | --------------------- |
| `tau_by_season.png`            | 34赛季τ对比（核心图） |
| `improvement_distribution.png` | 改进分布统计          |
| `metrics_comparison.png`       | 关键指标条形图        |
| `dynamic_weights.png`          | 动态权重曲线          |
| `engagement_comparison.png`    | 参与度指标对比        |
| `statistical_summary.png`      | 统计显著性表格        |
| `rank_deviation_boxplot.png`   | 排名偏差分布          |
