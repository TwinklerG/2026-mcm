# Task 3: 双轨分层混合效应模型 (DT-HMEM)

## 概述

本任务分析职业舞伴 (Pro Dancers) 和明星特征 (Celebrity Characteristics) 对比赛成绩的影响，直接回答题目问题：**Do they impact judges' scores and fan votes in the same way?**

## 核心发现速览

| 发现 | 评委模型 | 粉丝模型 | 启示 |
|------|---------|---------|------|
| **职业舞伴 ICC** | 9.5% | **16.4%** | Pro 是"流量操盘手"而非"技术教练" |
| **运动员悖论** | β = -0.13 (负) | β = +0.05 (正) | 特征影响方向可相反！ |
| **表现转化率 γ** | — | ≈0, 不显著 | DWTS 是 Identity-Driven Game |

## 模型设计

### 核心思想

数据具有层级结构：
- Level 1: 观测（每周的成绩）
- Level 2: 职业舞伴（分组变量）

> **实现说明**：代码中仅使用职业舞伴作为随机效应分组变量 (`groups=pro_id`)，未包含赛季随机效应或选手个体随机效应。

### 双轨模型

**Track 1: 评委评分模型 (Performance Model)**

$$Y_{i,t}^{J} = \beta_0^J + \beta_{age}^J \cdot \text{Age}_{i}^{centered} + \beta_{week}^J \cdot \text{Week}_t + \sum_{k} \beta_{k}^J \cdot \text{Industry}_{k,i} + u_{p(i)}^J + \epsilon_{i,t}^J$$

**Track 2: 粉丝投票模型 (Popularity Model)**

$$Y_{i,t}^{F} = \beta_0^F + \beta_{age}^F \cdot \text{Age}_{i}^{centered} + \beta_{week}^F \cdot \text{Week}_t + \gamma \cdot Y_{i,t}^{J} + \sum_{k} \beta_{k}^F \cdot \text{Industry}_{k,i} + u_{p(i)}^F + \epsilon_{i,t}^F$$

其中：
- $u_{p(i)} \sim \mathcal{N}(0, \sigma_{pro}^2)$：职业舞伴随机截距
- $\gamma$：表现转化率（评委分对粉丝投票的边际影响）
- 基准行业组：**Actor/Actress（演员）**

### 关键统计量

1. **ICC (组内相关系数)**: 职业舞伴效应解释的方差比例
2. **γ (表现转化率)**: 评委分数对粉丝投票的影响（控制 Week 后）
3. **IDI (影响差异度指数)**: 检验特征对两个模型的影响是否相同

### 重要说明

- **基准组**: 所有行业系数相对于 **Actor/Actress（演员）** 进行比较
- **Week 系数解读**: 粉丝模型中 Week 的正系数主要反映**选票集中效应 (Vote Concentration Effect)**——随着选手淘汰，剩余选手的基准期望份额自然上升，这是数学必然而非行为发现

## 文件结构

```
task3_mixed_effects/
├── __init__.py           # 模块导出
├── data_preparation.py   # 数据准备
├── model.py              # DT-HMEM 模型实现
├── analysis.py           # 结果分析
├── visualization.py      # 可视化
├── run_analysis.py       # 主运行脚本
├── README.md             # 本文件
├── data/                 # 处理后的数据
│   └── panel_data.csv
├── outputs/              # 分析结果
│   ├── judge_model_fixed_effects.csv
│   ├── fan_model_fixed_effects.csv
│   ├── pro_rankings.csv
│   ├── impact_divergence_index.csv
│   └── conclusions.json
└── figures/              # 图表
    ├── pro_scatter.png
    ├── coefficient_comparison.png
    ├── variance_decomposition.png
    ├── industry_effects.png
    ├── learning_curve.png
    └── impact_divergence.png
```

## 运行方法

```bash
cd draft/lly
uv run python -m task3_mixed_effects.run_analysis
```

## 核心发现

### 1. 职业舞伴效应 (O奖级洞察)

**反直觉发现**: 职业舞伴对粉丝投票的 ICC (16.4%) **高于**对评委评分的 ICC (9.5%)。

**启示**: 
- 选 Pro 不仅是选"老师"，更是选"粉丝动员能力"
- Pro 的"流量属性"大于"技术属性"
- 建议为弱势行业选手配对 Fan Favorite 或 Kingmaker 类型舞伴

**四象限分类**:
- **Kingmaker**: 技术高 + 流量高（最理想的搭档）
- **Technician**: 技术高 + 流量低（技术派）
- **Fan Favorite**: 技术低 + 流量高（人气派）
- **Underperformer**: 技术低 + 流量低

### 2. 运动员悖论 (The Athlete Paradox)

这是直接回答题目问题的完美案例：

| 维度 | 系数 | 方向 | 原因推测 |
|------|------|------|----------|
| 评委 | β = -0.13 | **负面** | 身体僵硬、缺乏艺术流畅感 |
| 粉丝 | β = +0.05 | **正面** | 拼搏精神、竞技性、国民度 |

**结论**: 同一特征对评委和粉丝的影响**方向相反**，直接回答了 "Do they impact in the same way?" —— **No**.

### 3. Identity-Driven Game (身份驱动游戏)

在控制 Week（选票集中效应）后，评委打分对粉丝份额的边际贡献 **γ ≈ 0 且不显著**。

**这不是问题，而是核心发现**:
- DWTS 是"身份驱动游戏"而非"表现驱动游戏"
- 粉丝是"粘性"的 (Sticky)，不会因单周表现好坏剧烈改变投票倾向
- 一旦明星建立粉丝基本盘，周间技术波动几乎不影响人气

**政策启示**: 现行赛制无法有效激励选手追求卓越舞蹈表现，这为 Task 4（设计更公平赛制）提供了改进靶点。

### 4. 行业效应说明

所有行业系数均相对于基准组 **Actor/Actress（演员）** 进行比较：
- **Singer**: 评委负、粉丝正 → 可能是过度自信抵消先天乐感
- **Athlete**: 见上述"运动员悖论"
- **Other/Politician**: 两边都弱 → 既无先天优势也无粉丝基础

## 参考

- Raudenbush, S. W., & Bryk, A. S. (2002). Hierarchical Linear Models
- statsmodels.formula.api.mixedlm 文档
