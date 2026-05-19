# Task 3: 双轨分层混合效应模型 (DT-HMEM)

## 目标

分析职业舞伴和明星特征对评委打分与粉丝投票的差异化影响。

回答核心问题：**Do they impact judges' scores and fan votes in the same way?**

---

## 1. 符号定义

| 符号           | 定义                                                 |
| -------------- | ---------------------------------------------------- |
| $i$            | 明星选手索引                                         |
| $t$            | 周次 (Week)                                          |
| $p(i)$         | 选手 $i$ 绑定的职业舞伴 (Pro Partner)                |
| $Y_{i,t}^{J}$  | 选手 $i$ 在第 $t$ 周的评委得分 (`judge_score`)       |
| $Y_{i,t}^{F}$  | 选手 $i$ 在第 $t$ 周的对数粉丝份额 (`log_fan_share`) |
| $\mathbf{X}_i$ | 选手特征向量（年龄、行业哑变量）                     |

**注意**：
- $Y_{i,t}^{F} = \ln(\hat{f}_{i,t} + \epsilon)$，其中 $\hat{f}_{i,t}$ 来自 Task 1 的后验均值
- 基准组 (Reference Category) 为 **Actor/Actress（演员）**

---

## 2. 模型结构

### 数据层级

- **Level 1**: 观测（选手-周）
- **Level 2**: 职业舞伴（分组变量）

> **与初版设计的差异**：实际代码中**未包含**赛季随机效应 $u_s$ 和选手个体随机效应 $\nu_i$，仅使用职业舞伴作为分组变量。

### Track 1: 评委评分模型 (Performance Model)

**代码实现** (`model.py:fit_judge_model`):

$$Y_{i,t}^{J} = \beta_0^J + \beta_{age}^J \cdot \text{Age}_{i}^{centered} + \beta_{week}^J \cdot \text{Week}_t + \sum_{k} \beta_{k}^J \cdot \text{Industry}_{k,i} + u_{p(i)}^J + \epsilon_{i,t}^J$$

其中：
- $u_{p(i)}^J \sim \mathcal{N}(0, \sigma_{pro,J}^2)$：职业舞伴随机截距
- $\epsilon_{i,t}^J \sim \mathcal{N}(0, \sigma_{\epsilon,J}^2)$：观测误差
- `re_formula="~1"` 表示仅有随机截距，无随机斜率

**过滤规则**：仅保留 ≥10 个观测的职业舞伴（42/60 位）

### Track 2: 粉丝投票模型 (Popularity Model)

**代码实现** (`model.py:fit_fan_model`):

$$Y_{i,t}^{F} = \beta_0^F + \beta_{age}^F \cdot \text{Age}_{i}^{centered} + \beta_{week}^F \cdot \text{Week}_t + \gamma \cdot Y_{i,t}^{J} + \sum_{k} \beta_{k}^F \cdot \text{Industry}_{k,i} + u_{p(i)}^F + \epsilon_{i,t}^F$$

其中：
- $\gamma$：**表现转化率** (Performance-to-Popularity Conversion)
- 若 $\gamma > 0$ 且显著：粉丝看重当晚表现
- 若 $\gamma \approx 0$：粉丝投票与当晚评委分无关（身份驱动游戏）

---

## 3. 关键统计量

### 3.1 ICC (组内相关系数)

**代码实现** (`model.py:compute_variance_decomposition`):

$$\text{ICC} = \frac{\sigma_{pro}^2}{\sigma_{pro}^2 + \sigma_{\epsilon}^2}$$

衡量职业舞伴效应解释的方差比例。ICC 越大，说明"跟谁跳舞"越重要。

### 3.2 IDI (影响差异度指数)

**代码实现** (`model.py:compute_impact_divergence_index`):

$$\text{IDI}_k = \frac{|\hat{\beta}_k^J - \hat{\beta}_k^F|}{\sqrt{\text{SE}(\hat{\beta}_k^J)^2 + \text{SE}(\hat{\beta}_k^F)^2}}$$

- IDI > 1.96：5% 显著性水平下存在差异
- IDI > 3：实质性差异（经验规则）

**p 值计算**：双侧 z 检验 $p = 2 \cdot (1 - \Phi(|\text{IDI}|))$

### 3.3 BLUPs (最佳线性无偏预测)

**代码实现** (`model.py:get_pro_rankings`):

职业舞伴的随机效应估计（条件模式）：

$$\hat{u}_p^J = E[u_p^J | \mathbf{Y}^J, \hat{\boldsymbol{\beta}}^J]$$

- **技术加成** (Tech Boost): $\text{Score}_{pro}^{Tech} = \hat{u}_p^J$
- **流量加成** (Pop Boost): $\text{Score}_{pro}^{Pop} = \hat{u}_p^F$

**四象限分类** (基于 Z-score 标准化):
$$z_{tech} = \frac{\hat{u}_p^J - \bar{u}^J}{s_{u^J}}, \quad z_{pop} = \frac{\hat{u}_p^F - \bar{u}^F}{s_{u^F}}$$

| 类型           | 条件                                  |
| -------------- | ------------------------------------- |
| Kingmaker      | $z_{tech} > 0$ 且 $z_{pop} > 0$       |
| Technician     | $z_{tech} > 0$ 且 $z_{pop} \leq 0$    |
| Fan Favorite   | $z_{tech} \leq 0$ 且 $z_{pop} > 0$    |
| Underperformer | $z_{tech} \leq 0$ 且 $z_{pop} \leq 0$ |

**综合得分**：
$$\text{Combined} = \frac{z_{tech} + z_{pop}}{2}$$

### 3.4 γ 敏感性分析（多重共线性检验）

**代码实现** (`model.py:run_gamma_sensitivity_analysis`):

运行三种模型变体：

1. **原模型**：`log_fan_share ~ age + week + judge_score + industry + (1|pro)`
2. **不含 week**：`log_fan_share ~ age + judge_score + industry + (1|pro)`
3. **含交互项**：`log_fan_share ~ age + week + judge_score + judge_score×week + industry + (1|pro)`

**诊断逻辑**：
- 若原模型 γ 不显著，但移除 week 后 γ 显著 → 多重共线性
- 若交互项显著 → γ 的效应随时间变化

---

## 数据说明

| 指标          | 值            |
| ------------- | ------------- |
| 观测数        | 2738          |
| 选手数        | 411           |
| 赛季数        | 34            |
| 职业舞伴数    | 60            |
| 建模舞伴数    | 42 (≥10 观测) |
| 行业类别数    | 7             |
| 年龄范围      | 14–82 岁      |
| 周次范围      | 1–11          |
| 基准组 (行业) | Actor/Actress |

**行业分布**：

| 行业          | 观测数 |
| ------------- | ------ |
| Actor         | 875    |
| Athlete       | 642    |
| TV            | 585    |
| Singer        | 400    |
| Entertainment | 89     |
| Model         | 76     |
| Other         | 71     |

---

## 固定效应估计

### 评委模型 (Judge Score)

| 变量                   | β      | SE    | z      | p-value | 显著 |
| ---------------------- | ------ | ----- | ------ | ------- | ---- |
| Intercept              | 6.580  | 0.069 | 95.28  | < 0.001 | ***  |
| age_centered           | -0.033 | 0.002 | -18.81 | < 0.001 | ***  |
| week                   | 0.311  | 0.007 | 44.97  | < 0.001 | ***  |
| industry_Athlete       | -0.129 | 0.054 | -2.38  | 0.0171  | *    |
| industry_Entertainment | -0.223 | 0.119 | -1.88  | 0.0604  | ns   |
| industry_Model         | -0.535 | 0.130 | -4.11  | < 0.001 | ***  |
| industry_Other         | -0.689 | 0.131 | -5.26  | < 0.001 | ***  |
| industry_Singer        | -0.021 | 0.061 | -0.34  | 0.7312  | ns   |
| industry_TV            | -0.419 | 0.055 | -7.58  | < 0.001 | ***  |

### 粉丝模型 (Log Fan Share)

| 变量                   | β          | SE    | z      | p-value | 显著 |
| ---------------------- | ---------- | ----- | ------ | ------- | ---- |
| Intercept              | -2.737     | 0.038 | -72.92 | < 0.001 | ***  |
| age_centered           | -0.002     | 0.000 | -4.11  | < 0.001 | ***  |
| week                   | 0.114      | 0.002 | 52.17  | < 0.001 | ***  |
| industry_Athlete       | **+0.052** | 0.013 | 4.05   | < 0.001 | ***  |
| industry_Entertainment | -0.125     | 0.028 | -4.43  | < 0.001 | ***  |
| industry_Model         | -0.217     | 0.031 | -6.96  | < 0.001 | ***  |
| industry_Other         | -0.246     | 0.032 | -7.68  | < 0.001 | ***  |
| industry_Singer        | -0.019     | 0.015 | -1.27  | 0.2025  | ns   |
| industry_TV            | -0.033     | 0.013 | -2.51  | 0.0121  | *    |
| judge_score (γ)        | 0.004      | 0.005 | 0.80   | 0.4220  | ns   |

---

## 方差分解与 ICC

| 模型 | Pro 方差 ($\sigma^2_{pro}$) | 残差方差 ($\sigma^2_\epsilon$) | 总方差 | ICC        |
| ---- | --------------------------- | ------------------------------ | ------ | ---------- |
| 评委 | 0.0951                      | 0.9074                         | 1.0025 | **9.49%**  |
| 粉丝 | 0.0141                      | 0.0508                         | 0.0649 | **21.77%** |

> **解读**：职业舞伴对粉丝投票的 ICC (21.8%) **显著高于**对评委评分的 ICC (9.5%)。  
> 职业舞伴的"明星效应"在粉丝投票中更为显著，评委评分相对更独立于舞伴身份。

---

## 影响差异度指数 (IDI)

| 变量                   | β_Judge | β_Fan      | 差值   | SE_diff | IDI       | p-value | 显著差异 |
| ---------------------- | ------- | ---------- | ------ | ------- | --------- | ------- | -------- |
| week                   | 0.311   | 0.114      | 0.198  | 0.007   | **27.26** | < 0.001 | ✓        |
| age_centered           | -0.033  | -0.002     | -0.031 | 0.002   | **17.19** | < 0.001 | ✓        |
| industry_TV            | -0.419  | -0.033     | -0.386 | 0.057   | **6.79**  | < 0.001 | ✓        |
| industry_Other         | -0.689  | -0.246     | -0.444 | 0.135   | **3.29**  | 0.0010  | ✓        |
| **industry_Athlete**   | -0.129  | **+0.052** | -0.181 | 0.055   | **3.26**  | 0.0011  | ✓        |
| industry_Model         | -0.535  | -0.217     | -0.318 | 0.134   | 2.37      | 0.0176  | ✓        |
| industry_Entertainment | -0.223  | -0.125     | -0.098 | 0.122   | 0.80      | 0.4223  | ✗        |
| industry_Singer        | -0.021  | -0.019     | -0.002 | 0.063   | 0.04      | 0.9691  | ✗        |

> **IDI > 3** 表示两系统间存在实质性差异。

---

## 核心发现

### 1. 运动员悖论 (Athlete Paradox)

| 维度 | 系数   | 方向   | 显著 |
| ---- | ------ | ------ | ---- |
| 评委 | -0.129 | **负** | *    |
| 粉丝 | +0.052 | **正** | ***  |

同一特征对两个系统的影响**方向相反**：

- 评委认为运动员舞技较差（β < 0）
- 粉丝却更支持运动员（β > 0）

**直接回答问题**：**No, they do NOT impact in the same way.**

### 2. 年龄效应

| 维度 | 系数   | 解读                 |
| ---- | ------ | -------------------- |
| 评委 | -0.033 | 年龄每增 1 岁扣 0.03 |
| 粉丝 | -0.002 | 年龄影响几乎可忽略   |

> 年龄对评委评分的影响是粉丝的 **18 倍**（IDI = 17.19）。

### 3. 周次效应

| 维度 | 系数  | 解读                 |
| ---- | ----- | -------------------- |
| 评委 | 0.311 | 每周平均提升 0.31 分 |
| 粉丝 | 0.114 | 每周平均提升 11.4%   |

> 评委对持续比赛的奖励幅度是粉丝的 **2.7 倍**。

### 4. 表现转化率 (γ) 与多重共线性

原始结论：
$$\gamma = 0.004, \quad p = 0.42 \text{ (不显著)}$$

**问题**：由于 `week` 与 `judge_score` 存在生存者偏差导致的正相关，原结论需要修正。

---

## γ 敏感性分析（多重共线性检验）

| 模型              | γ         | p-value | 显著   |
| ----------------- | --------- | ------- | ------ |
| 原模型            | 0.0037    | 0.4220  | **否** |
| **不含 week**     | **0.161** | < 0.001 | **是** |
| 含交互项          | -0.040    | < 0.001 | 是     |
| 交互项 (γ × week) | 0.012     | < 0.001 | 是     |

> **结论：多重共线性确认！**
>
> 移除 `week` 后 γ 从 0.004 (ns) 变为 **0.161 (p<0.001)**。
>
> **修正后的解读**：
>
> - "Fans conflate longevity with quality"——粉丝确实看重表现，但表现和持续时间混在一起无法区分
> - 交互项显著说明评委分数对粉丝投票的影响随周次变化：后期（week 大）时，评委分数影响更强

---

## 职业舞伴效应分析

### 方差解释

| 模型 | 职业舞伴解释比例 |
| ---- | ---------------- |
| 评委 | 9.49%            |
| 粉丝 | 21.77%           |

### 四象限分类

基于 Z-score 标准化的技术提升 (tech_boost_z) 和流量提升 (pop_boost_z)：

| 类型           | 定义              | 人数 | 典型代表                      |
| -------------- | ----------------- | ---- | ----------------------------- |
| Kingmaker      | tech_z>0, pop_z>0 | 8    | Derek Hough, Tony Dovolani    |
| Technician     | tech_z>0, pop_z≤0 | 13   | Artem Chigvintsev             |
| Fan Favorite   | tech_z≤0, pop_z>0 | 9    | Alec Mazo, Ashly DelGrosso    |
| Underperformer | tech_z≤0, pop_z≤0 | 12   | Gleb Savchenko, Britt Stewart |

### Top 5 Kingmakers（综合得分）

| 排名 | 舞伴               | tech_boost | pop_boost | combined |
| ---- | ------------------ | ---------- | --------- | -------- |
| 1    | Derek Hough        | +0.656     | +0.060    | **1.46** |
| 2    | Jonathan Roberts   | +0.149     | +0.180    | **1.07** |
| 3    | Kym Johnson        | +0.213     | +0.103    | **0.84** |
| 4    | Tony Dovolani      | +0.286     | +0.072    | **0.84** |
| 5    | Maksim Chmerkoskiy | +0.299     | +0.061    | **0.82** |

### Top 5 技术型 (Technician)

| 排名 | 舞伴                  | tech_boost |
| ---- | --------------------- | ---------- |
| 1    | Derek Hough           | +0.656     |
| 2    | Artem Chigvintsev     | +0.429     |
| 3    | Witney Carson (S33)   | +0.329     |
| 4    | Valentin Chmerkovskiy | +0.325     |
| 5    | Maksim Chmerkoskiy    | +0.299     |

### Top 5 流量型 (Popularity)

| 排名 | 舞伴             | pop_boost |
| ---- | ---------------- | --------- |
| 1    | Ashly DelGrosso  | +0.373    |
| 2    | Alec Mazo        | +0.203    |
| 3    | Jonathan Roberts | +0.180    |
| 4    | Edyta Sliwinska  | +0.151    |
| 5    | Louis van Amstel | +0.111    |

### Bottom 5（表现最差）

| 排名 | 舞伴              | combined |
| ---- | ----------------- | -------- |
| 42   | Britt Stewart     | -1.53    |
| 41   | Rylee Arnold      | -1.32    |
| 40   | Ezra Sosa (S33)   | -0.83    |
| 39   | Chelsie Hightower | -0.80    |
| 38   | Daniella Karagach | -0.69    |

### 经验与效应的相关性

| 指标         | 相关系数 | p-value | 显著 |
| ------------ | -------- | ------- | ---- |
| 技术 vs 经验 | 0.250    | 0.111   | ✗    |
| 流量 vs 经验 | 0.205    | 0.192   | ✗    |

> 职业舞伴的"经验"（参赛次数）与其技术/流量提升能力**无显著相关**，说明效应主要来自个人特质而非经验积累。

---

## 成分数据假设验证

**假设声明**：Fan Share 是归一化的（总和为 1），这意味着同一周内选手间存在结构性负相关 $\rho \approx -1/(N-1)$。

**诊断结果**：

| 指标           | 值     |
| -------------- | ------ |
| 每周平均选手数 | 7.92   |
| 周次数         | 335    |
| 理论负相关     | -0.145 |
| 经验残差相关   | ≈ 0    |
| 相关配对数     | 10648  |
| SE 膨胀因子    | 1.0    |
| 关注程度       | **低** |

> 经验残差相关 (≈0) 弱于理论值 (-0.145)，说明成分数据约束的影响已被模型部分吸收。  
> **结论**：独立性假设近似成立，无需使用成分数据专用方法（如 Dirichlet 回归）。

---

## 综合结论

### 核心问题回答

**Q: Do [pro dancers and celebrity characteristics] impact judges' scores and fan votes in the same way?**

**A: No.**

1. **运动员悖论**：运动员在评委眼中表现较差 (β=-0.129\*)，但在粉丝中更受欢迎 (β=+0.052\*\*\*)
2. **年龄差异**：年龄对评委影响是粉丝的 18 倍
3. **职业舞伴效应**：舞伴对粉丝投票的解释力 (21.8%) 是评委评分的 2.3 倍
4. **表现转化**：粉丝将"持久参赛"与"高水平表演"混为一谈

### 实践启示

1. **选角策略**：运动员是"粉丝收割机"但"评委杀手"
2. **舞伴配对**：顶级 Kingmaker（如 Derek Hough）可同时提升技术分和粉丝票
3. **赛制设计**：当前规则下粉丝投票与技术水平存在系统性偏差

---

## 输出文件清单

| 文件                             | 内容                        |
| -------------------------------- | --------------------------- |
| `judge_model_fixed_effects.csv`  | 评委模型固定效应系数        |
| `fan_model_fixed_effects.csv`    | 粉丝模型固定效应系数        |
| `judge_model_blups.csv`          | 评委模型职业舞伴 BLUP 估计  |
| `fan_model_blups.csv`            | 粉丝模型职业舞伴 BLUP 估计  |
| `variance_decomposition.json`    | ICC 和方差分解              |
| `impact_divergence_index.csv`    | IDI 指标                    |
| `fixed_effects_analysis.json`    | 固定效应对比分析            |
| `pro_rankings.csv`               | 职业舞伴排名与分类          |
| `pro_rankings_full.csv`          | 完整职业舞伴排名（含 Z 分） |
| `pro_effects_analysis.json`      | 职业舞伴效应详细分析        |
| `celebrity_analysis.json`        | 明星特征效应分析            |
| `performance_conversion.json`    | γ 系数统计                  |
| `gamma_sensitivity.json`         | γ 多重共线性敏感性分析      |
| `compositional_diagnostics.json` | 成分数据诊断结果            |
| `data_summary.json`              | 数据集基本统计              |
| `conclusions.json`               | 核心结论汇总                |
