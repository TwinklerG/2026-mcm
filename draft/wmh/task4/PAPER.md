# Task 4: Progressive Technical Fairness System Design

## 论文写作指南

> 本文件为问题四的论文部分，采用双语格式：英文为论文正文，中文为解释说明。本部分应作为论文的一个子章节（如 Section 6），而非独立成章。本章节综合运用 Task 1 的粉丝份额估计和 Task 3 的影响因素分析，提出一个新的投票组合系统。

---

## 6. A Progressive Technical Fairness System for DWTS

### 6.1 Motivation and Problem Diagnosis

**【论文正文】**

The final challenge posed in Problem C asks us to *"propose another system using fan votes and judge scores each week that you believe is more 'fair' (or 'better' in some other way such as making the show more exciting for the fans)."* This question recognizes a fundamental tension in competitive entertainment: technical merit and popular appeal do not always align.

Our analyses from previous sections provide a rigorous diagnostic foundation:

**Finding 1 (From Task 1)**: The Judges' Save mechanism introduced in Season 28 has largely failed to protect technically superior contestants. In 90.5% of cases, judges saved the contestant with higher estimated fan share, showing no significant preference for technical merit ($p = 0.56$, binomial test).

**Finding 2 (From Task 3)**: The **Athlete Paradox** reveals systematic divergence between judge and fan evaluations. Athletes receive lower judge scores ($\beta^J = -0.129$, $p = .017$) yet higher fan votes ($\beta^F = +0.052$, $p < .001$)—effects in opposite directions (IDI = 3.26).

**Finding 3 (From Task 3)**: Professional dancers explain 2.3× more variance in fan voting (ICC = 21.8%) than in judge scoring (ICC = 9.5%), indicating that "who you dance with" disproportionately influences popularity outcomes.

These findings suggest that neither a pure judge-based system (which carries its own biases) nor a pure fan-based system (which prioritizes celebrity recognition over dance skill) can achieve fairness in isolation. The challenge is to design a hybrid system that progressively balances these competing objectives.

**【中文解释】**

> **问题诊断的核心逻辑**：
> - Task 1 发现：Judges' Save 机制名存实亡（90.5% 情况下救了粉丝更高的人）
> - Task 3 发现：运动员悖论揭示了评委与粉丝的系统性分歧
> - Task 3 发现：舞伴对粉丝票的影响是对评委分的 2.3 倍
>
> **结论**：现有系统无法平衡技术公平与观众参与，需要全新设计。

---

### 6.2 Design Principles

**【论文正文】**

Drawing from social choice theory @arrow1951social @sen1970collective, we identify three desirable properties for any improved voting system:

1. **Technical Fairness**: Final rankings should correlate positively with cumulative technical performance, ensuring that skilled dancers are rewarded.

2. **Audience Engagement**: The system should maintain competitive tension and meaningful fan participation, preserving the show's entertainment value @nurmi2010voting.

3. **Robustness**: Performance should be consistent across different seasons and contestant compositions, avoiding pathological outcomes @brandt2016handbook.

These properties admit trade-offs. Maximizing technical fairness (e.g., using 100% judge scores) would eliminate fan engagement entirely. Conversely, pure fan voting, as demonstrated by the Bobby Bones controversy (Season 27), can produce outcomes widely perceived as unfair @patty2019measuring.

We therefore propose a **Progressive Technical Fairness System (PTFS)** that dynamically adjusts the balance between judge scores and fan votes across the competition timeline.

**【中文解释】**

> **设计原则来源于社会选择理论**：
> 1. **技术公平性**：最终排名应与技术能力相关
> 2. **观众参与度**：保持悬念和粉丝投入感
> 3. **鲁棒性**：跨赛季表现稳定
>
> **关键洞见**：这三个目标存在内在张力，不可能同时最大化。PTFS 的设计思路是在比赛进程中动态调整权重，实现"早期重参与、后期重技术"的平衡。

---

### 6.3 Model Specification

**【论文正文】**

#### 6.3.1 Core Scoring Formula

Let $n_t$ denote the number of contestants competing in week $t$, and let $T$ be the total number of weeks in the season. For contestant $i$ in week $t$, we define:

- $J_{i,t}$: Average judge score (normalized to $[0, 10]$)
- $\hat{f}_{i,t}$: Estimated fan share from Task 1's Bayesian MCMC model

The PTFS combined score is computed as:

$$S_{i,t} = w_J(t) \cdot \tilde{J}_{i,t} + (1 - w_J(t)) \cdot \tilde{F}_{i,t}$$

where $\tilde{J}_{i,t}$ and $\tilde{F}_{i,t}$ are normalized scores within each week:

$$\tilde{J}_{i,t} = \frac{J_{i,t}}{\sum_{j=1}^{n_t} J_{j,t}}, \quad \tilde{F}_{i,t} = \frac{\hat{f}_{i,t}}{\sum_{j=1}^{n_t} \hat{f}_{j,t}}$$

This normalization ensures both components sum to unity within each week, enabling meaningful combination regardless of absolute magnitudes.

#### 6.3.2 Dynamic Weight Function

The key innovation of PTFS is a time-varying judge weight:

$$w_J(t) = w_J^{\text{start}} + (w_J^{\text{end}} - w_J^{\text{start}}) \cdot \frac{t-1}{T-1}$$

This linear interpolation produces:
- **Week 1**: $w_J = w_J^{\text{start}}$ (lower judge weight, higher fan influence)
- **Final Week**: $w_J = w_J^{\text{end}}$ (higher judge weight, technical merit emphasized)

**Optimal Parameters** (from grid search over 34 seasons):

$$w_J^{\text{start}} = 0.45, \quad w_J^{\text{end}} = 0.80$$

**Figure 6.1: Dynamic Weight Progression** *(建议制图)*

> **图表说明**：折线图，x轴为周次（1 到 T），y轴为权重（0 到 1）。蓝色线表示评委权重 $w_J(t)$，从 0.45 线性增长到 0.80。红色线表示粉丝权重 $1 - w_J(t)$，从 0.55 线性下降到 0.20。标注关键里程碑：Week 1（45%-55%）、Midpoint（62.5%-37.5%）、Final（80%-20%）。

**【中文解释】**

> **PTFS 的核心创新**：
> 1. **归一化得分**：将评委分和粉丝份额都转换为周内百分比，使二者可比
> 2. **动态权重**：评委权重从 45% 线性增长到 80%
>
> **设计理念**：
> - **早期（45% 评委）**：粉丝投票占优，维持悬念和参与度，允许"黑马"存活
> - **后期（80% 评委）**：技术评分主导，确保决赛选手是真正有实力的舞者
>
> **为什么是 45% 和 80%？** 这是通过网格搜索在 34 个赛季数据上优化得到的，见敏感性分析。

---

### 6.4 Evaluation Framework

**【论文正文】**

To rigorously evaluate PTFS against existing systems (Rank-based and Percentage-based), we developed a comprehensive multi-dimensional evaluation framework.

#### 6.4.1 Technical Fairness Metrics

| Metric                 | Definition                                                      | Desired Direction |
| ---------------------- | --------------------------------------------------------------- | ----------------- |
| Kendall's $\tau$       | Correlation between cumulative judge ranks and final placements | Higher (↑)        |
| Spearman's $\rho$      | Rank correlation (alternative measure)                          | Higher (↑)        |
| Tech Top3 → Final Top3 | Proportion of technical top-3 who finish in final top-3         | Higher (↑)        |
| Tech Lowest Eliminated | Proportion of weeks where technical lowest is eliminated        | Higher (↑)        |
| Mean Rank Deviation    | Average absolute difference between technical and final rank    | Lower (↓)         |

Kendall's $\tau$ @kendall1938new serves as our primary fairness metric, measuring the ordinal association between cumulative technical performance and competitive outcomes:

$$\tau = \frac{(\text{concordant pairs}) - (\text{discordant pairs})}{\binom{n}{2}}$$

#### 6.4.2 Audience Engagement Metrics

| Metric                | Definition                                                    | Interpretation                 |
| --------------------- | ------------------------------------------------------------- | ------------------------------ |
| Close Call Rate       | Proportion of eliminations with score gap < 2%                | Higher = more suspense         |
| Tech Top50 Eliminated | Proportion of "upset" eliminations (tech top-half eliminated) | Moderate = balanced excitement |

#### 6.4.3 Robustness Metrics

| Metric                    | Definition                                     | Desired Direction |
| ------------------------- | ---------------------------------------------- | ----------------- |
| $\tau$ Std Across Seasons | Standard deviation of $\tau$ across 34 seasons | Lower (↓)         |
| Consistent Seasons        | Proportion of seasons with $\tau > 0.5$        | Higher (↑)        |

**【中文解释】**

> **评价框架的设计原则**：
> 1. **技术公平性**：核心指标是 Kendall's τ，衡量技术能力与最终名次的相关性
> 2. **观众参与度**：Close Call Rate 衡量"悬念时刻"的频率
> 3. **鲁棒性**：跨赛季稳定性，避免某些赛季表现异常
>
> **为什么不用"争议案例修正率"？** 针对特定历史案例（如 Bobby Bones）的指标会导致过拟合，我们追求的是系统性改进而非个案修复。

---

### 6.5 Experimental Results

**【论文正文】**

We simulated all three voting systems across 34 seasons of DWTS (1,411 contestant-weeks, 261 elimination events) using estimated fan shares from Task 1.

#### 6.5.1 Main Comparison

**Table 25: System Comparison Across All Metrics (34 Seasons)**

| Metric                            | Rank-based | Percentage | **PTFS**  | Best     |
| --------------------------------- | ---------- | ---------- | --------- | -------- |
| **Technical Fairness**            |            |            |           |          |
| Kendall's $\tau$                  | 0.674      | 0.727      | **0.749** | PTFS     |
| Spearman's $\rho$                 | 0.815      | 0.860      | **0.876** | PTFS     |
| Tech Top3 → Final Top3            | 59.8%      | 71.6%      | **72.5%** | PTFS     |
| Mean Rank Deviation               | 1.61       | 1.35       | **1.26**  | PTFS     |
| Tech Lowest Eliminated            | 41.9%      | 47.2%      | **52.8%** | PTFS     |
| **Engagement**                    |            |            |           |          |
| Close Call Rate                   | 30.2%      | 66.1%      | **79.1%** | PTFS     |
| Tech Top50 Eliminated (Upset)     | 7.0%       | 9.6%       | 8.6%      | Balanced |
| **Robustness**                    |            |            |           |          |
| $\tau$ Std Across Seasons         | 0.150      | 0.129      | **0.125** | PTFS     |
| Consistent Seasons ($\tau > 0.5$) | 91.2%      | 94.1%      | **97.1%** | PTFS     |

PTFS outperforms both existing systems across all primary metrics. The improvement is most pronounced in:
- **Technical fairness**: +0.022 $\tau$ over Percentage-based (+3.0% relative)
- **Engagement**: +13.0 percentage points in Close Call Rate
- **Consistency**: 97.1% of seasons achieve $\tau > 0.5$ (vs. 94.1% for Percentage)

#### 6.5.2 Statistical Significance

**Table 26: Paired Statistical Tests (PTFS vs. Alternatives)**

| Comparison          | Δτ     | Test Statistic  | $p$-value   | Significant at α = 0.05 |
| ------------------- | ------ | --------------- | ----------- | ----------------------- |
| PTFS vs. Percentage | +0.022 | $t_{33} = 2.26$ | **0.031**   | ✓ Yes                   |
| PTFS vs. Rank-based | +0.075 | $t_{33} = 4.12$ | **< 0.001** | ✓ Yes                   |

The improvement of PTFS over the Percentage-based system is statistically significant ($p = 0.031$) using a paired $t$-test across 34 seasons. This provides strong evidence that PTFS represents a genuine improvement rather than random variation.

**Figure 6.2: Kendall's τ by Season** *(建议制图)*

> **图表说明**：折线图或散点图，x轴为赛季（1-34），y轴为 Kendall's τ（0-1）。三条线分别代表 Rank-based（蓝色）、Percentage（绿色）、PTFS（红色）。在 S28 处添加垂直虚线标注"Rule Change"。突出显示 PTFS 在大多数赛季的优势。

**【中文解释】**

> **核心结果解读**：
> - **Kendall's τ = 0.749**：PTFS 实现了三种系统中最高的技术-排名相关性
> - **统计显著 (p = 0.031)**：PTFS 相对 Percentage 的改进不是偶然的
> - **Close Call Rate = 79.1%**：近 80% 的淘汰存在悬念（前两名差距 < 2%）
> - **97.1% 赛季达到 τ > 0.5**：PTFS 在几乎所有赛季都表现良好

---

### 6.6 Trade-off Analysis

**【论文正文】**

No voting system can simultaneously maximize all desirable properties—this is a fundamental insight from social choice theory @arrow1951social. PTFS represents a principled trade-off:

**Table 27: PTFS Trade-off Analysis vs. Percentage-based**

| Dimension                   | Change     | Interpretation                                      |
| --------------------------- | ---------- | --------------------------------------------------- |
| Technical Fairness ($\tau$) | **+0.022** | Statistically significant improvement ($p = 0.031$) |
| Ranking Consistency         | **+2.9%**  | 97% vs. 94% seasons with $\tau > 0.5$               |
| Suspense (Close Call Rate)  | **+13.0%** | Substantially more "edge-of-seat" eliminations      |
| Fair Elimination            | **+5.6%**  | Tech lowest eliminated more often (52.8% vs. 47.2%) |
| Upset Rate                  | **−1.0%**  | Slightly fewer "unfair" eliminations                |

The trade-off frontier reveals that PTFS achieves a Pareto improvement over the Percentage-based system: it is better on all measured dimensions without sacrificing any.

**Figure 6.3: Improvement Distribution Across Seasons** *(建议制图)*

> **图表说明**：直方图，x轴为 τ 改进量（PTFS - Percentage），y轴为赛季数量。标注正改进（> 0）和负改进（< 0）的数量。应显示约 70-75% 的赛季有正改进。

**【中文解释】**

> **权衡分析的关键发现**：
> - PTFS 实现了**帕累托改进**：在所有维度上都至少与 Percentage 持平或更好
> - **悬念大幅提升**：Close Call Rate 从 66% 增至 79%，观众体验更佳
> - **技术公平与参与度双赢**：动态权重策略成功平衡了两个目标

---

### 6.7 Sensitivity Analysis

**【论文正文】**

To assess the robustness of our parameter choices, we conducted a comprehensive sensitivity analysis over the parameter space.

#### 6.7.1 Parameter Grid

| Parameter                  | Search Range                   | Optimal Value |
| -------------------------- | ------------------------------ | ------------- |
| $w_J^{\text{start}}$       | [0.35, 0.40, 0.45, 0.50]       | **0.45**      |
| $w_J^{\text{end}}$         | [0.60, 0.65, 0.70, 0.75, 0.80] | **0.80**      |
| Improvement bonus $\alpha$ | [0.00, 0.05, 0.10, 0.15]       | **0.00**      |
| Protection threshold       | [0.00, 0.10, 0.15, 0.20]       | **0.00**      |

The grid search evaluated 320 parameter combinations, optimizing a composite objective function weighted toward technical fairness (40%), consistency (15%), tech winner rate (15%), top-3 retention (10%), and moderate upset rate (5%).

**Table 28: Sensitivity Analysis by Parameter**

| Parameter                  | Best Value | Mean Score | Sensitivity |
| -------------------------- | ---------- | ---------- | ----------- |
| $w_J^{\text{start}}$       | 0.45       | 0.530      | Moderate    |
| $w_J^{\text{end}}$         | 0.80       | 0.529      | Low         |
| Improvement bonus $\alpha$ | 0.00       | 0.566      | **High**    |
| Protection threshold       | 0.00       | 0.530      | Low         |

**Key Finding**: The improvement bonus ($\alpha$) and protection threshold show diminishing or negative returns when increased from zero. This suggests that the core dynamic weight mechanism alone is sufficient—additional complexity does not improve performance.

**Figure 6.4: Parameter Sensitivity Heatmap** *(建议制图)*

> **图表说明**：2D 热力图，x轴为 $w_J^{\text{start}}$，y轴为 $w_J^{\text{end}}$。颜色表示目标函数得分。标注最优点 (0.45, 0.80)。显示参数空间的平滑性。

**【中文解释】**

> **敏感性分析的核心发现**：
> 1. **最优参数稳健**：$w_J^{\text{start}} = 0.45$, $w_J^{\text{end}} = 0.80$ 在多种条件下表现最佳
> 2. **进步奖励无效**：improvement bonus 反而降低了性能，说明"奖励进步"的直觉并不总是正确
> 3. **保护机制无效**：技术保护门槛也未带来改进
>
> **结论**：PTFS 的核心价值在于动态权重机制本身，无需额外复杂性。

---

### 6.8 Comparison with Historical Controversies

**【论文正文】**

To illustrate PTFS's practical impact, we examine how it would have affected historically controversial seasons:

**Table 29: Controversy Case Analysis**

| Season | Controversy                                    | Percentage Outcome | PTFS Outcome | Improvement |
| ------ | ---------------------------------------------- | ------------------ | ------------ | ----------- |
| S2     | Jerry Rice (runner-up, tech bottom in 5 weeks) | 2nd place          | 4th place    | ✓ Lower     |
| S4     | Billy Ray Cyrus (5th, tech bottom in 6 weeks)  | 5th place          | 7th place    | ✓ Lower     |
| S11    | Bristol Palin (3rd, tech bottom 12 times)      | 3rd place          | 5th place    | ✓ Lower     |
| S27    | Bobby Bones (winner, consistently low scores)  | 1st place          | 3rd place    | ✓ Lower     |

In all four canonical controversy cases, PTFS would have produced outcomes more aligned with technical rankings, reducing the magnitude of fan-driven anomalies while not entirely eliminating fan influence.

**Caveat**: These counterfactual analyses assume that fan voting behavior would remain unchanged under PTFS. In reality, fans might adapt their strategies if they knew the rules were different—a consideration from mechanism design theory @myerson2013fundamentals.

**【中文解释】**

> **争议案例分析**：
> - 所有四个经典争议案例中，PTFS 都会产生更符合技术排名的结果
> - Jerry Rice 从第 2 降至第 4，Bobby Bones 从冠军降至第 3
>
> **重要警告**：这是反事实分析，假设粉丝行为不变。实际上，规则变化可能改变投票策略（机制设计理论的核心问题）。

---

### 6.9 Implementation Recommendations

**【论文正文】**

Based on our analysis, we provide the following recommendations for DWTS producers:

**Primary Recommendation**: Adopt the Progressive Technical Fairness System (PTFS) with parameters:
- Initial judge weight: 45%
- Final judge weight: 80%
- Linear progression across season

**Supporting Rationale**:

1. **Statistically Validated Improvement**: PTFS achieves significantly higher technical-outcome correlation ($p = 0.031$) than the current Percentage-based system.

2. **Enhanced Viewer Experience**: The 79.1% Close Call Rate (vs. 66.1%) suggests more nail-biting eliminations, potentially increasing viewer engagement and social media discussion.

3. **Reduced Controversy Risk**: By ensuring technically superior dancers advance further, PTFS mitigates the risk of outcomes that generate negative publicity (e.g., Bobby Bones backlash).

4. **Simple Implementation**: PTFS requires only a formula change in score computation—no infrastructure modifications or new voting mechanisms.

5. **Transparent Rules**: The linear weight progression is easy to communicate to audiences, maintaining fairness perception.

**Alternative Recommendation**: If producers prefer to retain the current system, we recommend enhancing the Judges' Save mechanism with explicit criteria requiring judges to evaluate technical merit, as our Task 1 analysis shows the current implementation fails to protect skilled dancers (90.5% alignment with fan preferences).

**【中文解释】**

> **对制片方的核心建议**：
> 1. **采用 PTFS**：参数简单（45% → 80%），统计显著（p=0.031）
> 2. **提升观众体验**：79% 的悬念淘汰比 66% 更刺激
> 3. **降低争议风险**：技术强者更可能晋级，减少"不公平"感知
> 4. **易于实施**：只需修改得分公式，无需改变基础设施
> 5. **规则透明**：线性权重变化易于向观众解释

---

### 6.10 Limitations and Future Work

**【论文正文】**

Several limitations should be acknowledged:

1. **Dependence on Fan Share Estimates**: PTFS evaluation relies on Task 1's Bayesian MCMC estimates. While these achieve 93.49% reconstruction accuracy, they are posterior means rather than ground truth.

2. **Static Behavioral Assumption**: Our counterfactual analyses assume fan voting patterns remain constant under rule changes. Strategic voting behavior may emerge @gibbard1973manipulation @satterthwaite1975strategy.

3. **Linear Weight Assumption**: The linear progression from $w_J^{\text{start}}$ to $w_J^{\text{end}}$ was chosen for simplicity. Non-linear functions (e.g., sigmoid) may offer further optimization.

4. **Single Objective Optimization**: Our grid search optimized a composite score. Multi-objective optimization (Pareto frontier exploration) could reveal additional trade-off configurations.

Future work could address these limitations through:
- Field experiments with real voting data under modified rules
- Game-theoretic modeling of strategic fan behavior
- Machine learning approaches to adaptive weight functions

**【中文解释】**

> **局限性诚实声明**：
> 1. 依赖 Task 1 的粉丝份额估计（虽然准确率高但不是真实数据）
> 2. 假设粉丝行为不变（但规则变化可能诱发策略性投票）
> 3. 线性权重只是一种选择（可能存在更优的非线性函数）
> 4. 单目标优化（多目标帕累托优化可能发现更多方案）

---

### 6.11 Answer to Problem 4

**【论文正文】**

**Question**: *Propose another system using fan votes and judge scores each week that you believe is more "fair" (or "better" in some other way such as making the show more exciting for the fans). Provide support for why your approach should be adopted by the show producers.*

**Answer**: We propose the **Progressive Technical Fairness System (PTFS)**, which dynamically adjusts the judge-fan weight ratio from 45:55 in Week 1 to 80:20 in the finale.

**Evidence of "More Fair"**:
- Kendall's $\tau$ increases from 0.727 (Percentage) to 0.749 (PTFS)
- Paired $t$-test: $t_{33} = 2.26$, $p = 0.031$ (statistically significant)
- 72.5% of technical top-3 reach final top-3 (vs. 71.6%)
- 52.8% of weeks eliminate the technical lowest (vs. 47.2%)

**Evidence of "More Exciting"**:
- Close Call Rate increases from 66.1% to 79.1%
- 97.1% of seasons achieve $\tau > 0.5$ (consistent outcomes reduce arbitrary-feeling results)

**Why Producers Should Adopt**:
1. **Data-Driven**: Validated on 34 seasons of historical data
2. **Statistically Significant**: Not a random improvement
3. **Easy to Implement**: Simple formula change
4. **Addresses Past Controversies**: Would have produced more "fair" outcomes in Bobby Bones, Bristol Palin, and other contentious seasons
5. **Balances Stakeholder Interests**: Satisfies both technically-oriented viewers and casual fans

**【中文解释】**

> **对问题四的直接回答**：
> - **"更公平"的证据**：τ 从 0.727 提升至 0.749（p = 0.031 显著）
> - **"更刺激"的证据**：Close Call Rate 从 66% 提升至 79%
> - **应采用的理由**：数据驱动、统计显著、易实施、解决历史争议、平衡各方利益

---

## Figures and Tables Summary

### 建议图表清单

| 图表编号 | 类型             | 内容                        | 用途               |
| -------- | ---------------- | --------------------------- | ------------------ |
| Fig. 6.1 | Line Chart       | 动态权重曲线 (45% → 80%)    | 解释 PTFS 核心机制 |
| Fig. 6.2 | Multi-line Chart | 34 赛季 τ 对比              | 主要证据展示       |
| Fig. 6.3 | Histogram        | PTFS vs Percentage 改进分布 | 支持统计显著性     |
| Fig. 6.4 | Heatmap          | 参数敏感性                  | 验证参数选择稳健性 |
| Table 25 | Comparison       | 系统全指标对比              | **核心结果表**     |
| Table 26 | Statistical Test | 配对 t 检验结果             | 显著性证据         |
| Table 27 | Trade-off        | PTFS vs Percentage 权衡分析 | 帕累托改进论证     |
| Table 28 | Sensitivity      | 参数敏感性                  | 稳健性验证         |
| Table 29 | Case Study       | 争议案例反事实分析          | 实践意义展示       |

### 数据展示建议

**小表（可完整纳入论文）**：
- Table 25 (系统对比) — **核心结果，必须包含**
- Table 26 (统计显著性) — 关键证据
- Table 27 (权衡分析) — 支持帕累托改进论点
- Table 28 (敏感性分析节选) — 验证稳健性

**大表（建议节选 + 附录）**：
- 34 赛季详细 τ 值 (`season_details.json`)：主文展示汇总统计，附录给出完整列表
- 完整参数网格搜索结果：主文展示最优点，附录给出全部结果

---

## 与其他章节的衔接

### 来自 Task 1 的输入
- 粉丝份额后验均值 $\hat{f}_{i,t}$ 作为 PTFS 计算的输入
- Judges' Save 失效的发现 (90.5% 救粉丝高者) 作为动机

### 来自 Task 3 的输入
- 运动员悖论 (β_J = -0.129 vs β_F = +0.052) 作为系统性偏差证据
- 舞伴 ICC 差异 (21.8% vs 9.5%) 作为粉丝偏见证据
- 年龄效应差异 (IDI = 17.19) 作为评委偏见证据

### 对后续 Memo 的输出
- 推荐采用 PTFS (45% → 80%)
- 核心证据：τ 提升 0.022 (p = 0.031)
- 实施建议：仅需修改得分公式

---

## 写作检查清单

- [ ] 确保所有数学符号与 Task 1/Task 3 一致（$f$, $J$, $Y$, $\tau$）
- [ ] 检查表格数据与 `outputs/*.json` 一致
- [ ] 制作 Fig. 6.1–6.4 后替换占位符
- [ ] 核对敏感性分析数据与 `sensitivity_analysis.json`
- [ ] 确认引用文献与 `references.bib` 的 key 一致
- [ ] 章节编号与主论文结构协调（假设为 Section 6）
- [ ] 检查 p 值和统计量与代码输出一致

---

## References

*见 references.bib 文件*
