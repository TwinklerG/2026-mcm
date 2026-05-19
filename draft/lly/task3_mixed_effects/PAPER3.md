# Task 3: Impact Analysis via Dual-Track Mixed-Effects Modeling

## 论文写作指南

> 本文件为问题三的论文部分，采用双语格式：英文为论文正文，中文为解释说明。本部分应作为论文的一个子章节（如 Section 5），而非独立成章。本章节依赖 Task 1 的粉丝份额估计结果。

---

## 5. Impact Analysis of Pro Dancers and Celebrity Characteristics

### 5.1 Research Question and Motivation

**【论文正文】**

A fundamental question posed in Problem C asks: *How much do [pro dancers and celebrity characteristics] impact how well a celebrity will do in the competition? Do they impact judges' scores and fan votes in the same way?*

This question probes beyond aggregate performance to examine the structural determinants of success in DWTS. We develop a **Dual-Track Hierarchical Mixed-Effects Model (DT-HMEM)** that simultaneously models two distinct outcome pathways:

1. **Track 1 (Performance)**: Judges' scores as a measure of technical dance quality
2. **Track 2 (Popularity)**: Fan vote shares as a measure of audience appeal

By explicitly modeling these dual tracks with shared hierarchical structure, we can directly test whether the same factors influence both systems identically or divergently—a question with significant implications for understanding the tension between meritocratic and populist dynamics in the competition.

**【中文解释】**

> **研究动机**：问题三要求分析职业舞伴和明星特征对比赛成绩的影响，并特别追问"它们对评委分和粉丝票的影响是否相同"。这是一个关于**双系统一致性**的实证问题。我们的核心发现是：**No, they do NOT impact in the same way.**——相同特征可能对两个系统产生方向相反的影响。

---

### 5.2 Model Specification

**【论文正文】**

We formulate a hierarchical structure where observations (contestant-week pairs) are nested within professional dancers. Let $i$ index contestants, $t$ index weeks, and $p(i)$ denote the professional partner assigned to contestant $i$.

#### 5.2.1 Track 1: Performance Model (Judge Scores)

The judge score model captures technical assessment:

$$Y_{i,t}^{J} = \beta_0^J + \beta_{age}^J \cdot \text{Age}_{i}^{c} + \beta_{week}^J \cdot t + \sum_{k \in \mathcal{K}} \beta_k^J \cdot \mathbb{I}[\text{Industry}_i = k] + u_{p(i)}^J + \epsilon_{i,t}^J$$

where:
- $Y_{i,t}^{J}$ is the average judge score for contestant $i$ in week $t$
- $\text{Age}_i^c = \text{Age}_i - \bar{\text{Age}}$ is mean-centered age
- $\mathcal{K} = \{\text{Athlete}, \text{Entertainment}, \text{Model}, \text{Other}, \text{Singer}, \text{TV}\}$ with Actor/Actress as the reference category
- $u_{p(i)}^J \sim \mathcal{N}(0, \sigma_{pro,J}^2)$ is the professional dancer random intercept
- $\epsilon_{i,t}^J \sim \mathcal{N}(0, \sigma_{\epsilon,J}^2)$ is the residual error

#### 5.2.2 Track 2: Popularity Model (Fan Shares)

The fan share model incorporates the posterior mean estimates $\hat{f}_{i,t}$ from Task 1:

$$Y_{i,t}^{F} = \beta_0^F + \beta_{age}^F \cdot \text{Age}_{i}^{c} + \beta_{week}^F \cdot t + \gamma \cdot Y_{i,t}^{J} + \sum_{k \in \mathcal{K}} \beta_k^F \cdot \mathbb{I}[\text{Industry}_i = k] + u_{p(i)}^F + \epsilon_{i,t}^F$$

where:
- $Y_{i,t}^{F} = \ln(\hat{f}_{i,t} + \epsilon)$ is the log-transformed fan share estimate
- $\gamma$ is the **performance-to-popularity conversion coefficient**, measuring how strongly current-week judge scores predict fan voting
- Other terms parallel Track 1

**【中文解释】**

> **模型设计的核心思路**：
> 1. **双轨并行**：评委分和粉丝票分别建模，但使用相同的固定效应结构
> 2. **职业舞伴随机效应**：通过 $u_p$ 捕捉"跟谁跳舞"带来的系统性优势/劣势
> 3. **关键参数 $\gamma$**：衡量粉丝是否"看表现投票"——如果 $\gamma \approx 0$，说明粉丝投票与当晚表现无关
> 4. **基准组设定**：演员（Actor/Actress）作为参照组，其他行业的系数表示相对于演员的差异

---

### 5.3 Data and Implementation

**【论文正文】**

**Table 13: Dataset Characteristics for Mixed-Effects Analysis**

| Characteristic                          | Value       |
| --------------------------------------- | ----------- |
| Total observations                      | 2,738       |
| Unique contestants                      | 411         |
| Seasons covered                         | 34          |
| Professional dancers (total)            | 60          |
| Professional dancers (≥10 observations) | 42          |
| Industry categories                     | 7           |
| Age range                               | 14–82 years |
| Week range                              | 1–11        |

We restrict the random effects analysis to professional dancers with at least 10 observations to ensure stable variance component estimation @snijders2012multilevel. The industry distribution of observations is: Actor (875), Athlete (642), TV (585), Singer (400), Entertainment (89), Model (76), Other (71).

Models are estimated using restricted maximum likelihood (REML) via the statsmodels Python package @seabold2010statsmodels, which implements the mixed model equations derived by @henderson1953estimation and elaborated in @laird1982random.

**【中文解释】**

> **数据说明**：
> - 2738 个观测来自 411 位选手在 34 个赛季的表现
> - 60 位职业舞伴中，仅保留有 ≥10 次观测的 42 位进行随机效应分析（样本量过小会导致方差估计不稳定）
> - 演员和运动员是最大的两个行业类别

---

### 5.4 Fixed Effects Estimates

**【论文正文】**

**Table 14: Fixed Effects Comparison Between Judge Score and Fan Share Models**

| Variable               | $\hat{\beta}^J$ | SE$^J$ | $p^J$ | $\hat{\beta}^F$ | SE$^F$ | $p^F$ |
| ---------------------- | --------------- | ------ | ----- | --------------- | ------ | ----- |
| Intercept              | 6.580           | 0.069  | <.001 | −2.737          | 0.038  | <.001 |
| Age (centered)         | **−0.033**      | 0.002  | <.001 | −0.002          | 0.000  | <.001 |
| Week                   | **0.311**       | 0.007  | <.001 | 0.114           | 0.002  | <.001 |
| Athlete                | **−0.129**      | 0.054  | .017  | **+0.052**      | 0.013  | <.001 |
| Entertainment          | −0.223          | 0.119  | .060  | −0.125          | 0.028  | <.001 |
| Model                  | −0.535          | 0.130  | <.001 | −0.217          | 0.031  | <.001 |
| Other                  | −0.689          | 0.131  | <.001 | −0.246          | 0.032  | <.001 |
| Singer                 | −0.021          | 0.061  | .731  | −0.019          | 0.015  | .203  |
| TV                     | −0.419          | 0.055  | <.001 | −0.033          | 0.013  | .012  |
| Judge Score ($\gamma$) | —               | —      | —     | 0.004           | 0.005  | .422  |

Note: Bold values highlight substantively important findings. Reference category is Actor/Actress.

The performance-to-popularity conversion coefficient $\gamma = 0.004$ is **not statistically significant** ($p = 0.42$), suggesting that fan voting behavior is largely independent of same-week judge scores when controlling for temporal trends.

**【中文解释】**

> **核心发现**：
> 1. **年龄效应差异巨大**：评委每增加 1 岁扣 0.033 分，而粉丝几乎不在意年龄（−0.002）
> 2. **运动员悖论**：评委认为运动员技术较差（−0.129），但粉丝更支持运动员（+0.052）——**方向完全相反**
> 3. **$\gamma$ 不显著**：表面上看，粉丝投票与当晚表现无关；但这一结论需要进一步诊断（见敏感性分析）

---

### 5.5 Impact Divergence Index (IDI)

**【论文正文】**

To formally test whether each covariate impacts the two systems differently, we construct the **Impact Divergence Index**:

$$\text{IDI}_k = \frac{|\hat{\beta}_k^J - \hat{\beta}_k^F|}{\sqrt{\text{SE}(\hat{\beta}_k^J)^2 + \text{SE}(\hat{\beta}_k^F)^2}}$$

Under the null hypothesis of equal effects, IDI follows an approximate standard normal distribution. Values exceeding 1.96 indicate significant divergence at the 5% level; values exceeding 3.0 suggest substantively important differences.

**Table 15: Impact Divergence Index for All Covariates**

| Variable       | $\beta^J$ | $\beta^F$ | Difference | IDI       | $p$-value | Significant |
| -------------- | --------- | --------- | ---------- | --------- | --------- | ----------- |
| Week           | 0.311     | 0.114     | 0.198      | **27.26** | <.001     | ✓           |
| Age (centered) | −0.033    | −0.002    | −0.031     | **17.19** | <.001     | ✓           |
| TV             | −0.419    | −0.033    | −0.386     | **6.79**  | <.001     | ✓           |
| Athlete        | −0.129    | +0.052    | −0.181     | **3.26**  | .001      | ✓           |
| Other          | −0.689    | −0.246    | −0.444     | **3.29**  | .001      | ✓           |
| Model          | −0.535    | −0.217    | −0.318     | 2.37      | .018      | ✓           |
| Entertainment  | −0.223    | −0.125    | −0.098     | 0.80      | .422      | ✗           |
| Singer         | −0.021    | −0.019    | −0.002     | 0.04      | .969      | ✗           |

**Figure 5.1: Impact Divergence Visualization** *(建议制图)*

> **图表说明**：条形图，x轴为变量名，y轴为 IDI 值。水平虚线标注 IDI = 1.96（显著阈值）和 IDI = 3.0（实质性差异阈值）。不同颜色区分显著/不显著变量。

**【中文解释】**

> **IDI 解读**：
> - **Week (IDI=27.26)**：评委对持续比赛的奖励幅度是粉丝的 2.7 倍
> - **Age (IDI=17.19)**：年龄对评委的影响是粉丝的 18 倍
> - **Athlete (IDI=3.26)**：唯一出现**方向逆转**的类别——评委负、粉丝正
> - Singer 和 Entertainment 的 IDI < 1.96，说明这两类行业在两系统中的影响没有显著差异

---

### 5.6 The Athlete Paradox

**【论文正文】**

The most striking finding is the **Athlete Paradox**: athletes receive systematically lower judge scores ($\beta^J = -0.129$, $p = .017$) yet higher fan vote shares ($\beta^F = +0.052$, $p < .001$) compared to the Actor/Actress baseline.

**Table 16: The Athlete Paradox in Detail**

| Outcome          | Coefficient | Direction    | Significance |
| ---------------- | ----------- | ------------ | ------------ |
| Judge Score      | −0.129      | Negative     | *            |
| Fan Share        | +0.052      | **Positive** | ***          |
| Divergence (IDI) | 3.26        | —            | ***          |

This divergence is substantively meaningful: athletes, despite being penalized by expert judges for presumably weaker technical skills, are rewarded by fan voters. This finding has direct implications for understanding DWTS controversies (Task 2)—athletes like Jerry Rice, Billy Ray Cyrus, and Bobby Bones thrived on fan support despite consistently low judge scores.

The paradox likely reflects differing evaluation criteria: judges assess technical dance quality, while fans respond to charisma, athletic physique, competitive spirit, and existing celebrity recognition from sports fandom @mccutcheon2002celebrity.

**【中文解释】**

> **运动员悖论的实践意义**：
> - 这解释了为什么运动员是 DWTS 历史上最具争议的群体
> - Jerry Rice (S2), Billy Ray Cyrus (S4), Bristol Palin (S11), Bobby Bones (S27) 都是粉丝主导导致评委"失控"的典型案例
> - **选角启示**：运动员是"粉丝收割机"但"评委杀手"

---

### 5.7 Variance Decomposition and ICC

**【论文正文】**

The **Intraclass Correlation Coefficient (ICC)** quantifies the proportion of total variance attributable to professional dancer effects:

$$\text{ICC} = \frac{\sigma_{pro}^2}{\sigma_{pro}^2 + \sigma_{\epsilon}^2}$$

**Table 17: Variance Decomposition**

| Model       | Pro Variance ($\sigma_{pro}^2$) | Residual Variance ($\sigma_{\epsilon}^2$) | Total Variance | ICC        |
| ----------- | ------------------------------- | ----------------------------------------- | -------------- | ---------- |
| Judge Score | 0.095                           | 0.907                                     | 1.002          | **9.49%**  |
| Fan Share   | 0.014                           | 0.051                                     | 0.065          | **21.77%** |

A critical finding emerges: the professional dancer explains **2.3 times more variance** in fan voting (21.8%) than in judge scoring (9.5%). This suggests that "who you dance with" matters far more for fan appeal than for technical assessment.

**Interpretation**: Judges appear to evaluate contestants' performances relatively independently of partner identity, consistent with their role as technical experts. In contrast, fans' voting decisions are substantially influenced by professional dancer associations—likely through name recognition, social media presence, and perceived teaching ability of the pro @driessens2013celebrity.

**Figure 5.2: ICC Comparison** *(建议制图)*

> **图表说明**：两组堆叠柱状图，分别展示评委模型和粉丝模型的方差分解。每个柱状图包含两部分：职业舞伴方差（深色）和残差方差（浅色）。标注 ICC 百分比。

**【中文解释】**

> **方差分解的关键洞见**：
> - 评委评分的 ICC = 9.5%：舞伴身份对评委分的影响相对有限
> - 粉丝投票的 ICC = 21.8%：超过五分之一的粉丝投票方差可由"跟谁跳舞"解释
> - **政策启示**：如果要降低粉丝投票的"身份偏见"，需要削弱职业舞伴的明星效应

---

### 5.8 Professional Dancer Effect Analysis

**【论文正文】**

Using Best Linear Unbiased Predictors (BLUPs), we extract individual professional dancer effects from both models. We define:
- **Tech Boost**: $\hat{u}_p^J$ — the random intercept from the judge model
- **Pop Boost**: $\hat{u}_p^F$ — the random intercept from the fan model

Standardizing these effects allows classification into four quadrants:

**Table 18: Professional Dancer Typology**

| Type               | Definition                      | Count | Characteristics                       |
| ------------------ | ------------------------------- | ----- | ------------------------------------- |
| **Kingmaker**      | Tech$_z > 0$, Pop$_z > 0$       | 8     | Elevates both scores and votes        |
| **Technician**     | Tech$_z > 0$, Pop$_z \leq 0$    | 13    | Improves dancing but not popularity   |
| **Fan Favorite**   | Tech$_z \leq 0$, Pop$_z > 0$    | 9     | Boosts votes despite weaker technique |
| **Underperformer** | Tech$_z \leq 0$, Pop$_z \leq 0$ | 12    | Negative impact on both dimensions    |

**Table 19: Top 5 Kingmakers (Combined Score)**

| Rank | Professional Dancer | Tech Boost | Pop Boost | Combined$^a$ |
| ---- | ------------------- | ---------- | --------- | ------------ |
| 1    | Derek Hough         | +0.656     | +0.060    | **1.46**     |
| 2    | Jonathan Roberts    | +0.149     | +0.180    | **1.07**     |
| 3    | Kym Johnson         | +0.213     | +0.103    | **0.84**     |
| 4    | Tony Dovolani       | +0.286     | +0.072    | **0.84**     |
| 5    | Maksim Chmerkovskiy | +0.299     | +0.061    | **0.82**     |

$^a$ Combined = (Tech$_z$ + Pop$_z$) / 2

**Table 20: Top 5 Technical Specialists**

| Rank | Professional Dancer   | Tech Boost |
| ---- | --------------------- | ---------- |
| 1    | Derek Hough           | +0.656     |
| 2    | Artem Chigvintsev     | +0.429     |
| 3    | Witney Carson         | +0.329     |
| 4    | Valentin Chmerkovskiy | +0.325     |
| 5    | Maksim Chmerkovskiy   | +0.299     |

**Table 21: Top 5 Popularity Boosters**

| Rank | Professional Dancer | Pop Boost |
| ---- | ------------------- | --------- |
| 1    | Ashly DelGrosso     | +0.373    |
| 2    | Alec Mazo           | +0.203    |
| 3    | Jonathan Roberts    | +0.180    |
| 4    | Edyta Sliwinska     | +0.151    |
| 5    | Louis van Amstel    | +0.111    |

**Figure 5.3: Professional Dancer Quadrant Classification** *(建议制图)*

> **图表说明**：散点图，x轴为 Tech Boost (标准化)，y轴为 Pop Boost (标准化)。四个象限分别用不同颜色标注。关键舞伴（如 Derek Hough）单独标注名字。垂直和水平虚线在零点标注象限边界。

**【中文解释】**

> **职业舞伴分类的实践价值**：
> - **Kingmaker（造王者）**：Derek Hough 是最成功的舞伴，同时提升技术分和粉丝票
> - **Technician（技术型）**：Artem Chigvintsev 等擅长教舞，但缺乏粉丝号召力
> - **Fan Favorite（流量型）**：Ashly DelGrosso 的流量加成是所有舞伴中最高的（+0.373）
> - **选角建议**：如果制片方希望选手获胜，应配对 Kingmaker；如果需要争议话题，可配对 Fan Favorite

---

### 5.9 Experience-Effect Correlation

**【论文正文】**

We test whether professional dancer effects correlate with experience (number of seasons appeared):

**Table 22: Experience-Effect Correlation Analysis**

| Relationship              | Correlation | $p$-value | Significant |
| ------------------------- | ----------- | --------- | ----------- |
| Tech Boost vs. Experience | 0.250       | 0.111     | ✗           |
| Pop Boost vs. Experience  | 0.205       | 0.192     | ✗           |

Neither correlation reaches statistical significance, suggesting that professional dancer effects arise primarily from **individual characteristics** (teaching style, personality, physical appearance) rather than accumulated experience on the show. This finding is consistent with talent-based theories of performance @ericsson2016expertise but inconsistent with pure learning-by-doing explanations.

**【中文解释】**

> **经验-效应相关性分析**：
> - 职业舞伴的经验（参赛次数）与其技术加成或流量加成均无显著相关
> - **结论**：舞伴的"能力"更多来自个人特质（教学天赋、个人魅力），而非经验积累
> - 这对制片方有启示：老舞伴不一定更好，关键是匹配合适的人

---

### 5.10 Sensitivity Analysis: The $\gamma$ Puzzle

**【论文正文】**

The non-significance of $\gamma$ (performance-to-popularity conversion) initially suggests that fans vote independently of same-week judge scores. However, this result warrants scrutiny due to potential multicollinearity between `week` and `judge_score`—later weeks have both higher average judge scores (skill improvement) and fewer contestants (survival bias).

We conduct a sensitivity analysis with three model specifications:

**Table 23: $\gamma$ Sensitivity Analysis**

| Model Specification                 | $\hat{\gamma}$ | $p$-value | Significant |
| ----------------------------------- | -------------- | --------- | ----------- |
| Original (week + judge_score)       | 0.004          | 0.422     | No          |
| Without week                        | **0.161**      | <.001     | **Yes**     |
| With week × judge_score interaction | −0.040         | <.001     | Yes         |
| Interaction term                    | +0.012         | <.001     | Yes         |

**Key Finding**: Removing `week` from the model increases $\gamma$ from 0.004 to **0.161** and renders it highly significant. This confirms **multicollinearity**: the `week` variable absorbs explanatory power that would otherwise be attributed to judge scores.

**Corrected Interpretation**: Fans do value performance, but they conflate *longevity* (surviving to later weeks) with *quality* (high judge scores). The significant interaction term ($+0.012$, $p < .001$) indicates that judge scores become increasingly predictive of fan voting as the competition progresses—consistent with fans paying more attention to technical merit in later rounds.

**【中文解释】**

> **$\gamma$ 敏感性分析的核心发现**：
> 1. **原始结论有误导性**：$\gamma = 0.004$ (ns) 不意味着粉丝不看表现
> 2. **多重共线性确认**：移除 `week` 后，$\gamma$ 从 0.004 跳至 0.161（显著）
> 3. **修正后的解读**："Fans conflate longevity with quality"——粉丝将"活得久"等同于"跳得好"
> 4. **交互效应**：后期比赛中，评委分对粉丝投票的影响更强（可能是因为后期粉丝更认真看比赛）

---

### 5.11 Compositional Data Diagnostics

**【论文正文】**

Since fan shares sum to unity within each week, they constitute compositional data with induced negative correlations among contestants @aitchison1982statistical. Under this constraint, the theoretical within-week correlation is:

$$\rho_{theoretical} = -\frac{1}{n-1}$$

where $n$ is the number of contestants in a given week.

**Table 24: Compositional Data Diagnostic Results**

| Diagnostic                       | Value   |
| -------------------------------- | ------- |
| Mean contestants per week        | 7.92    |
| Theoretical negative correlation | −0.145  |
| Empirical residual correlation   | ≈ 0     |
| SE inflation factor              | 1.0     |
| Concern level                    | **Low** |

The empirical residual correlation (≈0) is weaker than the theoretical expectation (−0.145), indicating that the compositional constraint is largely absorbed by the model structure. We conclude that standard mixed-effects inference remains valid without requiring specialized compositional data methods such as Dirichlet regression @hijazi2009modelling.

**【中文解释】**

> **成分数据诊断**：
> - 由于粉丝份额每周总和为 1，理论上存在结构性负相关（选手之间此消彼长）
> - 但实证残差相关接近 0，说明模型已充分吸收这一约束
> - **结论**：无需使用 Dirichlet 回归等专门方法，标准混合效应模型的推断仍然有效

---

### 5.12 Answer to Problem 3

**【论文正文】**

We now directly address the research question: *Do pro dancers and celebrity characteristics impact judges' scores and fan votes in the same way?*

**Answer: No.**

Our analysis reveals systematic and substantively important divergences:

1. **The Athlete Paradox**: Athletes receive lower judge scores ($\beta^J = -0.129$) but higher fan votes ($\beta^F = +0.052$)—the only industry category with opposite-signed effects.

2. **Differential Age Penalty**: Age decreases judge scores 18 times more than fan votes (IDI = 17.19), reflecting judges' technical bias against older contestants that fans do not share.

3. **Professional Dancer Influence**: Pro dancers explain 2.3× more variance in fan voting (ICC = 21.8%) than in judge scoring (ICC = 9.5%), indicating that "who you dance with" matters far more for popularity than for technical assessment.

4. **Performance-Longevity Conflation**: Fans implicitly reward survival to later rounds ($\gamma$ sensitive to week inclusion), conflating competitive longevity with current-week performance quality.

These findings explain the structural tension that produces DWTS controversies: contestants can systematically outperform on one track while underperforming on the other, creating misalignment between technical merit and ultimate competitive outcomes.

**【中文解释】**

> **对问题三的直接回答**：
> - **核心结论**：职业舞伴和明星特征对评委分和粉丝票的影响**不相同**
> - **四大证据**：
>   1. 运动员悖论（方向相反）
>   2. 年龄效应差 18 倍
>   3. 舞伴 ICC 差 2.3 倍
>   4. 粉丝混淆"活得久"和"跳得好"
> - **政策启示**：这些发现为 Task 4 的公平系统设计提供了实证基础

---

## Figures and Tables Summary

### 建议图表清单

| 图表编号    | 类型              | 内容             | 用途               |
| ----------- | ----------------- | ---------------- | ------------------ |
| Fig. 5.1    | Bar Chart         | IDI 值按变量排序 | 展示双系统差异程度 |
| Fig. 5.2    | Stacked Bar       | 方差分解对比     | 直观展示 ICC 差异  |
| Fig. 5.3    | Scatter Plot      | 舞伴四象限分类   | 可视化舞伴类型     |
| Table 14    | Coefficient Table | 固定效应对比     | 核心结果展示       |
| Table 15    | IDI Table         | 影响差异度指数   | 统计显著性检验     |
| Table 17    | Variance Table    | ICC 和方差分解   | 随机效应重要性     |
| Table 19–21 | Top-k Rankings    | 舞伴排名         | 实践应用价值       |

### 数据展示建议

**小表（可完整纳入论文）**：
- Table 13 (数据集特征)
- Table 14 (固定效应对比) — 核心结果
- Table 15 (IDI 指标)
- Table 16 (运动员悖论)
- Table 17 (方差分解)
- Table 22 (经验-效应相关)
- Table 23 ($\gamma$ 敏感性分析)
- Table 24 (成分数据诊断)

**大表（建议节选 + 附录）**：
- 完整的 42 位职业舞伴排名表 (`pro_rankings_full.csv`)：主文展示 Top 5 / Bottom 5，附录给出完整列表
- 详细固定效应表（含所有行业）：主文已包含完整版

---

## References

*见 references.bib 文件*

---

## 写作检查清单

- [ ] 确保所有数学符号与 Task 1 一致（$f$, $J$, $Y$）
- [ ] 检查表格数据与 outputs/ 中的 JSON/CSV 一致
- [ ] 制作 Fig. 5.1–5.3 后替换占位符
- [ ] 核对 IDI 计算公式与代码一致
- [ ] 确认 $\gamma$ 敏感性分析的三个模型公式正确
- [ ] 引用文献与 references.bib 的 key 一致
- [ ] 章节编号与主论文结构协调（假设为 Section 5）

---

## 与其他章节的衔接

### 来自 Task 1 的输入
- 粉丝份额后验均值 $\hat{f}_{i,t}$ 作为 Track 2 的因变量
- 验证了粉丝份额估计的可靠性（93.49% 重构准确率）

### 为 Task 4 提供的输出
- 运动员悖论揭示了投票规则的系统性偏差
- 职业舞伴 ICC 差异为动态权重设计提供依据
- 年龄效应差异支持了针对性公平校正的必要性

