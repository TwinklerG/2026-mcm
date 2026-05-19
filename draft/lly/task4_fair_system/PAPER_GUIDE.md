# Task 4: PAPER_GUIDE.md - 论文写作指南

## 问题回应

### 原题要求

> **Propose another system using fan votes and judge scores each week that you believe is more "fair" (or "better" in some other way such as making the show more exciting for the fans). Provide support for why your approach should be adopted by the show producers.**

### 核心论点

我们提出的 **渐进技术公平系统 (Progressive Technical Fairness System, PTFS)** 通过以下方式回应问题：

1. **"More Fair"**：Kendall's τ 从 0.727 (Percentage) 提升至 0.749 (p=0.031)，技术能力与最终排名的相关性显著提高
2. **"Better for Excitement"**：Close Call Rate 从 66% 提升至 79%，更多悬念时刻
3. **"Provide Support"**：基于 34 个赛季的全数据统计分析，配对 t 检验验证显著性

---

## 论文结构建议

### 4.1 引言（Introduction to Task 4）

**关键内容**：
- 承接 Task 1-3 的发现，动机明确
- 提出核心问题：如何平衡技术公平与观众参与度？

**建议写法**：
> Building on our findings from Tasks 1-3, we identified a fundamental tension in DWTS voting systems: maximizing technical fairness (rewarding skilled dancers) can reduce audience engagement (predictable outcomes), while prioritizing excitement may allow technically inferior contestants to advance unfairly.
>
> Task 1 revealed that 90.5% of "Judges' Save" decisions favored contestants with higher estimated fan shares, suggesting the existing fairness mechanism is ineffective. Task 3 showed systematic biases—the "Athlete Paradox" where judges penalize athletes while fans reward them (β_judge = -0.129 vs β_fan = +0.052)—indicating that neither pure judge scores nor pure fan votes alone can achieve fairness.

### 4.2 模型设计（Model Design）

**核心公式**（务必包含）：

$$S_{i,t} = w_J(t) \cdot \tilde{J}_{i,t} + (1 - w_J(t)) \cdot \tilde{F}_{i,t}$$

$$w_J(t) = 0.45 + 0.35 \cdot \frac{t-1}{T-1}$$

**设计理念表述**：
> The progressive weight adjustment reflects a principled tradeoff:
> - **Early weeks** (w_J = 45%): Emphasize fan engagement to maintain viewership and allow underdogs to survive
> - **Late weeks** (w_J = 80%): Shift toward technical merit to ensure finals are contested by skilled dancers

**与 Task 1/3 的联系**（必须强调）：
> - Fan share estimates from Task 1's Bayesian MCMC model provide the $\tilde{F}_{i,t}$ input
> - The progressive shift toward judge weights addresses the biases identified in Task 3, where fan preferences diverge from technical assessment

### 4.3 评价指标（Evaluation Metrics）

**关键指标解释**：

| Metric                 | Definition                                                      | Interpretation                                    |
| ---------------------- | --------------------------------------------------------------- | ------------------------------------------------- |
| Kendall's τ            | Correlation between cumulative judge ranks and final placements | Higher = technically better dancers finish higher |
| Tech Top3 → Final Top3 | % of technical top-3 who finish in final top-3                  | Higher = rewards skill                            |
| Tech Lowest Eliminated | % of weeks where technical lowest is eliminated                 | Higher = fair elimination                         |
| Close Call Rate        | % of eliminations with <2% score gap                            | Higher = more suspense                            |

**为什么不用"争议案例修正率"**：
> We deliberately avoid metrics that target specific controversial cases (e.g., Bobby Bones, Bristol Palin) as this would constitute overfitting to historical anomalies rather than measuring genuine system improvement.

### 4.4 实验结果（Results）

**主表格（Table X）**：

| System     | Kendall's τ | Tech Top3→Final Top3 | Mean Deviation | Close Call Rate |
| ---------- | ----------- | -------------------- | -------------- | --------------- |
| Rank-based | 0.674       | 59.8%                | 1.61           | 30.2%           |
| Percentage | 0.727       | 71.6%                | 1.35           | 66.1%           |
| **PTFS**   | **0.749**   | **72.5%**            | **1.26**       | **79.1%**       |

**统计显著性**：
> Paired t-test comparing PTFS vs Percentage across 34 seasons: t = 2.26, p = 0.031 (significant at α = 0.05)

### 4.5 权衡分析（Tradeoff Analysis）

**核心论述**：
> No voting system can simultaneously maximize all desirable properties. PTFS represents a principled tradeoff:
>
> | Dimension | PTFS vs Percentage | Interpretation |
> |-----------|-------------------|----------------|
> | Technical Fairness (τ) | **+0.022** | Significant improvement in skill-outcome correlation |
> | Consistency | **+2.9%** | 97% of seasons achieve τ > 0.5 vs 94% |
> | Excitement | **+13%** | Close calls increase from 66% to 79% |
> | Upset Rate | **-1%** | Slightly fewer "unfair" eliminations |

**图表建议**（引用 figures/）：
1. `tau_by_season.png`：展示 PTFS 在大多数赛季的改进
2. `improvement_distribution.png`：展示改进分布和比例
3. `dynamic_weights.png`：解释渐进权重机制

### 4.6 对制片人的建议（Memo Content）

**核心建议**：
> **Recommendation**: Adopt PTFS for future DWTS seasons.
>
> **Rationale**:
> 1. **Improved Fairness**: Technical skill is 3% more correlated with final placement (statistically significant, p=0.031)
> 2. **Enhanced Excitement**: 13% more "close call" eliminations maintain viewer engagement
> 3. **Reduced Controversy**: Higher correlation between technical merit and outcomes preempts fan complaints about "undeserving" winners
> 4. **Easy Implementation**: Simply adjust the weight formula—no infrastructure changes needed

---

## 关键图表使用指南

| Figure                         | Purpose                               | Where to Use                     |
| ------------------------------ | ------------------------------------- | -------------------------------- |
| `tau_by_season.png`            | Show PTFS outperforms in most seasons | Results section, main evidence   |
| `improvement_distribution.png` | Show 70%+ seasons improved            | Support statistical significance |
| `metrics_comparison.png`       | Compare all key metrics               | Results table companion          |
| `dynamic_weights.png`          | Explain PTFS mechanism                | Model design section             |
| `engagement_comparison.png`    | Show excitement metrics               | Tradeoff analysis                |
| `statistical_summary.png`      | Tabular p-values                      | Appendix or inline               |

---

## 常见问题回应

### Q: 为什么 PTFS 的技术冠军率 (44.1%) 低于 Rank-based (47.1%)？

> While PTFS has a slightly lower "tech winner rate" than Rank-based, it achieves significantly higher overall correlation (τ=0.749 vs 0.674). This reflects PTFS's balanced approach: rather than hyper-focusing on crowning the #1 technical dancer, it ensures the entire ranking better reflects technical merit while maintaining competitive finals.

### Q: Bottom25→Top3 率为什么都是 0%？

> This indicates that across all 34 seasons, no contestant in the technical bottom quartile reached the final top-3 under any system. The absence of extreme anomalies in our simulated results reflects the effectiveness of combining judge scores with fan votes—even historical controversies like Bobby Bones (S27 winner) were identified as technical bottom-quartile only in weekly snapshots, not in cumulative season rankings.

### Q: 为什么不直接用 100% 评委分？

> Pure judge scores would eliminate fan engagement entirely, transforming DWTS into a judged competition like Olympic figure skating. Our Task 3 analysis showed that pro dancer effects and systematic biases in judge scores (e.g., against athletes) mean judges are not perfectly objective arbiters. PTFS balances both perspectives while progressively favoring technical merit.

---

## 数据支持清单

论文中需要引用以下输出文件：

1. **`optimal_params.json`**：最优参数 (w_start=0.45, w_end=0.80)
2. **`detailed_metrics.json`**：所有系统的详细指标
3. **`season_details.json`**：34 赛季的逐赛季 τ 值（用于配对 t 检验）
4. **`system_comparison.json`**：系统对比汇总
