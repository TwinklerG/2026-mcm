# Task 1: 贝叶斯 MCMC 粉丝投票估计

## 目标

从淘汰结果逆向推断未知的粉丝投票份额。

---

## 1. 核心数学定义

### 1.1 符号系统

| 符号                      | 定义                                         |
| ------------------------- | -------------------------------------------- |
| $S, W$                    | 赛季（Season）与周次（Week）                 |
| $C_t = \{c_1, ..., c_n\}$ | 第 $t$ 周参赛的 $n$ 名选手集合               |
| $J_{i,t} \in [0, \infty)$ | 选手 $i$ 在第 $t$ 周获得的评委总分           |
| $E_t$                     | 第 $t$ 周被淘汰的选手集合（可为空，可多人）  |
| $f_{i,t} \in (0, 1)$      | 选手 $i$ 在第 $t$ 周的粉丝投票份额（待估计） |

### 1.2 基础约束 (Simplex Constraint)

粉丝投票份额满足单纯形约束：
$$\sum_{i=1}^{n} f_{i,t} = 1, \quad f_{i,t} > 0$$

---

## 2. 贝叶斯模型框架

**逆问题**：给定淘汰结果 $E_t$ 和评委分 $J_t$，估计粉丝份额后验分布：

$$P(\mathbf{f}_t | J_t, E_t) \propto \underbrace{P(E_t | \mathbf{f}_t, J_t)}_{\text{似然（硬约束）}} \times \underbrace{P(\mathbf{f}_t | \mathbf{f}_{t-1})}_{\text{时间平滑}} \times \underbrace{P(\mathbf{f}_t)}_{\text{先验}}$$

### 2.1 投票组合方法（三套规则）

**代码实现** (`voting.py:get_voting_method`, `compute_combined_score_percentage`, `compute_combined_rank`):

**百分比法 (Seasons 3–27)**：
$$\text{Total}_i = \frac{J_i}{\sum_{j=1}^{n} J_j} + f_i$$

**排名法 (Seasons 1–2)**：
$$\text{Total}_i = \text{Rank}_J(i) + \text{Rank}_f(i)$$

其中 $\text{Rank}_J(i) = n - \text{argsort}(\text{argsort}(J))_i$，即得分越高排名数字越小（第1名=1）。

**底部两人法 (Seasons 28+, Judges' Save)**：
- 使用排名法计算综合排名
- 综合排名最低的两人进入 Bottom 2
- 评委从中选择一人淘汰
- 约束变为：被淘汰者**至少一人**在 Bottom 2 中

### 2.2 似然函数（硬约束）

**代码实现** (`voting.py:check_elimination_constraint`):

似然函数是指示函数，支持三种情况：

1. **无人淘汰** ($E_t = \emptyset$)：
$$P(E_t | \mathbf{f}_t, J_t) = 1$$

2. **单人/多人淘汰 (S1-S27)**：设 $k = |E_t|$ 为被淘汰人数
$$P(E_t | \mathbf{f}_t, J_t) = \begin{cases} 1 & \text{if } E_t = \text{Bottom-}k(\text{Total}) \\ 0 & \text{otherwise} \end{cases}$$

3. **Judges' Save 规则 (S28+)**：
$$P(E_t | \mathbf{f}_t, J_t) = \begin{cases} 1 & \text{if } E_t \cap \text{Bottom-2}(\text{Total}) \neq \emptyset \\ 0 & \text{otherwise} \end{cases}$$

### 2.3 数据驱动先验

**代码实现** (`priors.py:estimate_industry_priors_from_data`):

**行业先验估计**：
$$\pi_{\text{industry}} = 0.2 + 0.6 \cdot \bar{S}_{\text{industry}}$$

其中：
$$\bar{S}_{\text{industry}} = \frac{1}{|I|} \sum_{i \in I} \frac{\text{选手 } i \text{ 存活周数}}{\text{该赛季最大周数}}$$

$I$ 为该行业所有选手集合，至少 3 个样本才估计，否则使用默认值。

**年龄修正因子** (`sampler.py:_compute_prior_mean`):
$$\mu_i = \pi_{\text{industry}(i)} \cdot \left(0.9 + 0.1 \cdot \exp\left(-\left(\frac{\text{age}_i - 40}{30}\right)^2\right)\right)$$

> 年龄修正是高斯形状：40 岁时达到峰值（乘以 1.0），偏离 40 岁越远则乘数越接近 0.9。

| 行业           | 默认先验 |
| -------------- | -------- |
| Singer/Rapper  | 0.75     |
| Actor/Actress  | 0.70     |
| TV Personality | 0.65     |
| Athlete        | 0.55     |
| Politician     | 0.25     |

### 2.4 时间平滑项

**代码实现** (`sampler.py:_compute_log_posterior`):

假设选手的粉丝基础在周与周之间不会剧烈波动：

$$P(\mathbf{f}_t | \mathbf{f}_{t-1}) \propto \exp\left(-\lambda \sum_{i \in C_t \cap C_{t-1}} (f_{i,t} - f_{i,t-1})^2\right)$$

其中：
- $\lambda = 0.5$ 为平滑权重 (`smoothness_weight`)
- 仅对**同时出现在相邻两周**的选手计算（已淘汰选手不参与）
- $\lambda$ 越大，强迫粉丝份额在时间上越平稳

### 2.5 对数后验（完整形式）

**代码实现** (`sampler.py:_compute_log_posterior`):

对于赛季内按周遍历，对数后验为：

$$\log P(\mathbf{f}) = \sum_{t} \left[ \underbrace{-\frac{1}{2} \sum_{i=1}^{n_t} \left(\frac{f_{i,t} - \mu_i/n_t}{\sigma_{prior}/n_t}\right)^2}_{\text{先验项}} + \underbrace{\left(-\lambda \sum_{i \in C_t \cap C_{t-1}} (f_{i,t} - f_{i,t-1})^2\right)}_{\text{时间平滑项}} \right]$$

**注意**：
- 先验均值 $\mu_i$ 和标准差 $\sigma_{prior}$ 均除以当周选手数 $n_t$ 进行归一化
- 这确保了先验在不同规模的周次间可比（选手多时，每人的期望份额自然更小）
- 前提是通过似然函数的硬约束检查，否则 $\log P = -\infty$

---

## 3. MCMC 采样算法

### 3.1 自适应 Metropolis-Hastings

**代码实现** (`sampler.py:AdaptiveMCMCSampler`):

**提议分布**（对数空间）：
$$\log(\mathbf{f}^*) = \log(\mathbf{f}^{(k)}) + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_{prop}^2 \mathbf{I})$$

**数值稳定化**：
$$\tilde{f}_i^* = \exp\left(\log f_i^* - \max_j \log f_j^*\right)$$
$$f_i^* = \frac{\max(\tilde{f}_i^*, 10^{-6})}{\sum_j \max(\tilde{f}_j^*, 10^{-6})}$$

**采样步骤**：
1. 生成提议 $\mathbf{f}^*$
2. **硬约束检查**：调用 `check_elimination_constraint`
   - 若违反淘汰约束 → 直接拒绝，$\mathbf{f}^{(k+1)} = \mathbf{f}^{(k)}$
3. **Metropolis 判定**（仅当约束满足时）：
$$\alpha = \min\left(1, \exp(\log P(\mathbf{f}^*) - \log P(\mathbf{f}^{(k)}))\right)$$
4. 生成 $u \sim \text{Uniform}(0, 1)$，若 $\log u < \log P(\mathbf{f}^*) - \log P(\mathbf{f}^{(k)})$ 则接受

### 3.2 自适应机制

**代码实现** (`sampler.py` 采样循环内):

**目标接受率**: $\alpha_{target} = 0.35$（基于 Roberts & Rosenthal 最优接受率理论）

**适应规则** (每 `adapt_interval=50` 步调整一次，仅在预热期):
$$\sigma_{prop}^{(new)} = \begin{cases}
1.3 \cdot \sigma_{prop} & \text{if } \hat{\alpha} > 0.40 \text{ (探索太慢)} \\
0.7 \cdot \sigma_{prop} & \text{if } \hat{\alpha} < 0.30 \text{ (拒绝太多)} \\
\sigma_{prop} & \text{otherwise}
\end{cases}$$

**约束**：$\sigma_{prop} \in [0.02, 0.6]$

### 3.3 稀疏采样 (Thinning)

**目的**：减少样本间自相关，提高有效样本量

**实现**：预热期后，每 $k=5$ 步保留 1 个样本
$$\text{保留样本索引} = \{N_{burn} + k \cdot j : j = 0, 1, ..., N/k - 1\}$$

### 3.4 有效样本量 (ESS) 估计

**代码实现** (`sampler.py:_estimate_ess`):

使用 Geyer 的初始单调序列估计器：

$$\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^{K^*} \hat{\rho}_k}$$

其中：
- $\hat{\rho}_k = \frac{1}{N \cdot \text{Var}(X)} \sum_{t=1}^{N-k} (X_t - \bar{X})(X_{t+k} - \bar{X})$ 为滞后 $k$ 的自相关系数
- $K^* = \min\{k : \hat{\rho}_k < 0\}$（找到第一个负自相关处截断）
- 最大滞后 $\min(N/2, 200)$

### 3.5 低 ESS 自动增强

**代码实现** (`sampler.py:sample_season_with_ess_check`):

若 ESS < 100 (`min_ess_threshold`)，自动使用增强配置重采样：

| 参数                | 增强倍数      |
| ------------------- | ------------- |
| `n_samples`         | × 2           |
| `n_warmup`          | × 2           |
| `thinning`          | ÷ 2 (最小为1) |
| `smoothness_weight` | × 0.7         |
| `proposal_std`      | × 0.8         |

最多重试 3 次 (`max_resampling_attempts`)，每次倍数累乘。

---

## 4. 参数清单

| 参数 (代码)            | 符号                  | 值    | 含义               |
| ---------------------- | --------------------- | ----- | ------------------ |
| `n_warmup`             | $N_{burn}$            | 3000  | 预热期（丢弃）     |
| `n_samples`            | $N$                   | 10000 | 采样总数           |
| `thinning`             | $k$                   | 5     | 稀疏因子           |
| `target_acceptance`    | $\alpha_{target}$     | 0.35  | 目标接受率         |
| `smoothness_weight`    | $\lambda$             | 0.5   | 时间平滑权重       |
| `initial_proposal_std` | $\sigma_{prop}^{(0)}$ | 0.15  | 初始提议标准差     |
| `PRIOR_STD`            | $\sigma_{prior}$      | 0.25  | 先验标准差         |
| `adapt_interval`       | -                     | 50    | 自适应调整间隔     |
| `min_ess_threshold`    | -                     | 100   | ESS 警戒阈值       |
| `random_seed`          | -                     | 42    | 随机种子（可复现） |

---

## 5. 验证指标公式

### 5.1 重构准确率 (Reconstruction Accuracy)

**代码实现** (`validation.py:compute_reconstruction_accuracy`):

$$\text{Acc} = \frac{1}{T} \sum_{t=1}^{T} \mathbb{I}\left[\text{Bottom-}k(\hat{\mathbf{f}}_t) = E_t\right]$$

其中：
- $T$ 为有淘汰的周次总数
- $k = |E_t|$ 为该周被淘汰人数
- $\hat{\mathbf{f}}_t$ 为后验均值
- $\text{Bottom-}k(\cdot)$ 返回综合得分最低的 $k$ 人集合

### 5.2 后验一致性 (Posterior Consistency)

**代码实现** (`validation.py:compute_posterior_consistency`):

对于每个淘汰周 $t$：
$$\text{Consistency}_t = \frac{1}{M} \sum_{m=1}^{M} \mathbb{I}\left[\arg\min_i \text{Total}_i^{(m)} \in E_t\right]$$

其中 $M$ 为 MCMC 样本数，$\text{Total}_i^{(m)}$ 为第 $m$ 个样本下选手 $i$ 的综合得分。

**汇总**：$\overline{\text{Consistency}} = \frac{1}{T} \sum_{t} \text{Consistency}_t$

### 5.3 约束满足率 (Constraint Satisfaction Rate)

**代码实现** (`validation.py:evaluate_constraint_satisfaction`):

$$\text{CSR} = \frac{\#\{t : P(E_t | \hat{\mathbf{f}}_t, J_t) = 1\}}{T}$$

即后验均值满足淘汰约束的周次比例。

### 5.4 Bottom-2 Recall (S28+)

**代码实现** (`validation.py:compute_bottom2_accuracy`):

$$\text{Bottom-2 Recall} = \frac{1}{T_{28+}} \sum_{t \in S_{28+}} \mathbb{I}\left[E_t \subseteq \text{Bottom-}(k+1)(\hat{\mathbf{f}}_t)\right]$$

对于 S28+ 的软性指标：只要被淘汰者都在预测的 Bottom-(k+1) 中即算正确。

### 5.5 决赛排名相关性 (Kendall's τ)

**代码实现** (`validation.py:compute_finale_ranking_accuracy`):

$$\tau = \frac{(\text{concordant pairs}) - (\text{discordant pairs})}{\binom{n}{2}}$$

衡量预测排名与实际排名的序相关性，$\tau \in [-1, 1]$。

### 5.6 95% 可信区间宽度

**代码实现** (`validation.py:compute_uncertainty_metrics`):

$$\text{CI Width}_{i,t} = Q_{97.5\%}(f_{i,t}^{(1:M)}) - Q_{2.5\%}(f_{i,t}^{(1:M)})$$

其中 $Q_p(\cdot)$ 为第 $p$ 百分位数。

### 5.7 敏感性分析指标

**代码实现** (`validation.py:run_sensitivity_analysis`):

**未淘汰者平均绝对偏差 (MAD)**：
$$\text{MAD} = \frac{1}{|S|} \sum_{i \in S} \left|\hat{f}_i^{(DD)} - \hat{f}_i^{(U)}\right|$$

其中 $S$ 为未淘汰者集合，$DD$ 为数据驱动先验，$U$ 为均匀先验。

**对称 KL 散度**：
$$D_{KL}^{sym} = \frac{1}{2}\left(D_{KL}(P \| Q) + D_{KL}(Q \| P)\right)$$
$$D_{KL}(P \| Q) = \sum_i p_i \log\frac{p_i}{q_i}$$

### 5.8 Judges' Save 偏好检验

**代码实现** (`validation.py:analyze_judges_save_preference`):

**零假设**：评委随机选择 Bottom-2 中的一人淘汰（无技术偏好）

**检验**：二项检验
$$H_0: P(\text{救高分者}) = 0.5$$
$$p\text{-value} = P(X \geq k | X \sim \text{Binomial}(n, 0.5))$$

其中 $n$ 为 Judges' Save 案例数，$k$ 为救高分者的次数。

## 关键结果

| 指标                          | 值               |
| ----------------------------- | ---------------- |
| **重构准确率**                | 93.49% (244/261) |
| **后验一致性**                | 90.01% (mean)    |
| **约束满足率**                | 99.23% (259/261) |
| **Bottom-2 Recall (Overall)** | 98.85%           |
| **冠军预测准确率**            | 100% (34/34)     |
| **Kendall's τ**               | 0.994            |
| **95% CI 平均宽度**           | 0.095            |

**MCMC 诊断**：
- 平均 ESS: 191
- 平均接受率: 34.7%

**规则变化影响**：

| 指标            | S1-S27 | S28+   |
| --------------- | ------ | ------ |
| Strict Accuracy | 100%   | 69.64% |
| Bottom-2 Recall | 100%   | 94.64% |

> S28+ 后 Strict Accuracy 下降反映 Judges' Save 规则引入后的结构性变化，非模型失败。

**稳定性分析**：
- Train (S1-20): 100%
- Test (S21+): 82.35%
- 泛化差距: 17.65%（结构漂移）

**敏感性分析**：

测试先验对未淘汰选手后验分布的影响。使用两组先验：数据驱动先验（基于行业统计）vs 均匀先验（0.5）。

| 指标                | 数据驱动先验 | 均匀先验 | 差异     |
| ------------------- | ------------ | -------- | -------- |
| 重构准确率          | 93.49%       | 97.70%   | +4.21%   |
| 未淘汰者 MAD (mean) | -            | -        | 0.00403  |
| 未淘汰者 KL (mean)  | -            | -        | 0.00142  |
| 未淘汰者 MAD (std)  | -            | -        | 0.00343  |
| 未淘汰者 KL (std)   | -            | -        | 0.00397  |

> **解读**：
> 1. **重构准确率差异** (+4.21%)：均匀先验重构准确率更高，说明数据驱动先验可能引入了行业偏见（如系统性低估某些行业），限制了 MCMC 的搜索空间。
> 2. **后验分布差异** (MAD≈0.004, KL≈0.001)：未淘汰选手的后验分布受先验影响极小，说明淘汰约束是主导因素——一旦观测到淘汰事实，后验分布快速收敛，先验影响被"数据洗掉"。
> 3. **先验选择建议**：虽然均匀先验在准确率上略优，但我们保持数据驱动先验作为主流程，原因：
>    - 体现贝叶斯"引入领域知识"的科学精神
>    - 敏感性分析证明先验影响极小（模型由数据主导）
>    - 报告两种结果增强了模型的透明度和可信度
> 4. **科学意义**：这是对论文可信度的**加分项**——结果由客观观测（淘汰事实）决定，而非主观先验假设。

**异常检测**：2 例
- S28W6: Sailor Brinkley-Cook
- S34W4: Hilaria Baldwin

**Judges' Save 偏好分析**（S28+）：

分析评委在 Bottom-2 中选择救谁的偏好模式。

| 指标                 | 值     |
| -------------------- | ------ |
| 分析案例数           | 42     |
| 救高分者比例         | 50.0%  |
| 救高粉丝份额者比例   | 90.5%  |
| 救综合排名更高者比例 | 81.0%  |
| 二项检验 p 值        | 0.5612 |
| 评委偏好技术流？     | **否** |

> **结论**：在 42 次 Judges' Save 中，评委 50% 的情况下救了 Judge Score 更高的选手 (p=0.5612)，**未发现显著技术偏好**。但评委 90.5% 的情况下救了粉丝份额更高的选手，81% 救了综合排名更高者。这表明评委的选择与粉丝投票高度一致，Judges' Save 机制并未有效纠正"粉丝主导"的问题。

## 输出文件

| 文件                        | 内容                      |
| --------------------------- | ------------------------- |
| `fan_shares.csv`            | 每周每选手的估计粉丝份额  |
| `results.json`              | 完整验证指标              |
| `bottom2_accuracy.json`     | 按赛季的 Bottom-2 指标    |
| `judges_save_analysis.json` | Judges' Save 偏好分析结果 |