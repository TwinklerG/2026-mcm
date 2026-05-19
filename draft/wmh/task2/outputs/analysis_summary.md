# Task 2 结果分析：DiffProb 与 FSAI（粉丝信号放大指数）

## 1. DiffProb：两种方法是否产生不同淘汰

### 定义
$$\text{DiffProb}_t = \mathbf{1}\{ e_t^{(\text{rank})} \neq e_t^{(\text{pct})} \}$$
即该周在「按名次加总」方法下被淘汰的人，与在「按百分比加总」方法下被淘汰的人是否不同。

### 结果
- **有淘汰的周数**：295 周
- **两方法淘汰不同人的周数**：30 周
- **平均 DiffProb（比例）**：30/295 ≈ **10.2%**

### 解读
- 约 **90%** 的周里，两种聚合规则选出的是**同一位**被淘汰者，说明在多数情况下两种方法结论一致。
- 约 **10%** 的周存在**方法差异**：选出的淘汰者不同，规则选择具有实质影响。

### 差异周分布（部分）
差异集中出现在部分赛季/周，例如：
- **S4 W2**：Rank 淘汰 Shandi Finnessey，Pct 淘汰 Paulina Porizkova
- **S8 W2**：Rank 淘汰 Steve Wozniak，Pct 淘汰 Belinda Carlisle
- **S11 W6**：Rank 淘汰 Bristol Palin，Pct 淘汰 Audrina Patridge
- **S28**：多周差异（W2, W4, W6, W7），如 Sean Spicer vs Lamar Odom / Sailor Brinkley-Cook / Karamo Brown
- **S31 W8**：Rank 淘汰 Vinny Guadagnino，Pct 淘汰 Heidi D'Amelio
- **S34**：W4, W7 等

说明：**S28 及之后**（规则或样本结构变化）差异周占比较高，与「两种规则在边际情形下谁被淘汰」的敏感性一致。

---

## 2. Fan Signal Amplification Index (FSAI)

### 定义
- **Score^{(J)}**：仅评委得分（judge_percent）。
- **Score^{(rank)}**：按名次加总下的综合得分（反向名次归一化，越高越好）。
- **Score^{(pct)}**：按百分比加总下的综合得分（percent_sum = fan_percent + judge_percent）。
- **FSAI_t^{(m)}** = Var_i(Score^{(m)} - Score^{(J)}) / Var_i(Score^{(m)})：衡量在规则 m 下，综合得分变异中有多大比例可归因于「与评委分的偏差」（即粉丝投票信号）。
- **Delta FSAI_t** = FSAI_t^{(pct)} - FSAI_t^{(rank)}。

### 解读规则
- **Delta FSAI_t > 0**：该周 Percentage 方法下粉丝信号在得分变异中占比更高 → Percentage 更放大粉丝投票。
- **Delta FSAI_t < 0**：该周 Rank 方法下粉丝信号占比更高 → **Rank 更偏粉丝**（机械上更敏感于粉丝投票）。
- **Delta FSAI_t ≈ 0**：两种规则对粉丝信号的放大程度相近。

### 结果
- **有效周数**（n≥2 且方差非零）：335 周
- **全样本平均** \(\overline{\Delta\text{FSAI}}\)：约 **-0.421**
- **按赛季平均** \(\overline{\Delta\text{FSAI}}\)：所有赛季均为**负**（约 -0.17 至 -0.64）

### 解读
- **Rank 方法**在得分变异中，由「与评委分的偏差」解释的比例**系统性高于** Percentage 方法，即机械上 **Rank 更放大粉丝投票信号**。
- 与方差分解一致：Rank 将名次加总后，综合得分对粉丝排名变化更敏感；Percentage 将百分比直接加总，评委份额与粉丝份额同尺度，粉丝造成的相对变异被压缩。
- 季节层面：S16、S28 等赛季 \(\overline{\Delta\text{FSAI}}\) 更负（约 -0.64、-0.60），Rank 相对 Pct 的「粉丝信号放大」更明显。

### 小结（FSAI）
- **整体**：\(\overline{\Delta\text{FSAI}} < 0\)，Rank 规则在**机械行为**上更偏粉丝投票（粉丝信号在综合得分变异中占比更大）。
- **政策含义**：若希望**降低**规则对粉丝投票的敏感度（让得分更贴近评委），Percentage 方法在方差意义上更「压缩」粉丝信号；若希望粉丝偏好更直接影响排名，则 Rank 方法放大效应更强。

---

## 3. 综合结论（可写入论文）

1. **方法是否不同**：约 10% 的周两种方法选出不同淘汰者（DiffProb ≈ 10.2%），说明**规则选择具有实质影响**，尤其在边际情形与部分赛季（如 S28 及以后）。
2. **谁更偏粉丝（FSAI）**：从粉丝信号放大指数看，**Rank 方法**在综合得分变异中，由粉丝投票带来的比例更高（\(\overline{\Delta\text{FSAI}} < 0\)），即**机械上 Rank 更偏粉丝**；Percentage 方法相对更压缩粉丝信号，得分更贴近评委基准。
3. **政策含义**：若希望**减少**规则对粉丝投票的放大，宜采用 Percentage 方法；若希望粉丝偏好更直接反映在得分变异中，则 Rank 方法放大效应更强。
