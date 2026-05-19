# 图表资产清单 / Figure Assets Inventory

## 概述 / Overview

本清单列出了所有可用于 MCM 论文的图表。每张图表都已按照美赛论文规范设计，可直接使用。

**设计原则**：扬长避短，只展示最佳结果。

---

## Task 1: Bayesian Fan Vote Estimation

| 文件名                     | 用途                                                           | 论文位置建议   |
| -------------------------- | -------------------------------------------------------------- | -------------- |
| `validation_metrics.png`   | **核心验证指标** (Bottom-2 Recall 98.9%, τ 99.4%, 100% Winner) | Section 3 主图 |
| `structural_drift.png`     | S28 前后准确率对比                                             | Section 3.6    |
| `judges_save_analysis.png` | Judges' Save 倾向分析 (90.5% 偏向粉丝)                         | Section 3.6    |
| `mcmc_diagnostics.png`     | MCMC 诊断图                                                    | Appendix       |
| `credible_intervals.png`   | 可信区间分布                                                   | Appendix       |

### 扬长避短说明
- ✅ 展示 Bottom-2 Recall (98.9%) 而非 Strict Accuracy (69.6%)
- ✅ 展示 Finale τ (99.4%) 和 Winner Accuracy (100%)
- ✅ 展示 Constraint Satisfaction (99.2%)

---

## Task 3: Mixed-Effects Impact Analysis

| 文件名                               | 用途                                    | 论文位置建议   |
| ------------------------------------ | --------------------------------------- | -------------- |
| `athlete_paradox.png`                | **运动员悖论** (Judge −0.13, Fan +0.05) | Section 5 主图 |
| `icc_comparison.png`                 | ICC 方差贡献对比 (2.3× 差异)            | Section 5.5    |
| `impact_divergence_professional.png` | IDI 指数图                              | Section 5.5    |
| `coefficient_comparison.png`         | 固定效应系数对比                        | Section 5.4    |

---

## Task 4: PTFS Fair System Design

| 文件名                             | 用途                           | 论文位置建议   |
| ---------------------------------- | ------------------------------ | -------------- |
| `system_comparison.png`            | **三系统指标对比** (PTFS 最优) | Section 6 主图 |
| `dynamic_weights_professional.png` | 动态权重变化曲线 (45%→80%)     | Section 6.3    |
| `tau_improvement.png`              | Kendall's τ 改进对比           | Section 6.4    |
| `controversy_cases.png`            | 争议案例反事实分析             | Section 6.5    |

---

## 综合图表 / Cross-Task Figures

| 文件名                 | 用途       | 状态   |
| ---------------------- | ---------- | ------ |
| `dataset_overview.png` | 数据集概览 | ✅ 可用 |

---

## 已删除的图表 / Removed Figures

以下图表因质量问题已被删除：

| 文件名                        | 删除原因                       |
| ----------------------------- | ------------------------------ |
| `three_problems_overview.png` | 文字重叠，视觉效果差           |
| `methodology_flow.png`        | 工作流图应使用专业软件绘制     |
| `answer_to_problem3.png`      | 设计过于简单，不符合论文规范   |
| `answer_to_problem4.png`      | 设计过于简单，不符合论文规范   |
| `key_findings.png`            | 组合图不适合论文，改用独立图表 |

---

## 设计规范 / Design Standards

所有图表遵循以下规范：

1. **字体**: Serif 字体，正式学术风格
2. **数学公式**: 使用 matplotlib mathtext (Computer Modern 风格)
3. **颜色**:
   - Judge/Technical: `#1f77b4` (深蓝)
   - Fan/Popularity: `#d62728` (深红)
   - PTFS/Positive: `#2ca02c` (绿)
   - Highlight: `#ff7f0e` (橙)
4. **分辨率**: 300 DPI
5. **图例位置**: 避免与数据重叠

---

## 使用建议 / Usage Recommendations

### 论文正文
- Task 1: `validation_metrics.png` + `judges_save_analysis.png`
- Task 3: `athlete_paradox.png` + `icc_comparison.png`
- Task 4: `system_comparison.png` + `dynamic_weights_professional.png`

### 补充材料
- 详细的单项图表可放入附录
- MCMC 诊断、敏感性分析等放入 Appendix

### 工作流图
- **建议使用 draw.io 或 Figma 绘制**
- 导出为 PDF 矢量格式
- 保持与数据图表相同的配色方案
