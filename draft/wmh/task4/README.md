# Task 4: 渐进技术公平系统 (PTFS)

## 概述

本模块实现问题四：提出一个更"公平"的投票组合系统。

**核心创新**：渐进技术公平系统 (Progressive Technical Fairness System, PTFS)

## 模型设计

### 问题诊断

基于 Task 1 和 Task 3 的分析发现：

1. **Judges' Save 失效**：90.5% 的情况下评委救了粉丝份额更高的选手
2. **运动员悖论**：评委负评 (β=-0.129)，粉丝正评 (β=+0.052)
3. **争议案例**：Jerry Rice、Bristol Palin、Bobby Bones 等技术分垫底却名次靠前

### PTFS 解决方案

1. **动态权重**：评委权重随周次递增（早期重粉丝参与，后期重技术评价）
2. **进步奖励**：激励技术进步而非仅看绝对分数
3. **技术保护**：顶级技术选手免疫淘汰

## 使用方法

```bash
cd draft/lly
uv run python -m task4_fair_system.run_analysis
```

## 输出文件

| 文件                        | 内容               |
| --------------------------- | ------------------ |
| `optimal_params.json`       | 优化后的参数       |
| `system_comparison.json`    | 三种系统的指标对比 |
| `detailed_metrics.json`     | 详细评价指标       |
| `sensitivity_analysis.json` | 参数敏感性分析     |

## 图表

| 文件                           | 内容             |
| ------------------------------ | ---------------- |
| `system_comparison_radar.png`  | 系统对比雷达图   |
| `controversy_analysis.png`     | 争议案例分析图   |
| `parameter_sensitivity.png`    | 参数敏感性分析图 |
| `metrics_comparison.png`       | 指标对比柱状图   |
| `dynamic_weights.png`          | 动态权重曲线     |

## 依赖

- Task 1 的 `fan_shares.csv`（估计的粉丝份额）
- 原始数据的 `contestants.csv` 和 `week_summary.csv`
