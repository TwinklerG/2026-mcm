# 论文图表与可视化总览

## 目录结构

```
draft/lly/
├── figures/                     # 跨问题综合图表
│   ├── three_problems_overview.png  # 三问题核心发现总览
│   ├── methodology_flow.png         # 方法论流程图
│   └── dataset_overview.png         # 数据集概览
│
├── shared/                      # 共享可视化配置
│   ├── __init__.py
│   └── viz_theme.py             # 统一主题、配色、样式
│
├── task1_bayesian_mcmc/
│   ├── figures/                 # Task 1 图表
│   ├── generate_paper_figures.py
│   └── FIGURES_INVENTORY.md     # 图表清单
│
├── task3_mixed_effects/
│   ├── figures/                 # Task 3 图表
│   ├── generate_paper_figures.py
│   └── FIGURES_INVENTORY.md     # 图表清单
│
└── task4_fair_system/
    ├── figures/                 # Task 4 图表
    ├── generate_paper_figures.py
    └── FIGURES_INVENTORY.md     # 图表清单
```

## 统一配色方案

### 核心双轨颜色
- **评委/技术**: `#2E86AB` (深蓝)
- **粉丝/流量**: `#A23B72` (紫红)

### 投票系统颜色
- **Rank-based**: `#C73E1D` (红色)
- **Percentage-based**: `#2E86AB` (深蓝)
- **PTFS**: `#2ECC71` (绿色)

### 状态颜色
- **成功/正向**: `#2ECC71` (绿色)
- **警告/负向**: `#E74C3C` (红色)
- **高亮**: `#F18F01` (橙色)
- **中性**: `#95A5A6` (灰色)

### 舞伴四象限
- **Kingmaker**: `#2ECC71` (绿色)
- **Technician**: `#2E86AB` (深蓝)
- **Fan Favorite**: `#A23B72` (紫红)
- **Underperformer**: `#95A5A6` (灰色)

## 论文推荐图表顺序

### 引言/数据部分
1. `dataset_overview.png` - 数据集规模与特征
2. `methodology_flow.png` - 方法论流程图

### Task 1: 粉丝投票估计 (Section 3)
1. `task1/paper_summary.png` - 核心验证指标
2. `task1/stacked_accuracy_by_season.png` - S28 规则变化影响
3. `task1/judges_save_preference.png` - Judges' Save 分析
4. `task1/credible_intervals.png` - 不确定性量化 (附录)

### Task 3: 影响因素分析 (Section 5)
1. `task3/paper_summary.png` - 核心发现总览
2. `task3/athlete_paradox_detail.png` - 运动员悖论
3. `task3/pro_scatter.png` - 职业舞伴四象限
4. `task3/answer_to_problem3.png` - 问题三答案

### Task 4: 公平系统设计 (Section 6)
1. `task4/paper_summary.png` - PTFS 核心结果
2. `task4/tau_by_season.png` - 34 赛季 τ 对比
3. `task4/dynamic_weights.png` - 动态权重曲线
4. `task4/controversy_cases.png` - 争议案例分析
5. `task4/answer_to_problem4.png` - 问题四答案

### 总结部分
1. `three_problems_overview.png` - 三问题核心发现总览

## 重新生成图表

```bash
# 重新生成所有图表
cd draft/lly
uv run python task1_bayesian_mcmc/generate_paper_figures.py
uv run python task3_mixed_effects/generate_paper_figures.py
uv run python task4_fair_system/generate_paper_figures.py
uv run python generate_cross_task_figures.py
```

## 图表质量检查清单

- [x] 统一配色方案
- [x] 统一字体 (serif, 10pt)
- [x] 300 DPI 输出
- [x] S28 规则变化标记 (橙色虚线)
- [x] 双语标注支持 (英文主标题)
- [x] 图例位置一致
- [x] 网格线样式统一

## PAPER.md 对应关系

| PAPER.md 建议图表                | 实际文件                         | 状态         |
| -------------------------------- | -------------------------------- | ------------ |
| Task1 Fig. 1 (Structural Drift)  | `stacked_accuracy_by_season.png` | ✅            |
| Task1 Fig. 2 (Uncertainty)       | `credible_intervals.png`         | ✅            |
| Task3 Fig. 5.1 (IDI)             | `impact_divergence.png`          | ✅            |
| Task3 Fig. 5.2 (ICC)             | `variance_decomposition.png`     | ✅            |
| Task3 Fig. 5.3 (Pro Quadrant)    | `pro_scatter.png`                | ✅            |
| Task4 Fig. 6.1 (Dynamic Weights) | `dynamic_weights.png`            | ✅            |
| Task4 Fig. 6.2 (τ by Season)     | `tau_by_season.png`              | ✅            |
| Task4 Fig. 6.3 (Improvement)     | `improvement_distribution.png`   | ✅            |
| Task4 Fig. 6.4 (Sensitivity)     | `sensitivity_heatmap.png`        | ⏸️ 需完整数据 |

## 注意事项

1. **编码问题**: 所有 JSON 文件读取时使用 `encoding="utf-8"`
2. **字体警告**: 某些特殊符号 (如 ✓) 可能缺失，使用文字替代
3. **数据兼容性**: 脚本已处理不同 JSON 结构格式
