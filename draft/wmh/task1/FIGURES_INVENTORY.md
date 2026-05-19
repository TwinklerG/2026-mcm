# Task 1 图表清单

## 论文核心图表 (Paper Main Figures)

| 文件名                           | 用途               | 对应 PAPER.md |
| -------------------------------- | ------------------ | ------------- |
| `paper_summary.png`              | 核心结果总览 (2x2) | Fig. 3.1      |
| `stacked_accuracy_by_season.png` | S28 规则变化影响   | Fig. 3.2      |
| `judges_save_preference.png`     | Judges' Save 分析  | Fig. 3.3      |
| `credible_intervals.png`         | 不确定性分布       | Fig. 3.4      |

## 附录图表 (Appendix Figures)

| 文件名                          | 用途           | 说明     |
| ------------------------------- | -------------- | -------- |
| `mcmc_diagnostics.png`          | MCMC 诊断      | 技术验证 |
| `mcmc_trace.png`                | MCMC 链/轨迹   | 采样过程 |
| `finale_ranking_accuracy.png`   | 决赛排名准确率 | 支持数据 |
| `structural_drift_analysis.png` | 泛化分析       | 支持数据 |
| `temporal_trends.png`           | 时间趋势       | 补充分析 |
| `fan_share_distribution.png`    | 粉丝份额分布   | 补充分析 |
| `uncertainty_decomposition.png` | 不确定性分解   | 技术细节 |
| `performance_heatmap.png`       | 后验一致性热图 | 技术细节 |

## 可选/备用图表 (Optional)

| 文件名                         | 用途         | 说明                  |
| ------------------------------ | ------------ | --------------------- |
| `validation_radar.png`         | 验证雷达图   | 与 paper_summary 重复 |
| `validation_summary.png`       | 验证总结     | 与 paper_summary 重复 |
| `comprehensive_summary.png`    | 综合总结     | 与 paper_summary 重复 |
| `reconstruction_by_season.png` | 按赛季准确率 | 与 stacked 重复       |
| `controversial_vs_stable.png`  | 争议选手分析 | 可选补充              |
| `anomaly_analysis.png`         | 异常检测     | 可选补充              |

## 建议操作

1. **保留**：前两组图表（论文核心 + 附录）
2. **可选删除**：第三组中的重复图表
3. **建议合并**：将多个验证图表整合到 `paper_summary.png`

## 图表风格一致性检查

- ✅ 配色方案：深蓝 (#2E86AB) + 紫红 (#A23B72) 双轨系统
- ✅ 字体：serif，10pt 正文
- ✅ S28 标记线：橙色虚线
- ✅ 分辨率：300 DPI
