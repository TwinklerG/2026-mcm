# LLY - 2026 MCM Problem C 解决方案

## 项目概述

本项目针对 2026 年 MCM 问题 C（Dancing with the Stars）提供完整的建模解决方案。

### 目录结构

```
draft/lly/
├── README.md                    # 本文件
├── main.py                      # 主入口（待实现）
├── pyproject.toml               # Python 项目配置
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   │   └── 2026_MCM_Problem_C_Data.csv
│   └── processed/               # 处理后的数据
│       ├── contestants.csv
│       ├── partner_stats.csv
│       ├── scores_long.csv
│       ├── season_stats.csv
│       └── week_summary.csv
├── problem/                     # 原题（不要修改）
│   └── C.md
└── task1_bayesian_mcmc/         # 任务 1：贝叶斯 MCMC 粉丝投票估计
    ├── data_structures.py       # 数据结构和配置
    ├── priors.py                # 先验估计
    ├── voting.py                # 投票方法和约束
    ├── sampler.py               # MCMC 采样器
    ├── validation.py            # 验证函数
    ├── model.py                 # 统一接口
    ├── run_inference.py         # 推断脚本
    ├── visualize.py             # 可视化脚本
    ├── README.md                # 详细文档
    ├── cache/                   # MCMC 缓存
    ├── outputs/                 # 结果输出
    │   ├── results.json
    │   ├── fan_shares.csv
    │   └── reconstruction_details.json
    └── figures/                 # 生成的图表
```

## 快速开始

### 安装依赖

```bash
cd mcm
uv sync
```

### 数据预处理

数据预处理已完成，处理后的数据位于 `data/processed/` 目录。

### 运行任务 1

```bash
# 完整运行（包含所有分析）
uv run python draft/lly/task1_bayesian_mcmc/run_inference.py

# 快速迭代（使用缓存，跳过耗时分析）
uv run python draft/lly/task1_bayesian_mcmc/run_inference.py --skip-sensitivity --skip-stability

# 生成可视化
uv run python draft/lly/task1_bayesian_mcmc/visualize.py
```

## 任务清单

- [x] **任务 1**：基于贝叶斯 MCMC 的粉丝投票估计
  - [x] 模型实现
  - [x] 验证框架
  - [x] 异常检测
  - [x] 可视化
  - [ ] 论文撰写
- [ ] **任务 2**：待定
- [ ] **任务 3**：待定
- [ ] **任务 4**：待定

## 任务 1 核心成果

### 模型性能

| 指标       | 数值   | 说明                     |
| ---------- | ------ | ------------------------ |
| 重构准确率 | 95.02% | 估计的淘汰与实际淘汰相符 |
| 后验一致性 | 89.97% | MCMC 样本的一致性        |
| 约束满足率 | 99.23% | 满足淘汰约束             |
| 异常检测   | 2 例   | 识别出异常淘汰           |

### 方法论

- **贝叶斯 MCMC**：自适应 Metropolis-Hastings 采样器
- **数据驱动先验**：基于历史数据的行业生存分析
- **高级验证**：重构准确率 + 后验一致性 + 稳定性分析
- **异常检测**：识别无法用标准投票模型解释的淘汰

详见 [task1_bayesian_mcmc/README.md](task1_bayesian_mcmc/README.md)

## 数据文件说明

### 原始数据

- `2026_MCM_Problem_C_Data.csv`：比赛原始数据

### 处理后数据

- `contestants.csv`：选手信息（ID、姓名、行业、年龄、舞伴等）
- `week_summary.csv`：每周表现汇总（评委分数、淘汰信息）
- `scores_long.csv`：评委分数详细数据（长格式）
- `season_stats.csv`：赛季统计
- `partner_stats.csv`：舞伴统计

## 代码规范

- 所有代码使用**中文注释和文档字符串**
- 图表标题、轴标签等输出内容使用**英文**
- 使用 `ruff` 进行代码格式化和检查
- NumPy 风格的文档字符串
- 类型提示必须完整

## 开发指南

### 代码质量检查

```bash
# 代码检查
uv run ruff check draft/lly

# 代码格式化
uv run ruff format draft/lly
```

### 模块化原则

任务 1 的代码已经被拆分为多个模块：

- `data_structures.py` - 数据类和配置
- `priors.py` - 先验估计
- `voting.py` - 投票规则
- `sampler.py` - MCMC 采样
- `validation.py` - 验证指标
- `model.py` - 统一接口（向后兼容）

这种结构使代码：
- 更易于理解和维护
- 每个模块职责单一
- 便于单元测试
- 支持独立导入

### 缓存机制

MCMC 采样结果会被缓存在 `cache/` 目录，基于配置参数的哈希值。这允许：
- 快速迭代开发
- 避免重复耗时计算
- 可选择性清除缓存重新运行

## 学术定位

本项目强调：

1. **贝叶斯方法的正确使用**
   - 使用"可信区间"而非"置信区间"
   - 后验分布的完整量化

2. **适当的验证方法**
   - 无真实标签时的验证策略
   - 重构准确率作为核心指标
   - 时间划分而非随机划分

3. **异常检测作为功能**
   - 不是"模型失败"
   - 识别特殊规则或数据问题

## 下一步工作

- [ ] 完成任务 1 的论文撰写
- [ ] 设计和实现任务 2
- [ ] 设计和实现任务 3
- [ ] 设计和实现任务 4
- [ ] 整合所有任务到主论文

## 参考资料

详见各子任务的 README 文件。
