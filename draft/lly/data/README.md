# DWTS 数据说明

## 数据概览

- **原始数据**: `raw/2026_MCM_Problem_C_Data.csv` (421选手 × 34季)
- **处理数据**: `processed/` 目录下 5 个文件

## 文件结构

```
data/
├── raw/
│   └── 2026_MCM_Problem_C_Data.csv       # 原始数据
├── processed/
│   ├── contestants.csv                   # 选手基础信息表
│   ├── scores_long.csv                   # 周评分长表
│   ├── week_summary.csv                  # 周汇总表（含趋势特征）
│   ├── season_stats.csv                  # 季度统计表
│   ├── partner_stats.csv                 # 舞伴统计表
│   └── data_preprocessing.ipynb          # 数据处理notebook
├── validate_data.py                      # 基础验证脚本
└── validate_details.py                   # 深度验证脚本
```

## 数据文件说明

### 1. `contestants.csv` (421行)

**用途**: 选手基础信息表，每行代表一个选手在某季的参赛记录。

**关键字段**:
- `contestant_id`: 唯一标识符 (格式: `{name}_S{season}`)
- `celebrity_name`: 选手姓名
- `ballroom_partner`: 舞伴姓名
- `celebrity_industry`: 职业类别 (Actor/Actress, Athlete, Singer/Rapper等)
- `celebrity_age_during_season`: 参赛时年龄
- `season`: 季数 (1-34)
- `placement`: 最终名次 (1=冠军)
- `results`: 结果描述 (1st Place, Eliminated Week X, Withdrew)
- `is_withdrew`: 是否退赛 (boolean)

**使用场景**:
- Task 1: 获取选手 placement 用于约束推断
- Task 3: 获取选手协变量（年龄、职业、舞伴）

### 2. `scores_long.csv` (13,783行)

**用途**: 长格式评分数据，每行代表一个「选手 × 周 × 评委」的评分。

**关键字段**:
- `contestant_id`: 选手ID
- `season`: 季数
- `week`: 周数 (1-11)
- `judge`: 评委编号 (1-4)
- `score`: 评分 (0-10.5, 超过10为bonus分数)

**特点**:
- 已过滤淘汰后的0分记录
- 保留了 bonus 分数（最高达 13.3333）
- 评委编号按出场顺序，可能每周不同

**使用场景**:
- Task 3: 构建长格式面板数据（核心输入）

### 3. `week_summary.csv` (2,777行)

**用途**: 周汇总表，每行代表一个「选手 × 周」的汇总信息。

**关键字段**:
- `contestant_id`, `season`, `week`: 标识符
- `total_score`: 该周总分 (所有评委分数之和)
- `avg_score`: 该周平均分
- `judge_count`: 评委数量 (通常3或4)
- `week_rank`: 该周排名 (1=最高分)
- `prev_week_score`: 上周总分
- `score_change`: 周环比变化
- `cumulative_avg`: 累计平均分
- `deviation_from_avg`: 相对累计平均的偏差

**特点**:
- 已排除淘汰后的0分周
- 包含趋势特征，便于时序分析

**使用场景**:
- Task 1: 核心输入，用于反推观众投票

### 4. `season_stats.csv` (421行)

**用途**: 季度统计表，每行代表一个选手的整季表现汇总。

**关键字段**:
- `contestant_id`, `season`: 标识符
- `weeks_competed`: 参赛周数
- `season_total_score`: 整季总分
- `season_avg_weekly_score`: 周均总分
- `season_avg_judge_score`: 评委平均分
- `score_volatility`: 分数波动性 (标准差)
- `avg_rank`: 平均周排名
- `weeks_ranked_first`: 周冠军次数
- `placement`: 最终名次
- `voting_method`: 投票方法 (`rank_v1`/`percentage`/`rank_v2`)
- `is_controversy_case`: 是否为争议案例
- `industry_code`: 职业编码 (0-25)

**投票方法分类**:
- `rank_v1`: S1-2 (排名法)
- `percentage`: S3-27 (百分比法)
- `rank_v2`: S28-34 (排名法 + 底二淘汰机制)

**争议案例** (4人):
- Jerry Rice (S2): 亚军但5周最低评委分
- Billy Ray Cyrus (S4): 第5名但6周最低评委分
- Bristol Palin (S11): 季军但12次最低评委分
- Bobby Bones (S27): 冠军但持续低评委分

**使用场景**:
- **当前未使用**，可用于投票方法对比分析等

### 5. `partner_stats.csv` (36行)

**用途**: 舞伴统计表，汇总每位专业舞者的带人成绩。

**关键字段**:
- `ballroom_partner`: 舞伴姓名
- `total_partners`: 合作次数 (>=3次才纳入)
- `avg_placement`: 平均名次
- `best_placement`: 最佳名次
- `championships`: 冠军次数
- `top3_finishes`: 前三名次数
- `avg_judge_score`: 平均评委得分

**排序规则**: 按 `avg_placement` 升序 (越小越好)

**使用场景**:
- **当前未使用**，可用于分析舞伴对选手成绩的影响

## 数据处理原则

### 缺失值处理
- **N/A 保留**: 不删除或填充，保留其业务含义
  - 评委4为空 → 该周只有3位评委
  - 后期周数为空 → 该季周数较短
  - `homestate` 为空 → 选手非美国人

### 特殊值处理
- **0分过滤**: 淘汰后的0分在汇总时排除 (`total_score > 0`)
- **Bonus分数保留**: 超过10分的分数不截断 (最高13.3333)
- **退赛标记**: 单独 `is_withdrew` 列标识

### 数据类型
- 评分列: `Float64`
- season/placement/age: `Int32`
- 分类变量: `Utf8`

## 已知特性

1. **Season 15 全明星季**: 所有选手均为重复参赛
2. **评委数量变化**: 通常3位，部分周4位
3. **Bonus 分数**: 存在超过10分的评分（舞蹈对决等奖励）
4. **选手重复参赛**: 13人参加多季（如 Pamela Anderson: S10, S15）