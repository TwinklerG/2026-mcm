# MCM 2026

## 安装

```bash
uv sync
```

## 模块结构

```
templates/
├── data/           # 数据加载、预处理、特征工程、探索
├── viz/            # Plotly 交互 + Matplotlib 论文级
├── ml/             # 回归、分类、聚类、评估
├── stats/          # 相关、假设检验、AHP/TOPSIS/熵权法、敏感性
├── network/        # 网络科学：分析、模型、动力学、可视化
└── utils/          # Typst 导出、格式化、计时
```

## 开发

```bash
uv run ruff check      # 代码检查
uv run ruff format     # 代码格式化
uv run pytest          # 单元测试
uv run demo.py         # 运行示例
```

## 文档

接口示例：[demo.py](demo.py)。