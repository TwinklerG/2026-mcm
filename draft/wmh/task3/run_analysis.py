"""
Task 3 主运行脚本 - 双轨分层混合效应模型 (DT-HMEM)。

分析职业舞伴和明星特征对比赛成绩的影响。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .analysis import (
        analyze_celebrity_characteristics,
        analyze_fixed_effects,
        analyze_pro_effects,
        generate_paper_conclusions,
        save_analysis_results,
    )
    from .data_preparation import (
        build_panel_data,
        get_summary_statistics,
        load_raw_data,
        prepare_model_data,
    )
    from .model import DualTrackHMEM
    from .visualization import generate_all_plots
except ImportError:
    from analysis import (
        analyze_celebrity_characteristics,
        analyze_fixed_effects,
        analyze_pro_effects,
        generate_paper_conclusions,
        save_analysis_results,
    )
    from data_preparation import (
        build_panel_data,
        get_summary_statistics,
        load_raw_data,
        prepare_model_data,
    )
    from model import DualTrackHMEM
    from visualization import generate_all_plots


def main() -> None:
    """主函数：执行完整的 Task 3 分析流程。"""
    # 设置随机种子以确保可重复性
    import random

    import numpy as np

    np.random.seed(42)
    random.seed(42)

    print("=" * 70)
    print("Task 3: 双轨分层混合效应模型 (DT-HMEM)")
    print("分析职业舞伴和明星特征对比赛成绩的影响")
    print("=" * 70)

    # 设置输出目录
    output_dir = Path(__file__).parent / "outputs"
    figures_dir = Path(__file__).parent / "figures"
    data_dir = Path(__file__).parent / "data"

    # Step 1: 数据准备
    print("\n[Step 1/6] 加载和准备数据...")
    contestants, scores_long, fan_shares = load_raw_data()
    panel = build_panel_data(contestants, scores_long, fan_shares)

    # 打印数据摘要
    stats = get_summary_statistics(panel)
    print(f"  观测数: {stats['n_observations']}")
    print(f"  选手数: {stats['n_contestants']}")
    print(f"  赛季数: {stats['n_seasons']}")
    print(f"  职业舞伴数: {stats['n_pros']}")

    # 保存面板数据
    data_dir.mkdir(exist_ok=True)
    panel.sort(["season", "week", "contestant_id"]).write_csv(
        data_dir / "panel_data.csv"
    )

    # 转换为 pandas 并准备模型数据
    model_data = prepare_model_data(panel)
    print(f"  模型数据形状: {model_data.shape}")

    # Step 2: 拟合模型
    print("\n[Step 2/7] 拟合双轨混合效应模型...")
    model = DualTrackHMEM(model_data)

    # Track 1: 评委评分模型
    model.fit_judge_model()

    # Track 2: 粉丝投票模型
    model.fit_fan_model(include_judge_score=True)

    # Step 3: 计算关键指标
    print("\n[Step 3/7] 计算关键指标...")

    # 影响差异度指数
    idi_df_pl = model.compute_impact_divergence_index()
    idi_df = idi_df_pl.to_pandas().sort_values("variable")  # 确保排序一致
    print("\n影响差异度指数 (IDI):")
    print(idi_df_pl)

    # 职业舞伴排名
    pro_rankings_pl = model.get_pro_rankings()
    pro_rankings = pro_rankings_pl.to_pandas().sort_values("pro_id")  # 确保排序一致
    print("\n职业舞伴排名 (Top 10):")
    print(pro_rankings_pl.head(10))

    # 方差分解
    var_decomp = model.compute_variance_decomposition()
    print("\n方差分解:")
    for track in sorted(var_decomp.keys()):
        data = var_decomp[track]
        print(f"  {track.upper()}:")
        print(
            f"    职业舞伴方差: {data['pro_variance']:.4f} ({data['pro_variance_pct']:.1f}%)"
        )
        print(f"    ICC: {data['icc']:.4f}")

    # 表现转化率
    gamma = model.get_performance_conversion_rate()
    if gamma:
        print("\n表现转化率 (γ):")
        print(f"  估计值: {gamma['gamma']:.4f}")
        print(f"  显著性: p = {gamma['p_value']:.4f}")
        print(f"  显著: {gamma['significant']}")

    # Step 4: γ 敏感性分析（检验多重共线性）
    print("\n[Step 4/7] γ 敏感性分析...")
    gamma_sensitivity = model.run_gamma_sensitivity_analysis()

    # Step 5: 成分数据诊断
    print("\n[Step 5/7] 成分数据诊断...")
    compositional_diagnostics = model.compute_compositional_diagnostics()

    # Step 6: 深度分析
    print("\n[Step 6/7] 执行深度分析...")
    fixed_analysis = analyze_fixed_effects(model)
    pro_analysis = analyze_pro_effects(model, model_data)
    celebrity_analysis = analyze_celebrity_characteristics(model)

    # 生成结论
    conclusions = generate_paper_conclusions(
        model, fixed_analysis, pro_analysis, celebrity_analysis
    )

    print("\n核心数据:")

    # Step 7: 保存结果
    print("\n[Step 7/7] 保存结果...")
    output_dir.mkdir(exist_ok=True)

    # 保存模型结果
    model.save_results(output_dir)

    # 保存分析结果
    save_analysis_results(
        output_dir, fixed_analysis, pro_analysis, celebrity_analysis, conclusions
    )

    # 保存职业舞伴排名
    pro_rankings_sorted = pro_rankings.sort_values("combined_score", ascending=False)
    pro_rankings_sorted.to_csv(
        output_dir / "pro_rankings_full.csv", index=False, float_format="%.8f"
    )

    # 保存 IDI
    idi_df_sorted = idi_df.sort_values("variable")
    idi_df_sorted.to_csv(
        output_dir / "impact_divergence_index.csv", index=False, float_format="%.8f"
    )

    # 保存摘要统计
    def _to_json_serializable(obj, precision: int = 8):
        """递归转换为 JSON 可序列化格式。"""
        import numpy as np

        if isinstance(obj, dict):
            return {
                k: _to_json_serializable(v, precision)
                for k in sorted(obj.keys())
                for k, v in [(k, obj[k])]
            }
        elif isinstance(obj, (list, tuple)):
            return [_to_json_serializable(i, precision) for i in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return round(float(obj), precision)
        elif hasattr(obj, "item"):
            val = obj.item()
            if isinstance(val, float):
                return round(float(val), precision)
            elif isinstance(val, bool):
                return bool(val)
            elif isinstance(val, int):
                return int(val)
            else:
                return val
        else:
            return obj

    with open(output_dir / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            _to_json_serializable(stats),
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    # 保存 γ 敏感性分析结果
    with open(output_dir / "gamma_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(
            _to_json_serializable(gamma_sensitivity),
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    print(f"  已保存: {output_dir / 'gamma_sensitivity.json'}")

    # 保存成分数据诊断结果
    with open(
        output_dir / "compositional_diagnostics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            _to_json_serializable(compositional_diagnostics),
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    print(f"  已保存: {output_dir / 'compositional_diagnostics.json'}")

    # 生成图表
    print("\n生成可视化图表...")
    figures_dir.mkdir(exist_ok=True)
    generate_all_plots(
        model=model,
        data=model_data,
        pro_rankings=pro_rankings,
        variance_decomp=var_decomp,
        celebrity_analysis=celebrity_analysis,
        idi_df=idi_df,
        output_dir=figures_dir,
    )

    # 完成
    print("\n" + "=" * 70)
    print("Task 3 分析完成!")
    print("=" * 70)
    print("\n输出文件:")
    print(f"  数据: {data_dir}")
    print(f"  结果: {output_dir}")
    print(f"  图表: {figures_dir}")


if __name__ == "__main__":
    main()
