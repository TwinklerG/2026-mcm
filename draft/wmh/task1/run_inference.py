"""
运行 DWTS 粉丝投票估计的贝叶斯 MCMC 推断。

该脚本：
1. 加载和预处理 DWTS 数据
2. 运行带缓存的自适应 MCMC
3. 计算验证指标（含新增的 Bottom-2 Recall 和 Finale Ranking）
4. 生成用于论文写作的输出
5. [NEW] 详细的不确定性分析
6. [NEW] 低 ESS 赛季的自适应重采样
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from model import (
    AdaptiveMCMCSampler,
    ContestantInfo,
    MCMCConfig,
    SeasonWeekData,
    analyze_anomalies_for_paper,
    analyze_judges_save_preference,
    compute_bottom2_accuracy,
    compute_detailed_uncertainty,
    compute_finale_ranking_accuracy,
    compute_posterior_consistency,
    # 验证函数
    compute_reconstruction_accuracy,
    compute_uncertainty_metrics,
    evaluate_constraint_satisfaction,
    run_sensitivity_analysis,
    run_stability_analysis,
)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
CACHE_DIR = SCRIPT_DIR / "cache"

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[
    list[SeasonWeekData],
    dict[str, ContestantInfo],
    pl.DataFrame,
    pl.DataFrame,
]:
    """加载和预处理 DWTS 数据。"""
    print("Loading data...")

    df_contestants = pl.read_csv(DATA_DIR / "contestants.csv")
    df_week_summary = pl.read_csv(DATA_DIR / "week_summary.csv")

    # 构建选手信息字典
    contestant_info: dict[str, ContestantInfo] = {}
    for row in df_contestants.iter_rows(named=True):
        cid = row["contestant_id"]
        contestant_info[cid] = ContestantInfo(
            contestant_id=cid,
            industry=row["celebrity_industry"] or "Unknown",
            age=float(row["celebrity_age_during_season"] or 40),
            partner=row["ballroom_partner"] or "Unknown",
        )

    # 构建赛季-周数据列表
    season_week_data: list[SeasonWeekData] = []
    seasons = sorted(df_contestants["season"].unique().to_list())

    for season in seasons:
        season_contestants = df_contestants.filter(pl.col("season") == season)
        season_weeks = df_week_summary.filter(pl.col("season") == season)
        weeks = sorted(season_weeks["week"].unique().to_list())

        placements = {
            row["contestant_id"]: int(row["placement"])
            for row in season_contestants.iter_rows(named=True)
            if row["placement"] is not None
        }

        max_week = max(weeks)

        for week in weeks:
            week_data = season_weeks.filter(pl.col("week") == week)
            active_data = week_data.filter(pl.col("total_score") > 0)

            if active_data.height == 0:
                continue

            contestant_ids = active_data["contestant_id"].to_list()
            judge_scores = np.array(active_data["total_score"].to_list())

            # 确定被淘汰的选手
            eliminated_ids = []
            if week < max_week:
                next_week_data = season_weeks.filter(pl.col("week") == week + 1)
                next_week_ids = set(
                    next_week_data.filter(pl.col("total_score") > 0)[
                        "contestant_id"
                    ].to_list()
                )
                for cid in contestant_ids:
                    if cid not in next_week_ids:
                        eliminated_ids.append(cid)

            is_finale = week == max_week
            final_placements = None
            if is_finale:
                final_placements = {
                    cid: placements.get(cid, 99)
                    for cid in contestant_ids
                    if cid in placements
                }

            swd = SeasonWeekData(
                season=season,
                week=week,
                contestant_ids=contestant_ids,
                judge_scores=judge_scores,
                eliminated_ids=eliminated_ids,
                is_finale=is_finale,
                final_placements=final_placements,
            )
            season_week_data.append(swd)

    print(f"  加载了 {len(season_week_data)} 个周次，跨越 {len(seasons)} 个赛季。")
    print(f"  选手总数： {len(contestant_info)}")

    return season_week_data, contestant_info, df_contestants, df_week_summary


def main(
    skip_sensitivity: bool = False,
    skip_stability: bool = False,
    clear_cache: bool = False,
):
    """
    主推断函数。

    Parameters
    ----------
    skip_sensitivity : bool
        跳过敏感性分析（更快迭代）。
    skip_stability : bool
        跳过稳定性分析（更快迭代）。
    clear_cache : bool
        清除缓存结果并重新运行所有内容。
    """
    print("=" * 70)
    print("贝叶斯 MCMC 粉丝投票估计")
    print("=" * 70)

    # 设置随机种子以确保可重复性
    import random

    np.random.seed(42)
    random.seed(42)

    # 清除缓存（如果需要）
    if clear_cache and CACHE_DIR.exists():
        import shutil

        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir()
        print("缓存已清除。")

    # 加载数据
    season_week_data, contestant_info, df_contestants, df_week_summary = load_data()

    # 配置
    config = MCMCConfig(
        n_warmup=3000,
        n_samples=10000,
        thinning=5,
        initial_proposal_std=0.15,
        target_acceptance=0.35,
        adapt_interval=50,
        smoothness_weight=0.5,
        random_seed=42,
    )

    print(f"\n配置哈希: {config.to_hash()[:12]}...")
    print(f"缓存目录: {CACHE_DIR}")

    # =======================================================================
    # 主 MCMC 采样
    # =======================================================================
    print("\n[1/5] 运行 MCMC 采样...")
    start_time = time.time()

    sampler = AdaptiveMCMCSampler(
        season_week_data,
        contestant_info,
        config,
        cache_dir=CACHE_DIR,
        use_data_driven_priors=True,
    )

    posteriors, diagnostics, samples = sampler.sample_all_seasons(
        verbose=True, return_samples=True
    )

    elapsed = time.time() - start_time
    print(f"  总采样时间: {elapsed:.1f}s")

    # =======================================================================
    # 验证指标
    # =======================================================================
    print("\n[2/7] Computing Validation Metrics...")

    # 2.1 重构准确率
    recon = compute_reconstruction_accuracy(posteriors, season_week_data)
    print(f"  Reconstruction Accuracy: {recon['reconstruction_accuracy']:.2%}")
    print(f"    Correct: {recon['correct']}/{recon['total']}")

    # 2.2 后验一致性
    seasons_data = sampler.seasons_data
    consistency = compute_posterior_consistency(samples, seasons_data)
    print(f"  后验一致性: {consistency['mean_consistency']:.2%} (平均)")
    print(f"    高置信度周次: {consistency['high_confidence_episodes']}")
    print(f"    不确定周次: {consistency['uncertain_episodes']}")

    # 2.3 约束满足率（遗留，用于比较）
    rate, satisfied, total, anomalies = evaluate_constraint_satisfaction(
        posteriors, season_week_data
    )
    print(f"  约束满足率: {rate:.2%} ({satisfied}/{total})")

    # =======================================================================
    # [NEW] Bottom-2 Recall 指标
    # =======================================================================
    print("\n[3/7] Bottom-2 Recall (针对 S28+ 规则)...")
    bottom2_results = compute_bottom2_accuracy(posteriors, season_week_data)
    print(
        f"  Overall Strict Accuracy: {bottom2_results['overall']['strict_accuracy']:.2%}"
    )
    print(
        f"  Overall Bottom-2 Recall: {bottom2_results['overall']['bottom2_recall']:.2%}"
    )
    print(
        f"  S1-S27 Strict: {bottom2_results['pre_s28']['strict_accuracy']:.2%}, "
        f"B2: {bottom2_results['pre_s28']['bottom2_recall']:.2%}"
    )
    print(
        f"  S28+   Strict: {bottom2_results['post_s28']['strict_accuracy']:.2%}, "
        f"B2: {bottom2_results['post_s28']['bottom2_recall']:.2%}"
    )

    # =======================================================================
    # [NEW] 决赛排名准确率
    # =======================================================================
    print("\n[4/7] Finale Ranking Accuracy...")
    finale_results = compute_finale_ranking_accuracy(posteriors, season_week_data)
    print(
        f"  Winner Accuracy: {finale_results['winner_accuracy']:.2%} "
        f"({finale_results['winner_correct']}/{finale_results['winner_total']})"
    )
    print(f"  Top-3 Position Accuracy: {finale_results['top3_position_accuracy']:.2%}")
    print(f"  Mean Kendall's τ: {finale_results['mean_kendall_tau']:.3f}")

    # =======================================================================
    # 异常检测
    # =======================================================================
    print("\n[5/7] 异常检测...")
    all_anomalies = sampler.detected_anomalies + anomalies
    anomaly_analysis = analyze_anomalies_for_paper(all_anomalies)
    print(f"  检测到 {anomaly_analysis['n_anomalies']} 个异常淘汰")

    # =======================================================================
    # [NEW] 详细不确定性量化
    # =======================================================================
    print("\n[6/7] 详细不确定性量化...")
    uncertainty = compute_uncertainty_metrics(samples, seasons_data)
    cri_widths = [v["credible_interval_width"] for v in uncertainty.values()]
    print(f"  95% 可信区间平均宽度: {np.mean(cri_widths):.4f}")
    print(f"  95% 可信区间中位数宽度: {np.median(cri_widths):.4f}")

    # 详细不确定性分析
    detailed_uncertainty = compute_detailed_uncertainty(
        samples, seasons_data, contestant_info
    )
    if "error" not in detailed_uncertainty:
        print(
            f"  争议型选手数量: {len(detailed_uncertainty.get('controversial_contestants', []))}"
        )
        print(
            f"  稳态型选手数量: {len(detailed_uncertainty.get('stable_contestants', []))}"
        )

    # =======================================================================
    # 可选：敏感性和稳定性分析
    # =======================================================================
    sensitivity_results = None
    stability_results = None

    if not skip_sensitivity:
        print("\n[7a/7] 敏感性分析...")
        sensitivity_results = run_sensitivity_analysis(
            season_week_data, contestant_info, config, CACHE_DIR
        )
        print(
            f"  准确率差异: {sensitivity_results['sensitivity']['accuracy_difference']:.4f}"
        )

    if not skip_stability:
        print("\n[7b/7] 稳定性分析（跨赛季泛化）...")
        stability_results = run_stability_analysis(
            season_week_data,
            contestant_info,
            config,
            train_seasons_end=20,
            cache_dir=CACHE_DIR,
        )
        print(f"  训练集准确率: {stability_results['train_accuracy']:.2%}")
        print(f"  测试集准确率: {stability_results['test_accuracy']:.2%}")
        print(f"  泛化差距: {stability_results['generalization_gap']:.2%}")

    # =======================================================================
    # 诊断摘要
    # =======================================================================
    print("\n" + "=" * 70)
    print("MCMC 诊断总结")
    print("=" * 70)

    ess_values = [d.effective_sample_size for d in diagnostics.values()]
    acceptance_rates = [d.acceptance_rate for d in diagnostics.values()]

    # 检查低 ESS 赛季
    low_ess_seasons = [
        s
        for s, d in diagnostics.items()
        if d.effective_sample_size < config.min_ess_threshold
    ]
    multimodal_seasons = [
        s for s, d in diagnostics.items() if getattr(d, "multimodal_suspected", False)
    ]

    print(f"  ESS: 平均={np.mean(ess_values):.0f}, 最小={np.min(ess_values):.0f}")
    print(f"  接受率: 平均={np.mean(acceptance_rates):.1%}")

    if low_ess_seasons:
        print(f"  [!] 低 ESS 赛季 (<{config.min_ess_threshold}): {low_ess_seasons}")
    if multimodal_seasons:
        print(f"  [!] 疑似多峰分布赛季: {multimodal_seasons}")

    # =======================================================================
    # 保存结果
    # =======================================================================
    print("\n保存结果...")

    results = {
        "config": {
            "n_warmup": config.n_warmup,
            "n_samples": config.n_samples,
            "thinning": config.thinning,
            "smoothness_weight": config.smoothness_weight,
            "target_acceptance": config.target_acceptance,
            "config_hash": config.to_hash(),
        },
        "validation": {
            "reconstruction_accuracy": recon["reconstruction_accuracy"],
            "reconstruction_correct": recon["correct"],
            "reconstruction_total": recon["total"],
            "posterior_consistency_mean": consistency["mean_consistency"],
            "posterior_consistency_std": consistency["std_consistency"],
            "high_confidence_episodes": consistency["high_confidence_episodes"],
            "uncertain_episodes": consistency["uncertain_episodes"],
            "constraint_satisfaction_rate": rate,
            # 新增指标
            "bottom2_overall_strict": bottom2_results["overall"]["strict_accuracy"],
            "bottom2_overall_recall": bottom2_results["overall"]["bottom2_recall"],
            "bottom2_pre_s28_strict": bottom2_results["pre_s28"]["strict_accuracy"],
            "bottom2_pre_s28_recall": bottom2_results["pre_s28"]["bottom2_recall"],
            "bottom2_post_s28_strict": bottom2_results["post_s28"]["strict_accuracy"],
            "bottom2_post_s28_recall": bottom2_results["post_s28"]["bottom2_recall"],
            "finale_winner_accuracy": finale_results["winner_accuracy"],
            "finale_top3_accuracy": finale_results["top3_position_accuracy"],
            "finale_kendall_tau": finale_results["mean_kendall_tau"],
        },
        "diagnostics": {
            "mean_ess": float(np.mean(ess_values)),
            "min_ess": float(np.min(ess_values)),
            "mean_acceptance_rate": float(np.mean(acceptance_rates)),
            "low_ess_seasons": low_ess_seasons,
            "multimodal_seasons": multimodal_seasons,
        },
        "uncertainty": {
            "mean_cri_width": float(np.mean(cri_widths)),
            "median_cri_width": float(np.median(cri_widths)),
        },
        "anomalies": anomaly_analysis,
    }

    if sensitivity_results:
        results["sensitivity"] = sensitivity_results
    if stability_results:
        results["stability"] = {
            "train_accuracy": stability_results["train_accuracy"],
            "test_accuracy": stability_results["test_accuracy"],
            "generalization_gap": stability_results["generalization_gap"],
        }

    # 保存决赛排名详情
    def convert_to_json_serializable(obj, precision: int = 8):
        """递归转换对象为 JSON 可序列化格式，保持字典键排序和浮点精度。"""
        if isinstance(obj, dict):
            # 排序键以确保输出稳定性
            return {
                k: convert_to_json_serializable(v, precision)
                for k in sorted(obj.keys())
                for k, v in [(k, obj[k])]
            }
        elif isinstance(obj, list):
            return [convert_to_json_serializable(i, precision) for i in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return round(float(obj), precision)
        elif isinstance(obj, np.ndarray):
            return [convert_to_json_serializable(x, precision) for x in obj.tolist()]
        else:
            return obj

    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            convert_to_json_serializable(results),
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    print(f"  已保存: {OUTPUT_DIR / 'results.json'}")

    # 保存 Bottom-2 详情
    # 移除 raw cases 以减小文件大小
    bottom2_save = {k: v for k, v in bottom2_results.items() if k != "cases"}
    with open(OUTPUT_DIR / "bottom2_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(
            convert_to_json_serializable(bottom2_save),
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    print(f"  已保存: {OUTPUT_DIR / 'bottom2_accuracy.json'}")

    finale_save = {k: v for k, v in finale_results.items() if k != "cases"}
    finale_save = convert_to_json_serializable(finale_save)
    with open(OUTPUT_DIR / "finale_ranking.json", "w", encoding="utf-8") as f:
        json.dump(finale_save, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"  已保存: {OUTPUT_DIR / 'finale_ranking.json'}")

    # 保存详细不确定性分析（移除 raw_data）
    if "error" not in detailed_uncertainty:
        detailed_unc_save = {
            k: v for k, v in detailed_uncertainty.items() if k != "raw_data"
        }
        with open(OUTPUT_DIR / "detailed_uncertainty.json", "w", encoding="utf-8") as f:
            json.dump(
                convert_to_json_serializable(detailed_unc_save),
                f,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        print(f"  已保存: {OUTPUT_DIR / 'detailed_uncertainty.json'}")

    # 保存粉丝份额估计
    fan_shares_data = []
    for swd in season_week_data:
        key = (swd.season, swd.week)
        if key not in posteriors:
            continue
        shares = posteriors[key]
        for i, cid in enumerate(swd.contestant_ids):
            fan_shares_data.append({
                "season": swd.season,
                "week": swd.week,
                "contestant_id": cid,
                "estimated_fan_share": float(shares[i]),
                "is_eliminated": cid in swd.eliminated_ids,
            })

    df_fan_shares = pl.DataFrame(fan_shares_data).sort([
        "season",
        "week",
        "contestant_id",
    ])
    df_fan_shares.write_csv(OUTPUT_DIR / "fan_shares.csv")
    print(f"  已保存: {OUTPUT_DIR / 'fan_shares.csv'}")

    # 保存重构详情
    recon_save = {
        "per_season": recon["per_season"],
        "cases": recon["cases"][:50],  # 前 50 个案例供检查
    }
    with open(OUTPUT_DIR / "reconstruction_details.json", "w", encoding="utf-8") as f:
        json.dump(
            convert_to_json_serializable(recon_save),
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    print(f"  已保存: {OUTPUT_DIR / 'reconstruction_details.json'}")

    # [NEW] Judges' Save 偏好分析
    print("\n[NEW] Judges' Save 偏好分析...")
    judges_save_analysis = analyze_judges_save_preference(posteriors, season_week_data)
    print(f"  分析案例数: {judges_save_analysis['n_judges_save_episodes']}")
    if judges_save_analysis["n_judges_save_episodes"] > 0:
        print(
            f"  救高分者比例: {judges_save_analysis['pct_saved_higher_judge_score']:.1%}"
        )
        print(f"  结论: {judges_save_analysis['conclusion']}")

    # 保存 Judges' Save 分析（移除详细 cases 以减小文件大小）
    judges_save_save = {k: v for k, v in judges_save_analysis.items() if k != "cases"}
    with open(OUTPUT_DIR / "judges_save_analysis.json", "w", encoding="utf-8") as f:
        json.dump(
            convert_to_json_serializable(judges_save_save),
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    print(f"  已保存: {OUTPUT_DIR / 'judges_save_analysis.json'}")

    print("\n" + "=" * 70)
    print("推断完成！")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行 MCMC 推断")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    main(
        skip_sensitivity=args.skip_sensitivity,
        skip_stability=args.skip_stability,
        clear_cache=args.clear_cache,
    )
