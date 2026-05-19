"""数据验证脚本 - 检查数据处理的正确性和完整性"""

from pathlib import Path

import polars as pl

# 路径配置
DATA_RAW = Path(__file__).parent / "raw"
DATA_PROCESSED = Path(__file__).parent / "processed"


def validate_raw_data() -> dict:
    """验证原始数据的基本特征"""
    print("=" * 70)
    print("1. 原始数据验证")
    print("=" * 70)

    df = pl.read_csv(
        DATA_RAW / "2026_MCM_Problem_C_Data.csv",
        null_values=["N/A", "NA", ""],
        infer_schema_length=10000,
    )

    results = {
        "total_rows": df.height,
        "total_cols": df.width,
        "seasons": sorted(df["season"].unique().to_list()),
        "n_seasons": df["season"].n_unique(),
    }

    print(f"✓ 总行数: {results['total_rows']}")
    print(f"✓ 总列数: {results['total_cols']}")
    print(f"✓ 季数范围: {min(results['seasons'])} - {max(results['seasons'])}")
    print(f"✓ 覆盖季数: {results['n_seasons']}")

    # 检查评分列
    score_cols = [col for col in df.columns if "judge" in col and "score" in col]
    print(f"✓ 评分列数量: {len(score_cols)}")

    # 检查关键列完整性
    key_cols = ["celebrity_name", "season", "results", "placement"]
    for col in key_cols:
        null_count = df[col].null_count()
        print(f"  - {col}: {null_count} 缺失")

    return results


def validate_processed_data() -> dict:
    """验证处理后数据的完整性和一致性"""
    print("\n" + "=" * 70)
    print("2. 处理后数据验证")
    print("=" * 70)

    # 加载所有处理后的数据
    contestants = pl.read_csv(DATA_PROCESSED / "contestants.csv")
    scores_long = pl.read_csv(DATA_PROCESSED / "scores_long.csv")
    week_summary = pl.read_csv(DATA_PROCESSED / "week_summary.csv")
    season_stats = pl.read_csv(DATA_PROCESSED / "season_stats.csv")
    partner_stats = pl.read_csv(DATA_PROCESSED / "partner_stats.csv")

    results = {
        "contestants_count": contestants.height,
        "scores_count": scores_long.height,
        "week_summary_count": week_summary.height,
        "season_stats_count": season_stats.height,
        "partner_stats_count": partner_stats.height,
    }

    print(f"✓ contestants.csv: {contestants.height} 行")
    print(f"✓ scores_long.csv: {scores_long.height} 行")
    print(f"✓ week_summary.csv: {week_summary.height} 行")
    print(f"✓ season_stats.csv: {season_stats.height} 行")
    print(f"✓ partner_stats.csv: {partner_stats.height} 行")

    return results


def validate_data_consistency() -> dict:
    """验证数据一致性"""
    print("\n" + "=" * 70)
    print("3. 数据一致性检查")
    print("=" * 70)

    contestants = pl.read_csv(DATA_PROCESSED / "contestants.csv")
    scores_long = pl.read_csv(DATA_PROCESSED / "scores_long.csv")
    week_summary = pl.read_csv(DATA_PROCESSED / "week_summary.csv")
    season_stats = pl.read_csv(DATA_PROCESSED / "season_stats.csv")

    issues = []

    # 检查1: contestants 与 season_stats 的选手数量应该一致
    if contestants.height != season_stats.height:
        issues.append(
            f"选手数量不一致: contestants={contestants.height}, "
            f"season_stats={season_stats.height}"
        )
    else:
        print(f"✓ 选手数量一致: {contestants.height}")

    # 检查2: contestant_id 的唯一性
    unique_ids_contestants = contestants["contestant_id"].n_unique()
    unique_ids_stats = season_stats["contestant_id"].n_unique()
    if unique_ids_contestants != contestants.height:
        issues.append(
            f"contestant_id 有重复: {unique_ids_contestants} != {contestants.height}"
        )
    else:
        print(f"✓ contestant_id 唯一: {unique_ids_contestants}")

    # 检查3: 每季冠军数量
    champions = season_stats.filter(pl.col("placement") == 1)
    champs_per_season = (
        champions.group_by("season").agg(pl.len().alias("count")).sort("season")
    )
    multi_champs = champs_per_season.filter(pl.col("count") > 1)
    if multi_champs.height > 0:
        issues.append(f"多冠军季数: {multi_champs.height}")
        print(f"⚠ 多冠军季数: {multi_champs.height}")
        print(multi_champs)
    else:
        print(f"✓ 每季仅一个冠军")

    # 检查4: placement 与 results 的一致性
    placement_1 = season_stats.filter(pl.col("placement") == 1)
    result_1st = placement_1.filter(pl.col("results").str.contains("1st Place"))
    if placement_1.height != result_1st.height:
        issues.append(
            f"冠军标记不一致: placement=1 有 {placement_1.height} 人, "
            f"results='1st Place' 有 {result_1st.height} 人"
        )
    else:
        print(f"✓ 冠军标记一致: {placement_1.height}")

    # 检查5: scores_long 中的 contestant_id 都应在 contestants 中
    contestant_ids_set = set(contestants["contestant_id"].to_list())
    score_ids = set(scores_long["contestant_id"].unique().to_list())
    missing_ids = score_ids - contestant_ids_set
    if missing_ids:
        issues.append(f"scores_long 中有 {len(missing_ids)} 个未知 contestant_id")
    else:
        print(f"✓ scores_long 中的 contestant_id 全部有效")

    # 检查6: 评分范围
    scores = scores_long["score"]
    min_score = scores.min()
    max_score = scores.max()
    print(f"✓ 评分范围: {min_score} - {max_score}")
    if min_score < 0:
        issues.append(f"存在负分: {min_score}")
    if max_score > 10.5:  # 允许少量bonus分数
        print(f"  ⚠ 最高分超过10: {max_score} (可能是bonus分数)")

    # 检查7: week_summary 与 scores_long 的一致性
    # week_summary 应该是 scores_long 的聚合
    week_groups = scores_long.filter(pl.col("score") > 0).group_by([
        "contestant_id",
        "season",
        "week",
    ])
    week_count_from_long = week_groups.agg(pl.len().alias("count")).height
    week_summary_count = week_summary.height
    if abs(week_count_from_long - week_summary_count) > 10:  # 允许小误差
        issues.append(
            f"week_summary 数量不匹配: {week_summary_count} vs "
            f"{week_count_from_long} (from scores_long)"
        )
        print(
            f"⚠ week_summary 数量: {week_summary_count} vs "
            f"{week_count_from_long} (from scores_long)"
        )
    else:
        print(f"✓ week_summary 数量合理: {week_summary_count}")

    return {"issues": issues}


def validate_task_usage() -> dict:
    """验证任务代码使用的数据"""
    print("\n" + "=" * 70)
    print("4. 任务数据使用验证")
    print("=" * 70)

    issues = []

    # 检查 task1 需要的数据
    try:
        contestants = pl.read_csv(DATA_PROCESSED / "contestants.csv")
        week_summary = pl.read_csv(DATA_PROCESSED / "week_summary.csv")
        print("✓ Task1 所需数据文件存在:")
        print("  - contestants.csv")
        print("  - week_summary.csv")

        # 检查必要列
        required_cols_contestants = ["contestant_id", "season", "placement"]
        required_cols_week = [
            "contestant_id",
            "season",
            "week",
            "total_score",
            "week_rank",
        ]
        for col in required_cols_contestants:
            if col not in contestants.columns:
                issues.append(f"contestants.csv 缺少列: {col}")
        for col in required_cols_week:
            if col not in week_summary.columns:
                issues.append(f"week_summary.csv 缺少列: {col}")

        if not issues:
            print("✓ Task1 所需列全部存在")

    except Exception as e:
        issues.append(f"Task1 数据加载失败: {e}")

    # 检查 task3 需要的数据
    try:
        scores_long = pl.read_csv(DATA_PROCESSED / "scores_long.csv")
        # task3 还需要 fan_shares.csv (从 task1 生成)
        print("✓ Task3 所需基础数据文件存在:")
        print("  - contestants.csv")
        print("  - scores_long.csv")

        required_cols_scores = ["contestant_id", "season", "week", "judge", "score"]
        for col in required_cols_scores:
            if col not in scores_long.columns:
                issues.append(f"scores_long.csv 缺少列: {col}")

        if not issues:
            print("✓ Task3 所需列全部存在")

    except Exception as e:
        issues.append(f"Task3 数据加载失败: {e}")

    return {"issues": issues}


def main():
    """主验证流程"""
    print("\n" + "=" * 70)
    print("DWTS 数据验证报告")
    print("=" * 70)

    all_issues = []

    # 验证原始数据
    raw_results = validate_raw_data()

    # 验证处理后数据
    processed_results = validate_processed_data()

    # 验证一致性
    consistency_results = validate_data_consistency()
    all_issues.extend(consistency_results["issues"])

    # 验证任务使用
    task_results = validate_task_usage()
    all_issues.extend(task_results["issues"])

    # 总结
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)

    if all_issues:
        print(f"⚠ 发现 {len(all_issues)} 个问题:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        return False
    else:
        print("✓ 所有验证通过，数据处理正确！")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
