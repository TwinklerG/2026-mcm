"""深度验证 - 检查数据处理中的关键细节"""

from pathlib import Path

import polars as pl

DATA_RAW = Path(__file__).parent / "raw"
DATA_PROCESSED = Path(__file__).parent / "processed"


def check_elimination_logic():
    """检查淘汰逻辑的处理是否正确"""
    print("=" * 70)
    print("1. 淘汰逻辑验证")
    print("=" * 70)

    # 加载原始数据和处理后数据
    df_raw = pl.read_csv(
        DATA_RAW / "2026_MCM_Problem_C_Data.csv",
        null_values=["N/A", "NA", ""],
        infer_schema_length=10000,
    )
    week_summary = pl.read_csv(DATA_PROCESSED / "week_summary.csv")
    contestants = pl.read_csv(DATA_PROCESSED / "contestants.csv")

    # 检查: 被淘汰后的0分是否被正确过滤
    # 以 Season 1 的 Trista Sutter 为例 (Eliminated Week 2)
    trista = contestants.filter(
        (pl.col("celebrity_name") == "Trista Sutter") & (pl.col("season") == 1)
    )
    if trista.height > 0:
        contestant_id = trista["contestant_id"][0]
        trista_weeks = week_summary.filter(pl.col("contestant_id") == contestant_id)
        print(f"✓ Trista Sutter (Eliminated Week 2):")
        print(f"  - week_summary 中的周数: {trista_weeks['week'].to_list()}")
        # 应该只有 week 1 和 week 2
        if max(trista_weeks["week"].to_list()) <= 2:
            print("  ✓ 淘汰后的周数被正确过滤")
        else:
            print("  ⚠ 淘汰后仍有记录！")

    # 检查原始数据中的0分
    # 找一个被淘汰的选手，检查原始数据
    score_cols = [col for col in df_raw.columns if "judge" in col and "score" in col]
    trista_raw = df_raw.filter(
        (pl.col("celebrity_name") == "Trista Sutter") & (pl.col("season") == 1)
    )
    if trista_raw.height > 0:
        # 检查 week3 的分数应该是 0
        week3_scores = [trista_raw[f"week3_judge{j}_score"][0] for j in range(1, 4)]
        print(f"  - 原始数据 week3 分数: {week3_scores}")
        if all(s == 0.0 for s in week3_scores if s is not None):
            print("  ✓ 原始数据中淘汰后确实是0分")


def check_voting_method_assignment():
    """检查投票方法的分类是否正确"""
    print("\n" + "=" * 70)
    print("2. 投票方法分类验证")
    print("=" * 70)

    season_stats = pl.read_csv(DATA_PROCESSED / "season_stats.csv")

    # 按题目说明验证
    # S1-2: rank_v1, S3-27: percentage, S28-34: rank_v2
    method_dist = (
        season_stats
        .group_by(["season", "voting_method"])
        .agg(pl.len().alias("count"))
        .sort("season")
    )

    print("投票方法分布:")
    for row in method_dist.iter_rows(named=True):
        season = row["season"]
        method = row["voting_method"]
        expected = (
            "rank_v1" if season <= 2 else "percentage" if season <= 27 else "rank_v2"
        )
        status = "✓" if method == expected else "⚠"
        print(f"  {status} Season {season:2d}: {method:12s} (expected: {expected})")


def check_week_rank_calculation():
    """检查每周排名计算是否正确"""
    print("\n" + "=" * 70)
    print("3. 周排名计算验证")
    print("=" * 70)

    week_summary = pl.read_csv(DATA_PROCESSED / "week_summary.csv")

    # 检查 Season 1, Week 4 的排名（题目示例中有提到）
    s1w4 = (
        week_summary
        .filter((pl.col("season") == 1) & (pl.col("week") == 4))
        .sort("week_rank")
        .select(["celebrity_name", "total_score", "week_rank"])
    )

    print("Season 1, Week 4 排名:")
    for row in s1w4.iter_rows(named=True):
        print(
            f"  Rank {row['week_rank']}: {row['celebrity_name']:20s} ({row['total_score']})"
        )

    # 验证排名应该按 total_score 降序
    scores = s1w4["total_score"].to_list()
    ranks = s1w4["week_rank"].to_list()
    is_correct = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    if is_correct:
        print("  ✓ 排名顺序正确（按总分降序）")
    else:
        print("  ⚠ 排名顺序有误！")


def check_bonus_scores():
    """检查bonus分数的处理"""
    print("\n" + "=" * 70)
    print("4. Bonus 分数处理验证")
    print("=" * 70)

    scores_long = pl.read_csv(DATA_PROCESSED / "scores_long.csv")

    # 找出所有超过10分的评分
    high_scores = scores_long.filter(pl.col("score") > 10).sort(
        "score", descending=True
    )

    print(f"超过10分的评分数量: {high_scores.height}")
    if high_scores.height > 0:
        print("示例（前10条）:")
        for row in high_scores.head(10).iter_rows(named=True):
            print(
                f"  {row['celebrity_name']:25s} S{row['season']:2d} W{row['week']:2d} "
                f"J{row['judge']}: {row['score']:.4f}"
            )
        print("  ✓ Bonus分数被保留（未截断）")


def check_special_cases():
    """检查特殊情况处理"""
    print("\n" + "=" * 70)
    print("5. 特殊情况验证")
    print("=" * 70)

    contestants = pl.read_csv(DATA_PROCESSED / "contestants.csv")
    season_stats = pl.read_csv(DATA_PROCESSED / "season_stats.csv")

    # 检查退赛选手
    withdrew = contestants.filter(pl.col("is_withdrew") == True)
    print(f"退赛选手数量: {withdrew.height}")
    for row in withdrew.iter_rows(named=True):
        print(f"  - {row['celebrity_name']:25s} S{row['season']:2d} ({row['results']})")

    # 检查争议案例标记
    controversy = season_stats.filter(pl.col("is_controversy_case") == True).sort(
        "season"
    )
    print(f"\n争议案例数量: {controversy.height}")
    for row in controversy.iter_rows(named=True):
        print(
            f"  - {row['celebrity_name']:20s} S{row['season']:2d} "
            f"Placement: {row['placement']}, Avg Score: {row['season_avg_judge_score']:.2f}"
        )

    # 验证争议案例应该是：Jerry Rice (S2), Billy Ray Cyrus (S4),
    # Bristol Palin (S11), Bobby Bones (S27)
    expected_cases = [
        "Jerry Rice_S2",
        "Billy Ray Cyrus_S4",
        "Bristol Palin_S11",
        "Bobby Bones_S27",
    ]
    actual_cases = set(controversy["contestant_id"].to_list())
    expected_set = set(expected_cases)

    if actual_cases == expected_set:
        print("  ✓ 争议案例标记正确")
    else:
        missing = expected_set - actual_cases
        extra = actual_cases - expected_set
        if missing:
            print(f"  ⚠ 缺少争议案例: {missing}")
        if extra:
            print(f"  ⚠ 多余争议案例: {extra}")


def check_partner_statistics():
    """检查舞伴统计的合理性"""
    print("\n" + "=" * 70)
    print("6. 舞伴统计验证")
    print("=" * 70)

    partner_stats = pl.read_csv(DATA_PROCESSED / "partner_stats.csv")

    print(f"统计的舞伴数量: {partner_stats.height} (>=3次合作)")
    print("\n前5名舞伴（按平均名次）:")
    top5 = partner_stats.head(5)
    for row in top5.iter_rows(named=True):
        print(
            f"  {row['ballroom_partner']:25s} "
            f"合作: {row['total_partners']:2d}次, "
            f"平均名次: {row['avg_placement']:.2f}, "
            f"冠军: {row['championships']}次"
        )

    # 验证：avg_placement 应该越小越好
    avg_placements = partner_stats["avg_placement"].to_list()
    if all(
        avg_placements[i] <= avg_placements[i + 1]
        for i in range(len(avg_placements) - 1)
    ):
        print("  ✓ 舞伴排序正确（按平均名次升序）")


def check_season_15_allstar():
    """检查 Season 15 全明星季的处理"""
    print("\n" + "=" * 70)
    print("7. Season 15 全明星季验证")
    print("=" * 70)

    contestants = pl.read_csv(DATA_PROCESSED / "contestants.csv")
    s15 = contestants.filter(pl.col("season") == 15)

    print(f"Season 15 选手数量: {s15.height}")
    print("选手名单:")
    for row in (
        s15
        .select(["celebrity_name", "placement"])
        .sort("placement")
        .iter_rows(named=True)
    ):
        print(f"  {row['placement']:2d}. {row['celebrity_name']}")

    # 题目说明 S15 是唯一的全明星季
    # 我们应该检查这些选手是否在其他季也出现过
    all_names = contestants["celebrity_name"].to_list()
    s15_names = s15["celebrity_name"].to_list()

    repeaters = [name for name in s15_names if all_names.count(name) > 1]
    print(f"\n参加多季的选手: {len(repeaters)}")
    if len(repeaters) > 0:
        print("  ✓ Season 15 确实是全明星季（选手重复参赛）")
    else:
        print("  ⚠ Season 15 选手没有重复参赛？")


def main():
    """运行所有深度验证"""
    print("\n" + "=" * 70)
    print("DWTS 数据深度验证报告")
    print("=" * 70 + "\n")

    check_elimination_logic()
    check_voting_method_assignment()
    check_week_rank_calculation()
    check_bonus_scores()
    check_special_cases()
    check_partner_statistics()
    check_season_15_allstar()

    print("\n" + "=" * 70)
    print("深度验证完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
