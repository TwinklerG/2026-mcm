# %%
import polars as pl
from consts import DATA_PROCESSED, DATA_RAW

# %%
df_raw = pl.read_csv(
    DATA_RAW / "2026_MCM_Problem_C_Data.csv",
    null_values=["N/A", ""],
    infer_schema_length=1000,
)
df_contestants = pl.read_csv(
    DATA_PROCESSED / "contestants.csv",
    null_values=[""],
)
df_scores_long = pl.read_csv(
    DATA_PROCESSED / "scores_long.csv",
    null_values=[""],
)
df_week_summary = pl.read_csv(
    DATA_PROCESSED / "week_summary.csv",
    null_values=[""],
)
df_season_stats = pl.read_csv(
    DATA_PROCESSED / "season_stats.csv",
    null_values=[""],
)
df_partner_stats = pl.read_csv(
    DATA_PROCESSED / "partner_stats.csv",
    null_values=[""],
)

# %% 数据完整性检查
print("=" * 60)
print("🔍 数据完整性检查")
print("=" * 60)

print("\n📊 数据表概览:")
print(f"  - 原始数据: {df_raw.shape[0]} 行")
print(f"  - 选手信息表: {df_contestants.shape[0]} 行")
print(f"  - 周评分长表: {df_scores_long.shape[0]} 行")
print(f"  - 周汇总表: {df_week_summary.shape[0]} 行")
print(f"  - 季度统计表: {df_season_stats.shape[0]} 行")
print(f"  - 舞伴统计表: {df_partner_stats.shape[0]} 行")

print("\n🔑 关键字段缺失检查 (季度统计表):")
# for col in ["placement", "season_avg_score", "weeks_competed"]:
#     null_count = df_season_stats.filter(pl.col(col).is_null()).shape[0]
#     print(f"  {col}: {null_count} 缺失")

# %% 交叉验证：检查placement与results的一致性
print("\n🔄 结果一致性检查:")

# 检查冠军placement是否为1
champions = df_season_stats.filter(pl.col("results").str.contains("1st Place"))
champ_placements = champions["placement"].unique().to_list()
print(f"  冠军placement值: {champ_placements} (应为 [1])")

# 检查每季是否有且仅有一个冠军
champs_per_season = champions.group_by("season").agg(pl.len().alias("count"))
multi_champs = champs_per_season.filter(pl.col("count") > 1)
print(f"  多冠军季数: {multi_champs.shape[0]} (应为 0)")
