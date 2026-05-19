# %%
import polars as pl
import regex as re
from consts import DATA_PROCESSED, DATA_RAW

df_raw = pl.read_csv(
    DATA_RAW / "2026_MCM_Problem_C_Data.csv",
    null_values=["N/A", ""],
    infer_schema_length=1000,
)

df = df_raw

for col in df.columns:
    if re.match(r"week\d+_judge\d+_score", col):
        df = df.with_columns(pl.col(col).cast(pl.Float64))

df = df.with_columns(
    [
        pl.col("season").cast(pl.Int32),
        pl.col("placement").cast(pl.Int32),
        pl.col("celebrity_age_during_season").cast(pl.Int32),
    ]
)

# df = df.with_columns(pl.col("results").str.contains("Withdrew").alias("is_withdrew"))

# %%

base_info_cols = [
    "celebrity_name",
    "ballroom_partner",
    "celebrity_industry",
    "celebrity_homestate",
    "celebrity_homecountry/region",
    "celebrity_age_during_season",
    "season",
    "results",
    "placement",
]

df_contestants = df.select(base_info_cols)

df_contestants = df_contestants.with_columns(
    (pl.col("celebrity_name") + "_S" + pl.col("season").cast(pl.Utf8)).alias(
        "contestant_id"
    )
)

df_contestants.head(10)

df_contestants.write_csv(DATA_PROCESSED / "contestants.csv")

# %%
weeks = list(range(1, 12))
judges = list(range(1, 5))

records = []

for row in df.iter_rows(named=True):
    contestant_id = f"{row['celebrity_name']}_S{row['season']}"
    for week in weeks:
        for judge in judges:
            score_col = f"week{week}_judge{judge}_score"
            if score_col in df.columns:
                score = row[score_col]
                if score is not None:
                    records.append(
                        {
                            "contestant_id": contestant_id,
                            "celebrity_name": row["celebrity_name"],
                            "season": row["season"],
                            "week": week,
                            "judge": judge,
                            "score": score,
                        }
                    )

df_scores_long = pl.DataFrame(records)

df_scores_long.head(10)

df_scores_long.write_csv(DATA_PROCESSED / "scores_long.csv")

# %%

df_week_summary = (
    df_scores_long.filter(
        pl.col("score") > 0
    )  # eliminate the 0 points after elimination
    .group_by(["contestant_id", "celebrity_name", "season", "week"])
    .agg(
        [
            pl.sum("score").alias("total_score"),
            pl.mean("score").alias("avg_score"),
            pl.count("score").alias("judge_count"),
            pl.min("score").alias("min_score"),
            pl.max("score").alias("max_score"),
        ]
    )
    .sort(["season", "week", "total_score"], descending=[False, False, True])
)

df_week_summary.write_csv(DATA_PROCESSED / "week_summary.csv")

# Week Rank
df_week_summary = df_week_summary.with_columns(
    pl.col("total_score")
    .rank(method="ordinal", descending=True)
    .over(["season", "week"])
    .alias("week_rank")
)

df_week_summary.filter((pl.col("season") == 1) & (pl.col("week") == 4))

df_week_summary.write_csv(DATA_PROCESSED / "week_summary.csv")


# %% Feature Engineering

df_season_stats = df_week_summary.group_by(
    ["contestant_id", "celebrity_name", "season"]
).agg(
    [
        pl.len().alias("weeks_competed"),
        pl.sum("total_score").alias("season_total_score"),
        pl.mean("total_score").alias("season_avg_weekly_score"),
        pl.mean("avg_score").alias("season_avg_score"),
        pl.std("total_score").alias("score_stddev"),
        pl.mean("week_rank").alias("avg_rank"),
        (pl.col("week_rank") == 1).sum().alias("weeks_ranked_first"),
    ]
)

df_season_stats = df_season_stats.with_columns(
    pl.when(pl.col("season") <= 2)
    .then(pl.lit("rank_v1"))
    .when(pl.col("season") <= 27)
    .then(pl.lit("percentage"))
    .otherwise(pl.lit("rank_v2"))
    .alias("voting_method")
)

df_season_stats = df_season_stats.join(
    df_contestants.select(base_info_cols + ["contestant_id"]),
    on="contestant_id",
    how="left",
)

df_week_trend = (
    df_week_summary.sort(["contestant_id", "week"])
    .with_columns(
        [
            pl.col("total_score")
            .shift(1)
            .over("contestant_id")
            .alias("prev_week_score"),
            (
                pl.col("total_score").cum_sum().over("contestant_id")
                / (pl.int_range(1, pl.len() + 1)).over("contestant_id")
            ).alias("cum_avg_score"),
        ]
    )
    .with_columns(
        [
            (pl.col("total_score") - pl.col("prev_week_score")).alias("diff_prev_week"),
            (pl.col("total_score") - pl.col("cum_avg_score")).alias("diff_cum_avg"),
        ]
    )
    .sort(["season", "week"])
)

df_week_trend.write_csv(DATA_PROCESSED / "week_trend.csv")

df_partner_stats = (
    df_season_stats.filter(pl.col("results") != "Withdrew")
    .group_by("ballroom_partner")
    .agg(
        [
            pl.len().alias("total_partners"),
            pl.mean("placement").alias("avg_placement"),
            pl.min("placement").alias("best_placement"),
            (pl.col("placement") == 1).sum().alias("num_wins"),
            (pl.col("placement") <= 3).sum().alias("num_top3"),
            pl.mean("season_avg_score").alias("avg_score"),
        ]
    )
    .filter(pl.col("total_partners") >= 3)
    .sort("avg_placement")
)

df_partner_stats.write_csv(DATA_PROCESSED / "partner_stats.csv")

# %%

controversy_cases = [
    ("Jerry Rice", 2, "亚军但5周最低评委分"),
    ("Billy Ray Cyrus", 4, "第5名但6周最低评委分"),
    ("Bristol Palin", 11, "季军但12次最低评委分"),
    ("Bobby Bones", 27, "冠军但持续低评委分"),
]

controversy_contestant_ids = [
    f"{name}_S{season}" for name, season, _ in controversy_cases
]

df_season_stats = df_season_stats.with_columns(
    pl.col("contestant_id").is_in(controversy_contestant_ids).alias("is_controversy")
)

df_season_stats.filter(pl.col("is_controversy"))

df_season_stats = df_season_stats.sort(
    ["season"],
)

df_season_stats.write_csv(DATA_PROCESSED / "season_stats.csv")

# %%

df_elimination_stats = df_season_stats.select(
    ["weeks_competed", "season"]
).with_columns(
    pl.col("weeks_competed")
    .max()
    .over(["season"])
    .cast(pl.Int32)
    .alias("season_weeks"),
)

df_elimination_stats = df_elimination_stats.with_columns(
    pl.when(pl.col("season") <= 2)
    .then(pl.lit("rank_v1"))
    .when(pl.col("season") <= 27)
    .then(pl.lit("percentage"))
    .otherwise(pl.lit("rank_v2"))
    .alias("voting_method")
)

for i in range(1, 12):
    df_elimination_stats = df_elimination_stats.with_columns(
        pl.col("weeks_competed")
        .eq(i)
        .sum()
        .over("season")
        .alias(f"week_{i}_elimination")
    )

df_elimination_stats = (
    df_elimination_stats.drop(["weeks_competed"])
    .unique(subset=["season"])
    .sort(["season"])
)

df_elimination_stats.write_csv(DATA_PROCESSED / "elimination_stats.csv")
