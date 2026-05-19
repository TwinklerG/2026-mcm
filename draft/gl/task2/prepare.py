import polars as pl
from consts import DATA_LINKED, DATA_PROCESSED, DATA_RES

df_fan_shares = pl.read_csv(
    DATA_LINKED / "fan_shares.csv",
)
df_scores = pl.read_csv(
    DATA_PROCESSED / "scores_long.csv",
)
df_week_trend = pl.read_csv(
    DATA_PROCESSED / "week_trend.csv",
)
df_fan_shares = df_fan_shares.rename({"estimated_fan_share": "fan_percent"})


df_scores = df_scores.drop("judge")

df_scores = df_scores.with_columns(
    pl.col("score").sum().over(["contestant_id", "season", "week"])
    / pl.col("score").sum().over(["season", "week"]),
).rename({"score": "judge_percent"})

df_scores = df_scores.unique(subset=["season", "week", "contestant_id"])

df_ranks = df_fan_shares.select(
    ["season", "week", "contestant_id", "fan_percent"]
).join(
    df_scores.select(["season", "week", "contestant_id", "judge_percent"]),
    on=["season", "week", "contestant_id"],
    how="left",
)

df_ranks = (
    df_ranks.with_columns(
        pl.len()
        .over(["season", "week"])
        .alias(
            "left_contestants",
        ),
        pl.col("fan_percent")
        .rank(descending=True)
        .over(["season", "week"])
        .alias("fan_rank")
        .cast(pl.Int32),
        pl.col("judge_percent")
        .rank(descending=True)
        .over(["season", "week"])
        .alias("judge_rank")
        .cast(pl.Int32),
    )
    .with_columns(
        (pl.col("fan_rank") + pl.col("judge_rank")).alias("rank_sum"),
        (pl.col("fan_percent") + pl.col("judge_percent")).alias("percent_sum"),
    )
    .with_columns(
        (
            pl.row_index().over(
                ["season", "week"],
                order_by=[pl.col("rank_sum"), -pl.col("fan_percent")],
            )
            + 1
        ).alias("rank_rank"),
        (
            pl.row_index().over(
                ["season", "week"],
                order_by=[-pl.col("percent_sum"), -pl.col("fan_percent")],
            )
            + 1
        ).alias("percent_rank"),
    )
    .with_columns(
        pl.col("rank_rank")
        .rank(descending=True)
        .over(["season", "week"])
        .cast(pl.Int32)
        .alias("rev_rank_rank"),
        pl.col("percent_rank")
        .rank(descending=True)
        .over(["season", "week"])
        .cast(pl.Int32)
        .alias("rev_percent_rank"),
    )
)


# df_ranks.filter((pl.col("season") == 1) & (pl.col("week") == 1))

df_ranks = df_ranks.join(
    df_week_trend.select(["contestant_id", "season", "week", "week_rank"]),
    on=["season", "week", "contestant_id"],
    how="left",
)

df_elimination_stats = pl.read_csv(DATA_PROCESSED / "elimination_stats.csv")

df_ranks = df_ranks.join(
    df_elimination_stats,
    on=["season"],
    how="left",
)

week_cols = [f"week_{i}_elimination" for i in range(1, 12)]
df_ranks = df_ranks.with_columns(
    pl.concat_list([pl.col(c) for c in week_cols]).alias("elim_list")
)
df_ranks = df_ranks.with_columns(
    pl.col("elim_list").list.get(pl.col("week") - 1).alias("week_elim")
)

for i in range(1, 12):
    df_ranks = df_ranks.drop(f"week_{i}_elimination")
df_ranks = df_ranks.drop("elim_list").with_columns(
    pl.col("week").max().over("season").alias("max_week_in_season"),
)

df_ranks = df_ranks.with_columns(
    (pl.col("rev_rank_rank") <= pl.col("week_elim")).alias("rank_rank_is_eliminated"),
    (pl.col("rev_percent_rank") <= pl.col("week_elim")).alias(
        "percent_rank_is_eliminated"
    ),
)

df_ranks.write_csv(DATA_RES / "ranks.csv")
