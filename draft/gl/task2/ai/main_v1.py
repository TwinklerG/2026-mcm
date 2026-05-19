# %%
import polars as pl
from consts import DATA_PROCESSED, DATA_RES

df_elimination_stats = pl.read_csv(DATA_PROCESSED / "elimination_stats.csv")
df_ranks = pl.read_csv(DATA_RES / "ranks.csv")

# %%
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

df_criteria = (
    df_ranks.with_columns(
        (
            pl.row_index().over(
                ["season", "week"],
                order_by=[pl.col("rev_rank_rank"), -pl.col("judge_rank")],
            )
            + 1
        ).alias("elim_rank_with_tiebreak"),
        (
            pl.row_index().over(
                ["season", "week"],
                order_by=[pl.col("rev_percent_rank"), -pl.col("judge_rank")],
            )
            + 1
        ).alias("elim_percent_with_tiebreak"),
    )
    .with_columns(
        (pl.col("elim_rank_with_tiebreak") <= pl.col("week_elim")).alias(
            "rank_rank_is_eliminated"
        ),
        (pl.col("elim_percent_with_tiebreak") <= pl.col("week_elim")).alias(
            "percent_rank_is_eliminated"
        ),
    )
    .drop(["elim_rank_with_tiebreak", "elim_percent_with_tiebreak"])
    # Bottom-Two Results are unknown, so we cannot use them to improve elimination prediction
    # .with_columns(
    #     (
    #         (pl.col("week_elim").gt(1) & pl.col("rank_rank_is_eliminated"))
    #         | (
    #             pl.col("week_elim").eq(1)
    #             & pl.col("rev_rank_rank").le(2)
    #             & pl.col("judge_rank").eq(
    #                 pl.col("judge_rank")
    #                 .filter(pl.col("rev_rank_rank").le(2))
    #                 .max()
    #                 .over(["season", "week"])
    #             )
    #         )
    #     ).alias("rank_v2_is_eliminated"),
    #     (
    #         (pl.col("week_elim").gt(1) & pl.col("percent_rank_is_eliminated"))
    #         | (
    #             pl.col("week_elim").eq(1)
    #             & pl.col("rev_percent_rank").le(2)
    #             & pl.col("judge_rank").eq(
    #                 pl.col("judge_rank")
    #                 .filter(pl.col("rev_percent_rank").le(2))
    #                 .max()
    #                 .over(["season", "week"])
    #             )
    #         )
    #     ).alias("percent_v2_is_eliminated"),
    # )
)
df_criteria = df_criteria.with_columns(
    pl.when(pl.col("week") == pl.col("max_week_in_season"))
    .then(pl.col("rank_rank"))
    .otherwise(None)
    # .cast(pl.Int32) # Missing values cannot be cast to Int32
    .alias("placement_rank_rank"),
    pl.when(pl.col("week") == pl.col("max_week_in_season"))
    .then(pl.col("percent_rank"))
    .otherwise(None)
    # .cast(pl.Int32)
    .alias("placement_percent_rank"),
)

df_criteria.write_csv(DATA_RES / "criteria.csv")
# %%
