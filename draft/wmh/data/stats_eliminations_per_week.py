"""
Eliminations per week by season (MCM 2026 Problem C).

Weekly elimination counts are not fixed: this script summarizes
eliminations per (season, week) and outputs summary statistics.
"""

from pathlib import Path

import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_RAW = SCRIPT_DIR / "raw" / "2026_MCM_Problem_C_Data.csv"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    df = pl.read_csv(DATA_RAW, infer_schema_length=0).select(
        ["season", "results", "celebrity_name"]
    )
    df = df.with_columns(pl.col("season").cast(pl.Int32))
    df = df.with_columns(
        pl.col("results")
        .str.extract(r"Eliminated Week (\d+)", 1)
        .cast(pl.Int32)
        .alias("elim_week"),
    )
    eliminated = df.filter(pl.col("elim_week").is_not_null())
    by_season_week = (
        eliminated.group_by(["season", "elim_week"])
        .agg(pl.len().alias("n_eliminated"))
        .sort(["season", "elim_week"])
    )

    # Pivot: rows = season, columns = week (1..11), values = n_eliminated
    pivot = by_season_week.pivot(
        index="season",
        on="elim_week",
        values="n_eliminated",
        aggregate_function="first",
    ).sort("season")

    # Column order: season, then week 1, 2, ..., 11 (numeric order)
    week_cols = [str(w) for w in range(1, 12)]
    existing_weeks = [c for c in week_cols if c in pivot.columns]
    pivot = pivot.select(["season"] + existing_weeks)
    # Empty cell = that (season, week) had no elimination; fill with 0 for clarity
    pivot = pivot.with_columns(
        [pl.col(c).fill_null(0) for c in existing_weeks]
    )

    # Summary: for each week that appears, distribution of elimination counts
    week_counts = (
        by_season_week.group_by("elim_week")
        .agg(
            pl.col("n_eliminated").min().alias("min_per_season"),
            pl.col("n_eliminated").max().alias("max_per_season"),
            pl.col("n_eliminated").mean().alias("mean_per_season"),
            pl.len().alias("seasons_with_this_week"),
        )
        .sort("elim_week")
    )

    # Overall: distribution of n_eliminated (how often do we see 1, 2, ... eliminated in a week?)
    dist = (
        by_season_week.group_by("n_eliminated")
        .agg(pl.len().alias("count"))
        .sort("n_eliminated")
    )

    # Withdrew: count and list
    withdrew = df.filter(pl.col("results").str.to_lowercase().str.contains("withdrew"))
    n_withdrew = withdrew.height

    # Save CSV: season x week matrix
    out_csv = OUTPUTS_DIR / "eliminations_per_week.csv"
    pivot.write_csv(out_csv)
    print(f"Saved: {out_csv}")

    # Print summary
    print("\n--- Eliminations per (season, week) ---")
    print(pivot)

    print("\n--- By week: min/max/mean eliminations across seasons ---")
    print(week_counts)

    print("\n--- Distribution of elimination count per (season, week) ---")
    print(dist)

    print(f"\n--- Withdrew: {n_withdrew} contestants ---")
    if n_withdrew > 0:
        print(withdrew.select(["season", "celebrity_name", "results"]))

    # JSON for downstream
    import json

    out_json = OUTPUTS_DIR / "eliminations_per_week.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "by_week_summary": week_counts.to_dicts(),
                "distribution": dist.to_dicts(),
                "n_withdrew": n_withdrew,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
