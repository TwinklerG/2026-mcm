"""
Data Cleaning statistics for MCM 2026 Problem C (DWTS).

Computes counts for:
1. Missing value sources (judge slot not used vs post-elimination)
2. Special score values (0 = post-elimination, >10 = bonus)

Output: JSON for downstream tables.
"""

import json
import re
from pathlib import Path

import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_RAW = SCRIPT_DIR / "raw" / "2026_MCM_Problem_C_Data.csv"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def parse_score_columns(columns: list[str]) -> list[tuple[str, int, int]]:
    """Return (col_name, week, judge) for each score column, sorted by week, judge."""
    out: list[tuple[str, int, int]] = []
    for c in columns:
        m = re.match(r"week(\d+)_judge(\d+)_score", c)
        if m:
            out.append((c, int(m.group(1)), int(m.group(2))))
    return sorted(out, key=lambda x: (x[1], x[2]))


def main() -> None:
    df = pl.read_csv(
        DATA_RAW,
        null_values=["N/A", "NA", ""],
        infer_schema_length=0,
    )
    score_cols = [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]
    col_info = parse_score_columns(score_cols)
    total_cells = len(df) * len(score_cols)

    for c in score_cols:
        df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False))

    # Elimination week per row: from "Eliminated Week k" -> k; Withdrew -> last week with score; else 99
    elim_week_expr = (
        pl.col("results").str.extract(r"Eliminated Week (\d+)", 1).cast(pl.Int32)
    )
    df = df.with_columns(elim_week_expr.alias("_elim_week_parsed"))

    last_week_with_score = pl.max_horizontal(
        pl.when(pl.col(c).is_not_null() & (pl.col(c) > 0))
        .then(pl.lit(w))
        .otherwise(None)
        for c, w, _ in col_info
    )
    df = df.with_columns(last_week_with_score.alias("_last_week"))

    df = df.with_columns(
        pl.when(pl.col("_elim_week_parsed").is_not_null())
        .then(pl.col("_elim_week_parsed"))
        .when(pl.col("results").str.to_lowercase().str.contains("withdrew"))
        .then(pl.col("_last_week"))
        .otherwise(pl.lit(99))
        .alias("elim_week")
    )

    # Long format: one row per (contestant, score_cell) with week, elim_week, value
    row_indices: list[int] = []
    weeks: list[int] = []
    elim_weeks: list[int] = []
    values: list[float | None] = []
    for i in range(len(df)):
        elim_w = int(df.item(i, "elim_week"))
        for c, w, _ in col_info:
            row_indices.append(i)
            weeks.append(w)
            elim_weeks.append(elim_w)
            v = df.item(i, c)
            values.append(float(v) if v is not None and str(v).strip() else None)
    long = pl.DataFrame(
        {
            "row_idx": row_indices,
            "week": weeks,
            "elim_week": elim_weeks,
            "value": values,
        }
    )

    long = long.with_columns(
        (pl.col("week") > pl.col("elim_week")).alias("is_post_elim"),
        pl.col("value").is_null().alias("is_null"),
        (pl.col("value") == 0).alias("is_zero"),
    )
    long = long.with_columns(
        (pl.col("is_null") | pl.col("is_zero")).alias("is_null_or_zero"),
    )

    judge_slot_unused = long.filter(~pl.col("is_post_elim") & pl.col("is_null")).height
    post_elimination = long.filter(
        pl.col("is_post_elim") & pl.col("is_null_or_zero")
    ).height

    # Special values: 0, >10, (0, 10]
    all_vals = df.select(score_cols).to_numpy().flatten()
    numeric: list[float] = []
    for x in all_vals:
        if x is None:
            continue
        try:
            v = float(x)
            numeric.append(v)
        except (TypeError, ValueError):
            pass
    count_zero = sum(1 for x in numeric if x == 0)
    count_gt10 = sum(1 for x in numeric if x > 10)
    count_normal = sum(1 for x in numeric if 0 < x <= 10)

    missing_sources = [
        {
            "source": "Judge slot not used",
            "description": "Fewer judges in some seasons/weeks (e.g. Judge 4 absent)",
            "count": int(judge_slot_unused),
            "action": "Retained (no imputation)",
        },
        {
            "source": "Post-elimination",
            "description": "No score recorded after contestant eliminated",
            "count": int(post_elimination),
            "action": "Retained (no imputation)",
        },
    ]
    special_values = [
        {
            "value": "0",
            "meaning": "Post-elimination",
            "count": int(count_zero),
            "action": "Retained",
        },
        {
            "value": "> 10",
            "meaning": "Bonus (competition rule)",
            "count": int(count_gt10),
            "action": "Retained",
        },
        {
            "value": "(4, 10]",
            "meaning": "Normal judge score",
            "count": int(count_normal),
            "action": "Retained",
        },
    ]

    out = {
        "missing_sources": missing_sources,
        "total_score_cells": total_cells,
        "special_values": special_values,
    }

    out_path = OUTPUTS_DIR / "data_cleaning_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")

    # Typst table snippets for paper
    typst_missing = _typst_table_missing(missing_sources, total_cells)
    typst_special = _typst_table_special(special_values)
    typst_path = OUTPUTS_DIR / "data_cleaning_tables.typ"
    with open(typst_path, "w", encoding="utf-8") as f:
        f.write(typst_missing)
        f.write("\n\n")
        f.write(typst_special)
    print(f"Saved: {typst_path}")

    print("\n--- Table 1: Missing value sources ---")
    for r in missing_sources:
        print(f"  {r['source']}: {r['count']}  ({r['description']}) -> {r['action']}")
    print(f"  Total score cells: {total_cells}")

    print("\n--- Table 2: Special score values ---")
    for r in special_values:
        print(f"  {r['value']}: {r['count']}  ({r['meaning']}) -> {r['action']}")


def _typst_table_missing(rows: list[dict], total_cells: int) -> str:
    """Typst table: missing value sources."""
    lines = [
        "// Table 1: Missing value sources (paste into main.typ)",
        "#table(",
        "  columns: (2fr, 4fr, 1.2fr, 2fr),",
        "  inset: 8pt,",
        "  align: (left, left, center, left),",
        "  [*Source*], [*Description*], [*Count*], [*Action*],",
    ]
    for r in rows:
        src = r["source"].replace("'", "\\'")
        desc = r["description"].replace("'", "\\'")
        lines.append(
            f"  [{repr(src)}], [{repr(desc)}], [{r['count']}], [{repr(r['action'])}],"
        )
    lines.append(f"  [], [Total score cells], [{total_cells}], [],")
    lines.append("  caption: [Missing value sources and handling.],")
    lines.append(")")
    return "\n".join(lines)


def _typst_table_special(rows: list[dict]) -> str:
    """Typst table: special score values."""
    lines = [
        "// Table 2: Special score values (paste into main.typ)",
        "#table(",
        "  columns: (1.2fr, 3fr, 1.2fr, 1.5fr),",
        "  inset: 8pt,",
        "  align: (center, left, center, left),",
        "  [*Value*], [*Meaning*], [*Count*], [*Action*],",
    ]
    for r in rows:
        lines.append(
            f"  [{repr(r['value'])}], [{repr(r['meaning'])}], [{r['count']}], [{repr(r['action'])}],"
        )
    lines.append("  caption: [Special score values and handling.],")
    lines.append(")")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
