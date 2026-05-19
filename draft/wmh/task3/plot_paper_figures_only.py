"""
仅从已有 outputs 重新生成论文用三张图，并复制到 paper/img。
不依赖模型拟合或 pyarrow。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from visualization import (
    plot_coefficient_comparison,
    plot_dual_track_factor_distributions,
    plot_pro_scatter,
    plot_separation_rationale,
    plot_variance_decomposition,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS = SCRIPT_DIR / "outputs"
FIGURES = SCRIPT_DIR / "figures"
DATA_DIR = SCRIPT_DIR / "data"
PAPER_IMG = SCRIPT_DIR.parent.parent.parent / "paper" / "img"


def _load_fixed_effects(csv_path: Path) -> dict[str, dict]:
    df = pd.read_csv(csv_path, index_col=0)
    return {
        str(name): {
            "estimate": float(row["estimate"]),
            "std_error": float(row["std_error"]),
            "p_value": float(row["p_value"]),
        }
        for name, row in df.iterrows()
    }


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    PAPER_IMG.mkdir(parents=True, exist_ok=True)

    with open(OUTPUTS / "variance_decomposition.json", encoding="utf-8") as f:
        var_decomp = json.load(f)
    plot_variance_decomposition(var_decomp, FIGURES / "variance_decomposition.png")

    judge_effects = _load_fixed_effects(OUTPUTS / "judge_model_fixed_effects.csv")
    fan_effects = _load_fixed_effects(OUTPUTS / "fan_model_fixed_effects.csv")
    plot_coefficient_comparison(
        judge_effects,
        fan_effects,
        FIGURES / "coefficient_comparison.png",
    )

    pro_rankings = pd.read_csv(OUTPUTS / "pro_rankings.csv")
    plot_pro_scatter(pro_rankings, FIGURES / "pro_scatter.png", top_n_labels=8)

    panel_path = DATA_DIR / "panel_data.csv"
    if panel_path.exists():
        panel_df = pd.read_csv(panel_path)
        plot_dual_track_factor_distributions(
            panel_df,
            FIGURES / "dual_track_factor_distributions.png",
        )
        idi_df = pd.read_csv(OUTPUTS / "impact_divergence_index.csv")
        plot_separation_rationale(
            panel_df,
            idi_df,
            FIGURES / "separation_rationale.png",
        )
    else:
        print("  Skip separation_rationale and factor distributions (no data/panel_data.csv)")

    for name in [
        "variance_decomposition.png",
        "coefficient_comparison.png",
        "pro_scatter.png",
        "dual_track_factor_distributions.png",
        "separation_rationale.png",
    ]:
        src = FIGURES / name
        if src.exists():
            shutil.copy(src, PAPER_IMG / name)
            print(f"  Copied to paper/img: {name}")
    print("Done.")


if __name__ == "__main__":
    main()
