"""
Task 2: DiffProb and Fan Signal Amplification Index (FSAI) from ranks.csv.

1. DiffProb_t = 1{S} sum_s 1{e_t^(rank,s) != e_t^(pct,s)}:
   Proportion of (simulation) runs where the eliminated contestant under rank
   differs from under percentage. With a single run per week, DiffProb_t = 1
   if rank-eliminated != percent-eliminated, else 0.

2. Fan Signal Amplification Index (FSAI), computed only on disagreement weeks
   (weeks where rank and percentage eliminate different contestants, DiffProb_t = 1):
   - Score^{(J)} = judge_percent (judge-only score).
   - Score^{(rank)} = (n+1 - rank_rank)/n (aggregate under rank method, higher=better).
   - Score^{(pct)} = percent_sum = fan_percent + judge_percent (aggregate under pct method).
   - FSAI_t^{(m)} = Var_i(Score^{(m)} - Score^{(J)}) / Var_i(Score^{(m)}).
   - Delta FSAI_t = FSAI_t^{(pct)} - FSAI_t^{(rank)}.
     > 0: Percentage method amplifies fan voting more; < 0: Rank method more fan-oriented.

3. Stochastic-dominance bias (SD), computed only on disagreement weeks:
   - Fan–judge disagreement subgroup F_t: fan-favored (top q_F in fan rank) and
     judge-disfavored (bottom q_J in judge rank). Default q_F = q_J = 0.3.
   - P_bar_t^(m) = (1/|F_t|) sum_{i in F_t} P(e_t=i|m); P(e_t=i|m) = 1 if i eliminated under m else 0.
   - Delta_t^SD = P_bar_t^(rank) - P_bar_t^(pct). < 0: Rank more fan-biased; > 0: Pct more fan-biased.

4. Fan Sensitivity Index (FSI), computed only on disagreement weeks:
   - beta_t^(m) = OLS slope of Score^(m) on fan_percent in week t (beta = Cov(Score,F)/Var(F)).
   - Delta FSI_t = beta_t^(pct) - beta_t^(rank). > 0: Pct more fan-biased (higher sensitivity to fan).

5. Judge-Favorite Elimination Gap (JFEG), computed only on disagreement weeks:
   - Subgroup H_t: judge top q_J, fan bottom q_F (judge favorites that fans don't like).
   - P_bar_t^(m) = elimination probability in H_t under method m.
   - Delta JFEG_t = P_bar_t^(pct) - P_bar_t^(rank). > 0: Pct more fan-biased (eliminates judge favorites more).

Outputs: outputs/diffprob_and_sd_bias.json, outputs/diffprob_by_week.csv,
         outputs/fsai_by_week.csv, outputs/sd_gap_by_week.csv, outputs/fsi_by_week.csv,
         outputs/jfeg_by_week.csv (FSAI/SD/FSI/JFEG rows only for disagreement weeks)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RANKS_PATH = SCRIPT_DIR / "ranks.csv"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

Q_FAN_TOP = 0.3
Q_JUDGE_BOTTOM = 0.3
Q_JUDGE_TOP = 0.3
Q_FAN_BOTTOM = 0.3


def _parse_bool(s: str) -> bool:
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in ("true", "1", "yes")


def compute_diffprob(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    DiffProb_t = 1 if rank-eliminated != percent-eliminated in week t, else 0.
    Returns (by_week DataFrame with columns season, week, diffprob_t, rank_elim, pct_elim),
    and overall mean diffprob.
    """
    df = df.copy()
    df["rank_elim"] = df["rank_rank_is_eliminated"].map(_parse_bool)
    df["pct_elim"] = df["percent_rank_is_eliminated"].map(_parse_bool)

    # Only weeks with at least one elimination
    elim_weeks = df[df["rank_elim"] | df["pct_elim"]].groupby(["season", "week"])
    rank_elim_id = elim_weeks.apply(
        lambda g: g.loc[g["rank_elim"], "contestant_id"].iloc[0]
        if g["rank_elim"].any() else None
    ).reset_index(name="rank_elim_contestant")
    pct_elim_id = elim_weeks.apply(
        lambda g: g.loc[g["pct_elim"], "contestant_id"].iloc[0]
        if g["pct_elim"].any() else None
    ).reset_index(name="pct_elim_contestant")

    by_week = rank_elim_id.merge(pct_elim_id, on=["season", "week"])
    by_week["diffprob_t"] = (
        by_week["rank_elim_contestant"] != by_week["pct_elim_contestant"]
    ).astype(int)
    mean_diffprob = float(by_week["diffprob_t"].mean()) if len(by_week) else 0.0
    return by_week, mean_diffprob


def fan_judge_disagreement_subgroup(
    g: pd.DataFrame, q_fan: float = Q_FAN_TOP, q_judge: float = Q_JUDGE_BOTTOM
) -> pd.Series:
    """Boolean mask for F_t: fan-favored (top q_fan) and judge-disfavored (bottom q_judge)."""
    n = len(g)
    k_fan = max(1, int(n * q_fan))
    k_judge = max(1, int(n * q_judge))
    fan_top = g["fan_rank"] <= g["fan_rank"].nsmallest(k_fan).max()
    judge_bottom = g["judge_rank"] >= g["judge_rank"].nlargest(k_judge).min()
    return fan_top & judge_bottom


def judge_fan_disagreement_subgroup(
    g: pd.DataFrame, q_judge: float = Q_JUDGE_TOP, q_fan: float = Q_FAN_BOTTOM
) -> pd.Series:
    """Boolean mask for H_t: judge-favored (top q_judge in judge rank) and fan-disfavored (bottom q_fan in fan rank)."""
    n = len(g)
    k_judge = max(1, int(n * q_judge))
    k_fan = max(1, int(n * q_fan))
    judge_top = g["judge_rank"] <= g["judge_rank"].nsmallest(k_judge).max()
    fan_bottom = g["fan_rank"] >= g["fan_rank"].nlargest(k_fan).min()
    return judge_top & fan_bottom


def compute_sd_gap(
    df: pd.DataFrame, disagreement_keys: set[tuple[int, int]] | None = None
) -> pd.DataFrame:
    """
    Delta_t^SD = P_bar_t^(rank) - P_bar_t^(pct) over fan–judge disagreement subgroup F_t.
    If disagreement_keys is provided, only (season, week) in that set are computed.
    """
    df = df.copy()
    df["rank_elim"] = df["rank_rank_is_eliminated"].map(_parse_bool)
    df["pct_elim"] = df["percent_rank_is_eliminated"].map(_parse_bool)

    rows = []
    for (season, week), g in df.groupby(["season", "week"]):
        if disagreement_keys is not None and (season, week) not in disagreement_keys:
            continue
        if not (g["rank_elim"].any() and g["pct_elim"].any()):
            continue
        in_F = fan_judge_disagreement_subgroup(g)
        n_F = in_F.sum()
        if n_F == 0:
            rows.append({
                "season": season,
                "week": week,
                "n_contestants": len(g),
                "n_F": 0,
                "P_bar_rank": float("nan"),
                "P_bar_pct": float("nan"),
                "delta_SD": float("nan"),
            })
            continue
        rank_elim_id = g.loc[g["rank_elim"], "contestant_id"].iloc[0]
        pct_elim_id = g.loc[g["pct_elim"], "contestant_id"].iloc[0]
        F_contestants = set(g.loc[in_F, "contestant_id"])
        P_rank = 1.0 / n_F if rank_elim_id in F_contestants else 0.0
        P_pct = 1.0 / n_F if pct_elim_id in F_contestants else 0.0
        delta = P_rank - P_pct
        rows.append({
            "season": season,
            "week": week,
            "n_contestants": len(g),
            "n_F": int(n_F),
            "P_bar_rank": P_rank,
            "P_bar_pct": P_pct,
            "delta_SD": delta,
        })
    return pd.DataFrame(rows)


def compute_fsi(df: pd.DataFrame, disagreement_keys: set[tuple[int, int]] | None = None) -> pd.DataFrame:
    """
    FSI_t^(m) = beta_t^(m) = OLS slope of Score^(m) on fan_percent; beta = Cov(Score,F)/Var(F).
    Delta FSI_t = beta_t^(pct) - beta_t^(rank). > 0: Pct more fan-biased (higher sensitivity).
    """
    rows = []
    for (season, week), g in df.groupby(["season", "week"]):
        if disagreement_keys is not None and (season, week) not in disagreement_keys:
            continue
        n = len(g)
        if n < 2:
            rows.append({
                "season": season,
                "week": week,
                "n_contestants": n,
                "beta_rank": float("nan"),
                "beta_pct": float("nan"),
                "delta_fsi": float("nan"),
            })
            continue
        fan = g["fan_percent"].astype(float)
        score_pct = g["percent_sum"].astype(float)
        score_rank = (n + 1 - g["rank_rank"].astype(float)) / n
        var_fan = fan.var(ddof=1)
        if var_fan <= 0:
            rows.append({
                "season": season,
                "week": week,
                "n_contestants": n,
                "beta_rank": float("nan"),
                "beta_pct": float("nan"),
                "delta_fsi": float("nan"),
            })
            continue
        beta_rank = score_rank.cov(fan) / var_fan
        beta_pct = score_pct.cov(fan) / var_fan
        delta = beta_pct - beta_rank
        rows.append({
            "season": season,
            "week": week,
            "n_contestants": n,
            "beta_rank": beta_rank,
            "beta_pct": beta_pct,
            "delta_fsi": delta,
        })
    return pd.DataFrame(rows)


def compute_jfeg(
    df: pd.DataFrame, disagreement_keys: set[tuple[int, int]] | None = None
) -> pd.DataFrame:
    """
    Delta JFEG_t = P_bar_t^(pct) - P_bar_t^(rank) over H_t (judge top, fan bottom).
    > 0: Pct eliminates judge favorites (that fans don't like) more -> Pct more fan-biased.
    """
    df = df.copy()
    df["rank_elim"] = df["rank_rank_is_eliminated"].map(_parse_bool)
    df["pct_elim"] = df["percent_rank_is_eliminated"].map(_parse_bool)

    rows = []
    for (season, week), g in df.groupby(["season", "week"]):
        if disagreement_keys is not None and (season, week) not in disagreement_keys:
            continue
        if not (g["rank_elim"].any() and g["pct_elim"].any()):
            continue
        in_H = judge_fan_disagreement_subgroup(g)
        n_H = in_H.sum()
        if n_H == 0:
            rows.append({
                "season": season,
                "week": week,
                "n_contestants": len(g),
                "n_H": 0,
                "P_bar_rank": float("nan"),
                "P_bar_pct": float("nan"),
                "delta_jfeg": float("nan"),
            })
            continue
        rank_elim_id = g.loc[g["rank_elim"], "contestant_id"].iloc[0]
        pct_elim_id = g.loc[g["pct_elim"], "contestant_id"].iloc[0]
        H_contestants = set(g.loc[in_H, "contestant_id"])
        P_rank = 1.0 / n_H if rank_elim_id in H_contestants else 0.0
        P_pct = 1.0 / n_H if pct_elim_id in H_contestants else 0.0
        delta = P_pct - P_rank
        rows.append({
            "season": season,
            "week": week,
            "n_contestants": len(g),
            "n_H": int(n_H),
            "P_bar_rank": P_rank,
            "P_bar_pct": P_pct,
            "delta_jfeg": delta,
        })
    return pd.DataFrame(rows)


def compute_fsai(df: pd.DataFrame, disagreement_keys: set[tuple[int, int]] | None = None) -> pd.DataFrame:
    """
    FSAI_t^{(m)} = Var_i(Score^{(m)} - Score^{(J)}) / Var_i(Score^{(m)}).
    Delta FSAI_t = FSAI_t^{(pct)} - FSAI_t^{(rank)}.
    Score^{(J)} = judge_percent; Score^{(pct)} = percent_sum; Score^{(rank)} = (n+1-rank_rank)/n.
    If disagreement_keys is provided, only (season, week) in that set are computed.
    """
    rows = []
    for (season, week), g in df.groupby(["season", "week"]):
        if disagreement_keys is not None and (season, week) not in disagreement_keys:
            continue
        n = len(g)
        if n < 2:
            rows.append({
                "season": season,
                "week": week,
                "n_contestants": n,
                "fsai_rank": float("nan"),
                "fsai_pct": float("nan"),
                "delta_fsai": float("nan"),
            })
            continue
        score_J = g["judge_percent"].astype(float)
        score_pct = g["percent_sum"].astype(float)
        score_rank = (n + 1 - g["rank_rank"].astype(float)) / n

        var_rank = score_rank.var(ddof=1)
        var_pct = score_pct.var(ddof=1)
        var_rank_minus_J = (score_rank - score_J).var(ddof=1)
        var_pct_minus_J = (score_pct - score_J).var(ddof=1)

        fsai_rank = var_rank_minus_J / var_rank if var_rank > 0 else float("nan")
        fsai_pct = var_pct_minus_J / var_pct if var_pct > 0 else float("nan")
        delta = (fsai_pct - fsai_rank) if not (pd.isna(fsai_rank) or pd.isna(fsai_pct)) else float("nan")
        rows.append({
            "season": season,
            "week": week,
            "n_contestants": n,
            "fsai_rank": fsai_rank,
            "fsai_pct": fsai_pct,
            "delta_fsai": delta,
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(RANKS_PATH)

    by_week_diff, mean_diffprob = compute_diffprob(df)
    by_week_diff.to_csv(OUTPUTS_DIR / "diffprob_by_week.csv", index=False)
    n_weeks = len(by_week_diff)
    n_differ = int(by_week_diff["diffprob_t"].sum())

    disagreement_weeks = by_week_diff[by_week_diff["diffprob_t"] == 1]
    disagreement_keys = set(zip(disagreement_weeks["season"], disagreement_weeks["week"]))
    fsai_df = compute_fsai(df, disagreement_keys=disagreement_keys)
    fsai_df.to_csv(OUTPUTS_DIR / "fsai_by_week.csv", index=False)
    fsai_valid = fsai_df.dropna(subset=["delta_fsai"])
    mean_delta_fsai = float(fsai_valid["delta_fsai"].mean()) if len(fsai_valid) else float("nan")
    season_avg_fsai = (
        fsai_valid.groupby("season")["delta_fsai"].mean().to_dict()
        if len(fsai_valid) else {}
    )

    sd_df = compute_sd_gap(df, disagreement_keys=disagreement_keys)
    sd_df.to_csv(OUTPUTS_DIR / "sd_gap_by_week.csv", index=False)
    sd_valid = sd_df.dropna(subset=["delta_SD"])
    mean_delta_SD = float(sd_valid["delta_SD"].mean()) if len(sd_valid) else float("nan")
    season_avg_sd = (
        sd_valid.groupby("season")["delta_SD"].mean().to_dict()
        if len(sd_valid) else {}
    )

    fsi_df = compute_fsi(df, disagreement_keys=disagreement_keys)
    fsi_df.to_csv(OUTPUTS_DIR / "fsi_by_week.csv", index=False)
    fsi_valid = fsi_df.dropna(subset=["delta_fsi"])
    mean_delta_fsi = float(fsi_valid["delta_fsi"].mean()) if len(fsi_valid) else float("nan")
    season_avg_fsi = (
        fsi_valid.groupby("season")["delta_fsi"].mean().to_dict()
        if len(fsi_valid) else {}
    )

    jfeg_df = compute_jfeg(df, disagreement_keys=disagreement_keys)
    jfeg_df.to_csv(OUTPUTS_DIR / "jfeg_by_week.csv", index=False)
    jfeg_valid = jfeg_df.dropna(subset=["delta_jfeg"])
    mean_delta_jfeg = float(jfeg_valid["delta_jfeg"].mean()) if len(jfeg_valid) else float("nan")
    season_avg_jfeg = (
        jfeg_valid.groupby("season")["delta_jfeg"].mean().to_dict()
        if len(jfeg_valid) else {}
    )

    summary = {
        "diffprob": {
            "description": "Proportion of weeks where rank and percentage methods eliminate different contestants",
            "mean_diffprob": mean_diffprob,
            "n_weeks_with_elimination": n_weeks,
            "n_weeks_differ": n_differ,
        },
        "fsai": {
            "description": "Fan Signal Amplification Index (computed only on disagreement weeks); Delta FSAI_t = FSAI_t^(pct) - FSAI_t^(rank); >0 pct amplifies fan voting more, <0 rank more fan-oriented",
            "subsample": "disagreement_weeks_only",
            "n_disagreement_weeks": len(disagreement_keys),
            "mean_delta_fsai": mean_delta_fsai,
            "mean_delta_fsai_by_season": {str(k): v for k, v in season_avg_fsai.items()},
            "n_weeks": len(fsai_valid),
        },
        "stochastic_dominance": {
            "description": "Delta_t^SD = P_bar_t^(rank) - P_bar_t^(pct) over fan-judge disagreement subgroup F_t (computed only on disagreement weeks); <0 rank more fan-biased, >0 pct more fan-biased",
            "subsample": "disagreement_weeks_only",
            "q_fan_top": Q_FAN_TOP,
            "q_judge_bottom": Q_JUDGE_BOTTOM,
            "mean_delta_SD": mean_delta_SD,
            "mean_delta_SD_by_season": {str(k): v for k, v in season_avg_sd.items()},
            "n_weeks": len(sd_valid),
        },
        "fsi": {
            "description": "Fan Sensitivity Index (computed only on disagreement weeks); Delta FSI_t = beta_t^(pct) - beta_t^(rank), beta = OLS slope of Score on fan_percent; >0 pct more fan-biased",
            "subsample": "disagreement_weeks_only",
            "mean_delta_fsi": mean_delta_fsi,
            "mean_delta_fsi_by_season": {str(k): v for k, v in season_avg_fsi.items()},
            "n_weeks": len(fsi_valid),
        },
        "jfeg": {
            "description": "Judge-Favorite Elimination Gap (computed only on disagreement weeks); Delta JFEG_t = P_bar_t^(pct) - P_bar_t^(rank) over H_t (judge top, fan bottom); >0 pct more fan-biased",
            "subsample": "disagreement_weeks_only",
            "q_judge_top": Q_JUDGE_TOP,
            "q_fan_bottom": Q_FAN_BOTTOM,
            "mean_delta_jfeg": mean_delta_jfeg,
            "mean_delta_jfeg_by_season": {str(k): v for k, v in season_avg_jfeg.items()},
            "n_weeks": len(jfeg_valid),
        },
    }
    with open(OUTPUTS_DIR / "diffprob_and_sd_bias.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("DiffProb: mean =", mean_diffprob, f"({n_differ}/{n_weeks} weeks differ)")
    print("FSAI:     mean Delta FSAI =", mean_delta_fsai, "(disagreement weeks only)")
    print("SD:       mean Delta^SD   =", mean_delta_SD, "(disagreement weeks only)")
    print("FSI:      mean Delta FSI  =", mean_delta_fsi, "(disagreement weeks only)")
    print("JFEG:     mean Delta JFEG =", mean_delta_jfeg, "(disagreement weeks only)")
    print("Outputs:", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
