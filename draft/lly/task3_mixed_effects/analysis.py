"""
模型结果分析模块。

深入分析 DT-HMEM 模型结果，生成论文所需的统计量和结论。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy import stats

try:
    from .model import DualTrackHMEM
except ImportError:
    from model import DualTrackHMEM


@dataclass
class AnalysisResults:
    """分析结果容器。"""

    # 核心发现
    pro_impact_significant: bool
    celebrity_impact_significant: bool
    impact_differs_between_tracks: bool

    # 关键统计量
    pro_variance_explained_judge: float
    pro_variance_explained_fan: float
    performance_conversion_rate: float
    performance_conversion_significant: bool

    # 详细结论
    conclusions: dict[str, str]


def analyze_fixed_effects(model: DualTrackHMEM) -> dict:
    """
    分析固定效应的影响。

    Parameters
    ----------
    model : DualTrackHMEM
        拟合后的模型

    Returns
    -------
    dict
        固定效应分析结果
    """
    results = {
        "judge_model": {},
        "fan_model": {},
        "comparisons": {},
    }

    # 分析评委模型
    if model.judge_model:
        for var in sorted(model.judge_model.fixed_effects.keys()):
            vals = model.judge_model.fixed_effects[var]
            if var == "Intercept":
                continue
            results["judge_model"][var] = {
                "effect": vals["estimate"],
                "significant": vals["p_value"] < 0.05,
                "direction": "positive" if vals["estimate"] > 0 else "negative",
            }

    # 分析粉丝模型
    if model.fan_model:
        for var in sorted(model.fan_model.fixed_effects.keys()):
            vals = model.fan_model.fixed_effects[var]
            if var == "Intercept":
                continue
            results["fan_model"][var] = {
                "effect": vals["estimate"],
                "significant": vals["p_value"] < 0.05,
                "direction": "positive" if vals["estimate"] > 0 else "negative",
            }

    # 比较两个模型
    idi_df = model.compute_impact_divergence_index()
    # 转换为 pandas 以便迭代
    idi_df_pd = idi_df.to_pandas()
    for _, row in idi_df_pd.iterrows():
        results["comparisons"][row["variable"]] = {
            "idi": row["IDI"],
            "significant_difference": row["significant"],
            "judge_effect": row["beta_judge"],
            "fan_effect": row["beta_fan"],
        }

    return results


def analyze_pro_effects(model: DualTrackHMEM, data: pd.DataFrame) -> dict:
    """
    深入分析职业舞伴效应。

    Parameters
    ----------
    model : DualTrackHMEM
        拟合后的模型
    data : pd.DataFrame
        原始数据

    Returns
    -------
    dict
        职业舞伴效应分析结果
    """
    rankings_pl = model.get_pro_rankings()
    rankings = rankings_pl.to_pandas()  # 转换为 pandas 以便后续操作
    var_decomp = model.compute_variance_decomposition()

    # 统计每个类别的舞伴数量
    category_counts = rankings["category"].value_counts().to_dict()

    # 找出 top performers
    top_tech = rankings.nlargest(5, "tech_boost")[["pro_id", "tech_boost", "category"]]
    top_pop = rankings.nlargest(5, "pop_boost")[["pro_id", "pop_boost", "category"]]
    top_overall = rankings.nlargest(5, "combined_score")[
        ["pro_id", "tech_boost", "pop_boost", "category"]
    ]

    # 计算职业舞伴与选手数量的关系
    pro_counts = (
        data
        .groupby("pro_id")
        .agg({"contestant_id": "nunique", "season": "nunique"})
        .reset_index()
    )
    pro_counts.columns = ["pro_id", "n_contestants", "n_seasons"]
    rankings_with_counts = rankings.merge(pro_counts, on="pro_id")

    # 检验经验是否与能力相关
    exp_tech_corr = stats.spearmanr(
        rankings_with_counts["n_contestants"], rankings_with_counts["tech_boost"]
    )
    exp_pop_corr = stats.spearmanr(
        rankings_with_counts["n_contestants"], rankings_with_counts["pop_boost"]
    )

    return {
        "variance_explained": {
            "judge_model_icc": var_decomp["judge"]["icc"],
            "fan_model_icc": var_decomp["fan"]["icc"],
            "judge_pro_pct": var_decomp["judge"]["pro_variance_pct"],
            "fan_pro_pct": var_decomp["fan"]["pro_variance_pct"],
        },
        "category_distribution": category_counts,
        "top_technicians": top_tech.to_dict("records"),
        "top_popularity": top_pop.to_dict("records"),
        "top_overall": top_overall.to_dict("records"),
        "experience_correlation": {
            "tech_vs_experience": {
                "correlation": exp_tech_corr.correlation,
                "p_value": exp_tech_corr.pvalue,
                "significant": exp_tech_corr.pvalue < 0.05,
            },
            "pop_vs_experience": {
                "correlation": exp_pop_corr.correlation,
                "p_value": exp_pop_corr.pvalue,
                "significant": exp_pop_corr.pvalue < 0.05,
            },
        },
    }


def analyze_celebrity_characteristics(model: DualTrackHMEM) -> dict:
    """
    分析明星特征的影响。

    Parameters
    ----------
    model : DualTrackHMEM
        拟合后的模型

    Returns
    -------
    dict
        明星特征分析结果
    """
    results = {"age_effect": {}, "industry_effects": {}, "week_effect": {}}

    # 年龄效应
    if model.judge_model and "age_centered" in model.judge_model.fixed_effects:
        j_age = model.judge_model.fixed_effects["age_centered"]
        f_age = model.fan_model.fixed_effects["age_centered"]

        results["age_effect"] = {
            "judge": {
                "coefficient": j_age["estimate"],
                "significant": j_age["p_value"] < 0.05,
            },
            "fan": {
                "coefficient": f_age["estimate"],
                "significant": f_age["p_value"] < 0.05,
            },
        }

    # 行业效应
    for var in model.judge_model.fixed_effects:
        if var.startswith("industry_"):
            industry = var.replace("industry_", "")
            j_ind = model.judge_model.fixed_effects[var]
            f_ind = (
                model.fan_model.fixed_effects[var]
                if var in model.fan_model.fixed_effects
                else None
            )

            results["industry_effects"][industry] = {
                "judge": {
                    "coefficient": j_ind["estimate"],
                    "significant": j_ind["p_value"] < 0.05,
                },
                "fan": (
                    {
                        "coefficient": f_ind["estimate"],
                        "significant": f_ind["p_value"] < 0.05,
                    }
                    if f_ind
                    else None
                ),
            }

    # 周次效应（学习曲线 vs 选票集中效应）
    if "week" in model.judge_model.fixed_effects:
        j_week = model.judge_model.fixed_effects["week"]
        f_week = model.fan_model.fixed_effects["week"]

        results["week_effect"] = {
            "judge": {
                "coefficient": j_week["estimate"],
                "significant": j_week["p_value"] < 0.05,
            },
            "fan": {
                "coefficient": f_week["estimate"],
                "significant": f_week["p_value"] < 0.05,
            },
        }

    # 运动员悖论分析 (The Athlete Paradox)
    if "Athlete" in results.get("industry_effects", {}):
        athlete = results["industry_effects"]["Athlete"]
        j_coef = athlete["judge"]["coefficient"] if athlete["judge"] else 0
        f_coef = athlete["fan"]["coefficient"] if athlete["fan"] else 0
        if j_coef < 0 and f_coef > 0:
            results["athlete_paradox"] = {
                "exists": True,
                "judge_coefficient": j_coef,
                "fan_coefficient": f_coef,
            }

    return results


def generate_paper_conclusions(
    model: DualTrackHMEM,
    fixed_analysis: dict,
    pro_analysis: dict,
    celebrity_analysis: dict,
) -> dict:
    """生成核心统计结论。"""
    gamma = model.get_performance_conversion_rate()

    conclusions = {
        "variance_explained": {
            "judge_icc": pro_analysis["variance_explained"]["judge_model_icc"],
            "fan_icc": pro_analysis["variance_explained"]["fan_model_icc"],
        },
        "performance_conversion": gamma if gamma else {},
        "pro_dancer_impact": {
            "kingmakers": [
                p["pro_id"]
                for p in pro_analysis["top_overall"]
                if p["category"] == "Kingmaker"
            ],
            "technicians": [
                p["pro_id"]
                for p in pro_analysis["top_technicians"]
                if p["category"] == "Technician"
            ],
            "category_distribution": pro_analysis["category_distribution"],
        },
        "celebrity_characteristics": celebrity_analysis.get("age_effect", {}),
        "athlete_paradox": celebrity_analysis.get("athlete_paradox", {}),
    }

    return conclusions


def _convert_to_json_serializable(obj, precision: int = 8):
    """递归转换对象为 JSON 可序列化格式，保持字典键排序和浮点精度。"""
    import numpy as np

    if isinstance(obj, dict):
        return {
            k: _convert_to_json_serializable(v, precision)
            for k in sorted(obj.keys())
            for k, v in [(k, obj[k])]
        }
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(i, precision) for i in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return round(float(obj), precision)
    elif hasattr(obj, "item"):  # numpy scalar
        val = obj.item()
        if isinstance(val, float):
            return round(float(val), precision)
        elif isinstance(val, bool):
            return bool(val)
        elif isinstance(val, int):
            return int(val)
        else:
            return val
    else:
        return obj


def save_analysis_results(
    output_dir: Path,
    fixed_analysis: dict,
    pro_analysis: dict,
    celebrity_analysis: dict,
    conclusions: dict,
) -> None:
    """
    保存分析结果。

    Parameters
    ----------
    output_dir : Path
        输出目录
    fixed_analysis : dict
        固定效应分析结果
    pro_analysis : dict
        职业舞伴分析结果
    celebrity_analysis : dict
        明星特征分析结果
    conclusions : dict
        论文结论
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "fixed_effects_analysis.json", "w", encoding="utf-8") as f:
        json.dump(
            _convert_to_json_serializable(fixed_analysis),
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    with open(output_dir / "pro_effects_analysis.json", "w", encoding="utf-8") as f:
        json.dump(
            _convert_to_json_serializable(pro_analysis),
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    with open(output_dir / "celebrity_analysis.json", "w", encoding="utf-8") as f:
        json.dump(
            _convert_to_json_serializable(celebrity_analysis),
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    with open(output_dir / "conclusions.json", "w", encoding="utf-8") as f:
        json.dump(
            _convert_to_json_serializable(conclusions),
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    print(f"分析结果已保存至: {output_dir}")
