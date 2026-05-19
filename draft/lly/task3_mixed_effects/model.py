"""
双轨分层混合效应模型 (Dual-Track Hierarchical Mixed-Effects Model, DT-HMEM)。

本模块实现论文中的核心模型：
- Track 1: 评委评分模型 (Performance Model)
- Track 2: 粉丝投票模型 (Popularity Model)

模型特点：
- 使用线性混合效应模型捕捉层级结构
- 随机效应：职业舞伴、赛季
- 固定效应：年龄、性别、行业、周次
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
from scipy import stats


@dataclass
class ModelResults:
    """模型拟合结果容器。"""

    model_name: str
    fixed_effects: dict[str, dict[str, float]]  # {coef_name: {estimate, se, pvalue}}
    random_effects_variance: dict[str, float]  # {group: variance}
    blups: dict[str, pd.DataFrame]  # {group: DataFrame with BLUPs}
    residual_variance: float
    n_obs: int
    n_groups: dict[str, int]
    aic: float
    bic: float
    log_likelihood: float
    convergence: bool
    model_object: Any = None  # 原始模型对象


class DualTrackHMEM:
    """
    双轨分层混合效应模型。

    Parameters
    ----------
    data : pd.DataFrame
        面板数据，需包含以下列：
        - judge_score: 评委得分
        - log_fan_share: 对数粉丝份额
        - age_centered: 中心化年龄
        - week: 周次
        - industry_*: 行业哑变量
        - pro_id: 职业舞伴 ID
        - season_id: 赛季 ID
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.judge_model: ModelResults | None = None
        self.fan_model: ModelResults | None = None
        self._validate_data()

    def _validate_data(self) -> None:
        """验证数据完整性。"""
        required_cols = [
            "judge_score",
            "log_fan_share",
            "age_centered",
            "week",
            "pro_id",
            "season_id",
        ]
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f"缺少必要列: {missing}")

        # 移除缺失值
        self.data = self.data.dropna(subset=required_cols)

    def _get_industry_formula_terms(self) -> str:
        """获取行业哑变量的公式项。"""
        industry_cols = [
            col for col in self.data.columns if col.startswith("industry_")
        ]
        if industry_cols:
            return " + ".join([
                f"Q('{col}')" if " " in col else col for col in industry_cols
            ])
        return ""

    def fit_judge_model(self) -> ModelResults:
        """
        拟合评委评分模型 (Track 1: Performance Model)。

        模型形式：
        Y^J_{i,t} = β_0 + β_demo·X_i + β_time·Week_t + u_p + ε

        Returns
        -------
        ModelResults
            模型拟合结果
        """
        print("=" * 60)
        print("Track 1: 评委评分模型 (Performance Model)")
        print("=" * 60)

        # 过滤掉样本太少的 Pro
        pro_counts = self.data["pro_id"].value_counts()
        valid_pros = pro_counts[pro_counts >= 10].index
        data_filtered = self.data[self.data["pro_id"].isin(valid_pros)].copy()
        print(f"过滤后保留 {len(valid_pros)} 位职业舞伴（至少10个观测）")

        # 重新设置分类变量
        data_filtered["pro_id"] = data_filtered["pro_id"].cat.remove_unused_categories()

        # 构建固定效应公式
        industry_terms = self._get_industry_formula_terms()
        fixed_formula = "judge_score ~ age_centered + week"
        if industry_terms:
            fixed_formula += f" + {industry_terms}"

        print(f"固定效应公式: {fixed_formula}")
        print("随机效应: 职业舞伴 (pro_id)")

        try:
            model_pro = smf.mixedlm(
                fixed_formula,
                data_filtered,
                groups=data_filtered["pro_id"],
                re_formula="~1",  # 随机截距
            )
            result_pro = model_pro.fit(method="powell", maxiter=500)

            # 提取结果
            self.judge_model = self._extract_results(result_pro, "Judge Score Model")
            self._print_model_summary(self.judge_model)

            return self.judge_model

        except Exception as e:
            print(f"模型拟合失败: {e}")
            raise

    def fit_fan_model(self, include_judge_score: bool = True) -> ModelResults:
        """
        拟合粉丝投票模型 (Track 2: Popularity Model)。

        模型形式：
        Y^F_{i,t} = β_0 + β_demo·X_i + γ·Y^J_{i,t} + u_p + ε

        Parameters
        ----------
        include_judge_score : bool
            是否包含评委得分作为协变量

        Returns
        -------
        ModelResults
            模型拟合结果
        """
        print("\n" + "=" * 60)
        print("Track 2: 粉丝投票模型 (Popularity Model)")
        print("=" * 60)

        # 过滤掉样本太少的 Pro
        pro_counts = self.data["pro_id"].value_counts()
        valid_pros = pro_counts[pro_counts >= 10].index
        data_filtered = self.data[self.data["pro_id"].isin(valid_pros)].copy()
        data_filtered["pro_id"] = data_filtered["pro_id"].cat.remove_unused_categories()

        # 构建固定效应公式
        industry_terms = self._get_industry_formula_terms()
        fixed_formula = "log_fan_share ~ age_centered + week"
        if industry_terms:
            fixed_formula += f" + {industry_terms}"
        if include_judge_score:
            fixed_formula += " + judge_score"

        print(f"固定效应公式: {fixed_formula}")
        print("随机效应: 职业舞伴 (pro_id)")

        try:
            model_pro = smf.mixedlm(
                fixed_formula,
                data_filtered,
                groups=data_filtered["pro_id"],
                re_formula="~1",
            )
            result_pro = model_pro.fit(method="powell", maxiter=500)

            self.fan_model = self._extract_results(result_pro, "Fan Share Model")
            self._print_model_summary(self.fan_model)

            return self.fan_model

        except Exception as e:
            print(f"模型拟合失败: {e}")
            raise

    def run_gamma_sensitivity_analysis(self) -> dict:
        """
        γ (表现转化率) 敏感性分析：检验多重共线性风险。

        由于 judge_score 与 week 存在生存者偏差导致的正相关，
        当二者同时放入回归时，week 可能"吸走"judge_score 的解释力。

        本函数运行三种模型变体进行比较：
        1. 原模型（包含 week + judge_score）
        2. 不含 week 的模型
        3. 含 judge_score × week 交互项的模型

        Returns
        -------
        dict
            敏感性分析结果
        """
        print("\n" + "=" * 60)
        print("γ 敏感性分析: 检验多重共线性")
        print("=" * 60)

        results = {
            "models": {},
            "conclusion": "",
        }

        # 过滤数据
        pro_counts = self.data["pro_id"].value_counts()
        valid_pros = pro_counts[pro_counts >= 10].index
        data_filtered = self.data[self.data["pro_id"].isin(valid_pros)].copy()
        data_filtered["pro_id"] = data_filtered["pro_id"].cat.remove_unused_categories()

        industry_terms = self._get_industry_formula_terms()

        # 模型 1: 原模型 (week + judge_score)
        print("\n[1/3] 原模型 (week + judge_score)...")
        formula_1 = "log_fan_share ~ age_centered + week + judge_score"
        if industry_terms:
            formula_1 += f" + {industry_terms}"

        try:
            model_1 = smf.mixedlm(
                formula_1,
                data_filtered,
                groups=data_filtered["pro_id"],
                re_formula="~1",
            )
            res_1 = model_1.fit(method="powell", maxiter=500)
            gamma_1 = res_1.fe_params.get("judge_score", np.nan)
            pval_1 = res_1.pvalues.get("judge_score", np.nan)
            results["models"]["original"] = {
                "formula": formula_1,
                "gamma": float(gamma_1),
                "gamma_pvalue": float(pval_1),
                "gamma_significant": pval_1 < 0.05 if not np.isnan(pval_1) else False,
                "aic": float(res_1.aic),
            }
            print(f"  γ = {gamma_1:.4f}, p = {pval_1:.4f}")
        except Exception as e:
            print(f"  模型拟合失败: {e}")
            results["models"]["original"] = {"error": str(e)}

        # 模型 2: 不含 week
        print("\n[2/3] 不含 week 的模型...")
        formula_2 = "log_fan_share ~ age_centered + judge_score"
        if industry_terms:
            formula_2 += f" + {industry_terms}"

        try:
            model_2 = smf.mixedlm(
                formula_2,
                data_filtered,
                groups=data_filtered["pro_id"],
                re_formula="~1",
            )
            res_2 = model_2.fit(method="powell", maxiter=500)
            gamma_2 = res_2.fe_params.get("judge_score", np.nan)
            pval_2 = res_2.pvalues.get("judge_score", np.nan)
            results["models"]["no_week"] = {
                "formula": formula_2,
                "gamma": float(gamma_2),
                "gamma_pvalue": float(pval_2),
                "gamma_significant": pval_2 < 0.05 if not np.isnan(pval_2) else False,
                "aic": float(res_2.aic),
            }
            print(f"  γ = {gamma_2:.4f}, p = {pval_2:.4f}")
        except Exception as e:
            print(f"  模型拟合失败: {e}")
            results["models"]["no_week"] = {"error": str(e)}

        # 模型 3: 含交互项
        print("\n[3/3] 含 judge_score × week 交互项...")
        data_filtered["judge_week_interaction"] = (
            data_filtered["judge_score"] * data_filtered["week"]
        )
        formula_3 = (
            "log_fan_share ~ age_centered + week + judge_score + judge_week_interaction"
        )
        if industry_terms:
            formula_3 += f" + {industry_terms}"

        try:
            model_3 = smf.mixedlm(
                formula_3,
                data_filtered,
                groups=data_filtered["pro_id"],
                re_formula="~1",
            )
            res_3 = model_3.fit(method="powell", maxiter=500)
            gamma_3 = res_3.fe_params.get("judge_score", np.nan)
            pval_3 = res_3.pvalues.get("judge_score", np.nan)
            interaction_coef = res_3.fe_params.get("judge_week_interaction", np.nan)
            interaction_pval = res_3.pvalues.get("judge_week_interaction", np.nan)
            results["models"]["with_interaction"] = {
                "formula": formula_3,
                "gamma": float(gamma_3),
                "gamma_pvalue": float(pval_3),
                "gamma_significant": pval_3 < 0.05 if not np.isnan(pval_3) else False,
                "interaction_coef": float(interaction_coef),
                "interaction_pvalue": float(interaction_pval),
                "interaction_significant": interaction_pval < 0.05
                if not np.isnan(interaction_pval)
                else False,
                "aic": float(res_3.aic),
            }
            print(f"  γ = {gamma_3:.4f}, p = {pval_3:.4f}")
            print(f"  交互项系数 = {interaction_coef:.4f}, p = {interaction_pval:.4f}")
        except Exception as e:
            print(f"  模型拟合失败: {e}")
            results["models"]["with_interaction"] = {"error": str(e)}

        # 生成结论
        orig = results["models"].get("original", {})
        no_week = results["models"].get("no_week", {})

        if "error" not in orig and "error" not in no_week:
            orig_sig = orig.get("gamma_significant", False)
            no_week_sig = no_week.get("gamma_significant", False)

            if not orig_sig and no_week_sig:
                # 移除 week 后 γ 变显著：多重共线性
                results["conclusion"] = (
                    "多重共线性确认：移除 week 后 γ 变为显著。"
                    "结论应修正为 'Fans conflate longevity with quality'——"
                    "粉丝确实看重表现，但表现和持续时间混在一起无法区分。"
                )
                results["multicollinearity_detected"] = True
            elif not orig_sig and not no_week_sig:
                # 两种情况下都不显著：身份驱动结论稳健
                results["conclusion"] = (
                    "身份驱动游戏结论稳健：无论是否控制 week，γ 均不显著。"
                    "粉丝投票与当晚评委分数确实无关，完全是身份驱动的游戏。"
                )
                results["multicollinearity_detected"] = False
            elif orig_sig:
                results["conclusion"] = "原模型中 γ 显著，粉丝投票部分受表现影响。"
                results["multicollinearity_detected"] = False
            else:
                results["conclusion"] = "需要进一步分析。"
                results["multicollinearity_detected"] = None
        else:
            results["conclusion"] = "部分模型拟合失败，无法得出结论。"
            results["multicollinearity_detected"] = None

        print(f"\n结论: {results['conclusion']}")
        return results

    def _extract_results(self, result: Any, model_name: str) -> ModelResults:
        """
        从 statsmodels 结果中提取信息。

        Parameters
        ----------
        result : MixedLMResults
            statsmodels 拟合结果
        model_name : str
            模型名称

        Returns
        -------
        ModelResults
            提取的结果
        """
        # 固定效应
        fixed_effects = {}
        for name in result.fe_params.index:
            fixed_effects[name] = {
                "estimate": result.fe_params[name],
                "std_error": result.bse_fe[name] if name in result.bse_fe else np.nan,
                "z_value": result.tvalues[name] if name in result.tvalues else np.nan,
                "p_value": result.pvalues[name] if name in result.pvalues else np.nan,
            }

        # 随机效应方差
        random_effects_variance = {"pro_id": result.cov_re.iloc[0, 0]}

        # 提取 BLUPs（最佳线性无偏预测）
        blups = {"pro_id": pd.DataFrame(result.random_effects).T}
        blups["pro_id"].columns = ["intercept"]
        blups["pro_id"].index.name = "pro_id"
        blups["pro_id"] = blups["pro_id"].reset_index()

        return ModelResults(
            model_name=model_name,
            fixed_effects=fixed_effects,
            random_effects_variance=random_effects_variance,
            blups=blups,
            residual_variance=result.scale,
            n_obs=int(result.nobs),
            n_groups={"pro_id": len(result.random_effects)},
            aic=result.aic,
            bic=result.bic,
            log_likelihood=result.llf,
            convergence=result.converged,
            model_object=result,
        )

    def _print_model_summary(self, results: ModelResults) -> None:
        """打印模型摘要。"""
        print(f"\n--- {results.model_name} 摘要 ---")
        print(f"观测数: {results.n_obs}")
        print(f"职业舞伴数: {results.n_groups['pro_id']}")
        print(f"收敛: {'是' if results.convergence else '否'}")
        print(f"AIC: {results.aic:.2f}")
        print(f"BIC: {results.bic:.2f}")

        print("\n固定效应:")
        # 构建 DataFrame 以获得更好的打印格式
        fe_data = []
        for name in sorted(results.fixed_effects.keys()):
            vals = results.fixed_effects[name]
            sig = ""
            if vals["p_value"] < 0.001:
                sig = "***"
            elif vals["p_value"] < 0.01:
                sig = "**"
            elif vals["p_value"] < 0.05:
                sig = "*"
            fe_data.append({
                "Variable": name,
                "Estimate": vals["estimate"],
                "Std.Error": vals["std_error"],
                "z-value": vals["z_value"],
                "p-value": vals["p_value"],
                "Sig.": sig,
            })
        fe_df = pl.DataFrame(fe_data)
        print(fe_df)

        print("\n随机效应方差:")
        for group in sorted(results.random_effects_variance.keys()):
            var = results.random_effects_variance[group]
            print(f"  {group}: {var:.4f} (SD: {np.sqrt(var):.4f})")
        print(f"  残差: {results.residual_variance:.4f}")

    def compute_impact_divergence_index(self) -> pl.DataFrame:
        """
        计算影响差异度指数 (Impact Divergence Index, IDI)。

        用于检验特征对评委打分和粉丝投票的影响是否相同。

        IDI_k = |β^J_k - β^F_k| / sqrt(Var(β^J_k) + Var(β^F_k))

        Returns
        -------
        pl.DataFrame
            每个特征的 IDI 及显著性检验结果
        """
        if self.judge_model is None or self.fan_model is None:
            raise ValueError("请先拟合两个模型")

        results = []
        common_vars = set(self.judge_model.fixed_effects.keys()) & set(
            self.fan_model.fixed_effects.keys()
        )

        for var in sorted(common_vars):
            if var == "Intercept":
                continue

            j = self.judge_model.fixed_effects[var]
            f = self.fan_model.fixed_effects[var]

            # 标准化系数（除以各自的标准差）
            # 这里简化处理，直接比较原始系数
            diff = j["estimate"] - f["estimate"]
            se_diff = np.sqrt(j["std_error"] ** 2 + f["std_error"] ** 2)
            idi = abs(diff) / se_diff if se_diff > 0 else np.nan

            # 双侧 z 检验
            p_value = 2 * (1 - stats.norm.cdf(abs(idi)))

            results.append({
                "variable": var,
                "beta_judge": j["estimate"],
                "beta_fan": f["estimate"],
                "difference": diff,
                "se_difference": se_diff,
                "IDI": idi,
                "p_value": p_value,
                "significant": p_value < 0.05,
            })

        df = pl.DataFrame(results).sort("IDI", descending=True)
        return df

    def get_pro_rankings(self) -> pl.DataFrame:
        """
        获取职业舞伴排名。

        基于 BLUPs 将舞伴分为四类：
        - Kingmakers: 高技术加成，高流量加成
        - Technicians: 高技术，低流量
        - Fan Favorites: 低技术，高流量
        - Underperformers: 双低

        Returns
        -------
        pl.DataFrame
            职业舞伴排名及分类
        """
        if self.judge_model is None or self.fan_model is None:
            raise ValueError("请先拟合两个模型")

        # 获取 BLUPs
        judge_blups = self.judge_model.blups["pro_id"].copy()
        judge_blups.columns = ["pro_id", "tech_boost"]

        fan_blups = self.fan_model.blups["pro_id"].copy()
        fan_blups.columns = ["pro_id", "pop_boost"]

        # 合并
        rankings = judge_blups.merge(fan_blups, on="pro_id")

        # 标准化以便分类
        rankings["tech_boost_z"] = (
            rankings["tech_boost"] - rankings["tech_boost"].mean()
        ) / rankings["tech_boost"].std()
        rankings["pop_boost_z"] = (
            rankings["pop_boost"] - rankings["pop_boost"].mean()
        ) / rankings["pop_boost"].std()

        # 分类
        def classify_pro(row):
            if row["tech_boost_z"] > 0 and row["pop_boost_z"] > 0:
                return "Kingmaker"
            elif row["tech_boost_z"] > 0 and row["pop_boost_z"] <= 0:
                return "Technician"
            elif row["tech_boost_z"] <= 0 and row["pop_boost_z"] > 0:
                return "Fan Favorite"
            else:
                return "Underperformer"

        rankings["category"] = rankings.apply(classify_pro, axis=1)

        # 计算综合得分
        rankings["combined_score"] = (
            rankings["tech_boost_z"] + rankings["pop_boost_z"]
        ) / 2

        rankings_sorted = rankings.sort_values("combined_score", ascending=False)
        return pl.from_pandas(rankings_sorted)

    def compute_variance_decomposition(self) -> dict[str, dict[str, float]]:
        """
        计算方差分解。

        将总方差分解为：
        - 职业舞伴方差（Pro Variance）
        - 残差方差（Residual Variance）

        Returns
        -------
        dict
            方差分解结果
        """
        results = {}

        for model, name in [
            (self.judge_model, "judge"),
            (self.fan_model, "fan"),
        ]:
            if model is None:
                continue

            pro_var = model.random_effects_variance["pro_id"]
            resid_var = model.residual_variance
            total_var = pro_var + resid_var

            results[name] = {
                "pro_variance": pro_var,
                "residual_variance": resid_var,
                "total_variance": total_var,
                "pro_variance_pct": pro_var / total_var * 100,
                "icc": pro_var / total_var,  # 组内相关系数
            }

        return results

    def get_performance_conversion_rate(self) -> dict | None:
        """
        获取表现转化率 (γ)。

        γ 表示评委分数对粉丝投票的影响。
        - γ > 0 且显著：粉丝是理性的，投票给表现好的人
        - γ ≈ 0：粉丝投票与表现无关

        Returns
        -------
        dict | None
            转化率及其统计信息
        """
        if self.fan_model is None:
            return None

        if "judge_score" not in self.fan_model.fixed_effects:
            return None

        gamma = self.fan_model.fixed_effects["judge_score"]
        return {
            "gamma": gamma["estimate"],
            "std_error": gamma["std_error"],
            "z_value": gamma["z_value"],
            "p_value": gamma["p_value"],
            "significant": gamma["p_value"] < 0.05,
        }

    def compute_compositional_diagnostics(self) -> dict:
        """
        成分数据诊断：检验 Fan Share 归一化导致的负相关性。

        Fan Share 是归一化的（总和为 1），这意味着同一周内选手间存在
        结构性负相关，违反了残差独立性假设。

        本函数计算：
        1. 同周残差相关系数的估计
        2. 理论负相关强度 ρ ≈ -1/(N-1)
        3. 对标准误的影响评估

        Returns
        -------
        dict
            诊断结果
        """
        print("\n" + "=" * 60)
        print("成分数据诊断: 检验负相关性")
        print("=" * 60)

        if self.fan_model is None or self.fan_model.model_object is None:
            return {"error": "需要先拟合粉丝模型"}

        # 获取残差
        residuals = self.fan_model.model_object.resid

        # 从数据中识别同周观测
        data_with_residuals = self.data.copy()
        # 只保留用于拟合的观测
        pro_counts = data_with_residuals["pro_id"].value_counts()
        valid_pros = pro_counts[pro_counts >= 10].index
        data_filtered = data_with_residuals[
            data_with_residuals["pro_id"].isin(valid_pros)
        ].copy()

        # 添加残差
        if len(data_filtered) != len(residuals):
            return {"error": "残差长度不匹配"}

        data_filtered = data_filtered.reset_index(drop=True)
        data_filtered["residual"] = residuals

        # 计算每周的选手数
        week_counts = data_filtered.groupby(["season_id", "week"]).size()
        mean_contestants_per_week = week_counts.mean()

        # 理论负相关强度
        theoretical_rho = -1 / (mean_contestants_per_week - 1)

        # 计算同周残差的实际相关性（采样）
        correlations = []
        residual_values = []
        for (season, week), group in data_filtered.groupby(["season_id", "week"]):
            if len(group) < 3:
                continue
            resids = group["residual"].values
            residual_values.extend(resids)
            # 计算所有配对的相关
            n = len(resids)
            for i in range(n):
                for j in range(i + 1, n):
                    correlations.append(resids[i] * resids[j])

        if correlations and residual_values:
            # 这是残差乘积的均值，近似于协方差
            # 需要标准化为相关系数
            resid_var = np.var(residual_values)
            empirical_cov = np.mean(correlations)
            empirical_rho = empirical_cov / resid_var if resid_var > 1e-10 else 0.0
        else:
            empirical_rho = 0.0

        # 对 SE 的影响评估
        # 如果 ρ < 0，实际上 SE 被低估的程度约为 sqrt(1 + (N-1)*ρ)
        # 但由于 ρ ≈ -1/(N-1)，影响接近 sqrt(1 - 1) = 0，即 SE 被显著低估
        # 但这只在 ρ 接近理论值时成立
        se_inflation_factor = np.sqrt(
            max(0, 1 + (mean_contestants_per_week - 1) * empirical_rho)
        )

        results = {
            "mean_contestants_per_week": float(mean_contestants_per_week),
            "theoretical_negative_correlation": float(theoretical_rho),
            "empirical_residual_correlation": float(empirical_rho),
            "se_inflation_factor": float(se_inflation_factor),
            "n_week_episodes": int(len(week_counts)),
            "n_correlation_pairs": len(correlations),
        }

        # 生成结论
        if abs(empirical_rho - theoretical_rho) < 0.05:
            results["assessment"] = (
                f"经验残差相关性 ({empirical_rho:.3f}) 接近理论值 ({theoretical_rho:.3f})。"
                "成分数据的负相关性存在，但由于每周平均 "
                f"{mean_contestants_per_week:.0f} 位选手，单对选手间的负相关性足够弱 "
                f"(ρ ≈ {theoretical_rho:.3f})，可以近似满足独立性假设。"
            )
            results["concern_level"] = "low"
        elif empirical_rho < theoretical_rho:
            results["assessment"] = (
                f"经验残差相关性 ({empirical_rho:.3f}) 比理论值 ({theoretical_rho:.3f}) 更负。"
                "可能存在额外的竞争效应，建议谨慎解释显著性结果。"
            )
            results["concern_level"] = "medium"
        else:
            results["assessment"] = (
                f"经验残差相关性 ({empirical_rho:.3f}) 弱于理论值 ({theoretical_rho:.3f})。"
                "成分数据约束的影响可能已被模型部分吸收。"
            )
            results["concern_level"] = "low"

        print(f"  每周平均选手数: {mean_contestants_per_week:.1f}")
        print(f"  理论负相关: {theoretical_rho:.4f}")
        print(f"  经验残差相关: {empirical_rho:.4f}")
        print(f"  评估: {results['assessment']}")

        return results

    def save_results(self, output_dir: Path) -> None:
        """
        保存所有结果。

        Parameters
        ----------
        output_dir : Path
            输出目录
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存固定效应
        if self.judge_model:
            pd.DataFrame(self.judge_model.fixed_effects).T.to_csv(
                output_dir / "judge_model_fixed_effects.csv", float_format="%.8f"
            )
            judge_blups = self.judge_model.blups["pro_id"].sort_values("pro_id")
            judge_blups.to_csv(
                output_dir / "judge_model_blups.csv", index=False, float_format="%.8f"
            )

        if self.fan_model:
            pd.DataFrame(self.fan_model.fixed_effects).T.to_csv(
                output_dir / "fan_model_fixed_effects.csv", float_format="%.8f"
            )
            fan_blups = self.fan_model.blups["pro_id"].sort_values("pro_id")
            fan_blups.to_csv(
                output_dir / "fan_model_blups.csv", index=False, float_format="%.8f"
            )

        # 保存 IDI
        if self.judge_model and self.fan_model:
            idi_df = self.compute_impact_divergence_index()
            # 四舍五入浮点数列到 8 位小数
            idi_df = idi_df.with_columns([
                pl.col("beta_judge").round(8),
                pl.col("beta_fan").round(8),
                pl.col("difference").round(8),
                pl.col("se_difference").round(8),
                pl.col("IDI").round(8),
                pl.col("p_value").round(8),
            ])
            idi_df.sort("variable").write_csv(
                output_dir / "impact_divergence_index.csv"
            )

            # 保存职业舞伴排名
            pro_rankings = self.get_pro_rankings()
            # 四舍五入浮点数列到 8 位小数
            pro_rankings = pro_rankings.with_columns([
                pl.col("tech_boost").round(8),
                pl.col("pop_boost").round(8),
                pl.col("tech_boost_z").round(8),
                pl.col("pop_boost_z").round(8),
                pl.col("combined_score").round(8),
            ])
            pro_rankings.sort("combined_score", descending=True).write_csv(
                output_dir / "pro_rankings.csv"
            )

            # 保存方差分解
            var_decomp = self.compute_variance_decomposition()

            def _to_json_serializable(obj, precision: int = 8):
                """递归转换为 JSON 可序列化格式。"""
                import numpy as np

                if isinstance(obj, dict):
                    return {
                        k: _to_json_serializable(v, precision)
                        for k in sorted(obj.keys())
                        for k, v in [(k, obj[k])]
                    }
                elif isinstance(obj, list):
                    return [_to_json_serializable(i, precision) for i in obj]
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, (np.integer, int)):
                    return int(obj)
                elif isinstance(obj, (np.floating, float)):
                    return round(float(obj), precision)
                elif hasattr(obj, "item"):
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

            with open(output_dir / "variance_decomposition.json", "w") as f:
                json.dump(
                    _to_json_serializable(var_decomp), f, indent=2, sort_keys=True
                )

            # 保存转化率
            gamma = self.get_performance_conversion_rate()
            if gamma:
                with open(output_dir / "performance_conversion.json", "w") as f:
                    json.dump(_to_json_serializable(gamma), f, indent=2, sort_keys=True)

        print(f"\n结果已保存至: {output_dir}")
