"""
自适应 MCMC 采样器。

带缓存支持的自适应 Metropolis-Hastings 采样器，用于估计粉丝投票份额。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

try:
    from .data_structures import (
        ContestantInfo,
        MCMCConfig,
        MCMCDiagnostics,
        SeasonWeekData,
    )
    from .priors import (
        DEFAULT_INDUSTRY_PRIOR,
        DEFAULT_PRIOR_MEAN,
        PRIOR_STD,
        estimate_industry_priors_from_data,
    )
    from .voting import (
        check_elimination_constraint,
        check_finale_constraint,
        get_voting_method,
    )
except ImportError:
    from data_structures import (
        ContestantInfo,
        MCMCConfig,
        MCMCDiagnostics,
        SeasonWeekData,
    )
    from priors import (
        DEFAULT_INDUSTRY_PRIOR,
        DEFAULT_PRIOR_MEAN,
        PRIOR_STD,
        estimate_industry_priors_from_data,
    )
    from voting import (
        check_elimination_constraint,
        check_finale_constraint,
        get_voting_method,
    )


class AdaptiveMCMCSampler:
    """
    带缓存支持的自适应 MCMC 采样器，用于快速迭代。

    特性
    ----
    1. 自适应提议标准差，达到最优接受率（~35%）
    2. 稀疏采样以减少自相关
    3. 缓存机制避免重复运行未更改的赛季
    4. 异常检测功能，识别无法解释的淘汰
    """

    def __init__(
        self,
        season_week_data: list[SeasonWeekData],
        contestant_info: dict[str, ContestantInfo],
        config: MCMCConfig,
        cache_dir: Path | None = None,
        use_data_driven_priors: bool = True,
    ):
        self.data = season_week_data
        self.contestant_info = contestant_info
        self.config = config
        self.cache_dir = cache_dir

        # 按赛季组织数据
        self.seasons_data: dict[int, list[SeasonWeekData]] = {}
        for swd in season_week_data:
            self.seasons_data.setdefault(swd.season, []).append(swd)
        for season in self.seasons_data:
            self.seasons_data[season].sort(key=lambda x: x.week)

        # 先验估计
        self.prior_estimation_diagnostics = {}
        if use_data_driven_priors:
            self.industry_priors, self.prior_estimation_diagnostics = (
                estimate_industry_priors_from_data(contestant_info, season_week_data)
            )
        else:
            self.industry_priors = DEFAULT_INDUSTRY_PRIOR.copy()

        # 计算每个选手的先验
        self.prior_means: dict[str, float] = {}
        for cid, info in contestant_info.items():
            self.prior_means[cid] = self._compute_prior_mean(info.industry, info.age)

        # 计算先验哈希（用于缓存键区分不同先验）
        self.prior_hash = self._compute_prior_hash()

        self.rng = np.random.default_rng(config.random_seed)
        self.detected_anomalies: list[dict] = []

    def _compute_prior_mean(self, industry: str, age: float) -> float:
        """计算选手的先验均值。"""
        base = self.industry_priors.get(industry, DEFAULT_PRIOR_MEAN)
        age_factor = np.exp(-(((age - 40.0) / 30.0) ** 2))
        return base * (0.9 + 0.1 * age_factor)

    def _compute_prior_hash(self) -> str:
        """计算先验的哈希值，用于缓存键区分。"""
        import hashlib

        # 对行业先验字典排序后计算哈希
        prior_str = str(sorted(self.industry_priors.items()))
        return hashlib.md5(prior_str.encode()).hexdigest()[:8]

    def _get_cache_path(self, season: int) -> Path | None:
        """获取赛季的缓存文件路径（包含先验哈希）。"""
        if self.cache_dir is None:
            return None
        return (
            self.cache_dir
            / f"cache_s{season}_{self.config.to_hash()}_{self.prior_hash}.pkl"
        )

    def _load_cache(self, season: int) -> tuple | None:
        """加载缓存结果（如果可用）。"""
        cache_path = self._get_cache_path(season)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_cache(self, season: int, data: tuple):
        """保存结果到缓存。"""
        cache_path = self._get_cache_path(season)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

    def _find_feasible_solution(
        self, swd: SeasonWeekData, max_iterations: int = 2000
    ) -> tuple[np.ndarray | None, dict]:
        """
        寻找满足约束的可行解，或标记为异常。

        支持三种情况：
        1. 无人淘汰 - 不添加淘汰约束
        2. 单人淘汰 - 确保被淘汰者有最低综合得分
        3. 多人淘汰 - 确保所有被淘汰者都在最低 k 个综合得分中
        """
        n = len(swd.contestant_ids)
        diagnostics = {
            "season": swd.season,
            "week": swd.week,
            "n_contestants": n,
            "is_anomaly": False,
            "anomaly_type": None,
            "n_eliminated": len(swd.eliminated_ids) if swd.eliminated_ids else 0,
        }

        # 从先验初始化
        shares = np.array([
            self.prior_means.get(cid, DEFAULT_PRIOR_MEAN) for cid in swd.contestant_ids
        ])
        shares = np.maximum(shares, 0.01)
        shares = shares / shares.sum()

        # 淘汰约束（仅当有人被淘汰时）
        if swd.eliminated_ids:
            eliminated_indices = [
                swd.contestant_ids.index(eid)
                for eid in swd.eliminated_ids
                if eid in swd.contestant_ids
            ]

            if eliminated_indices:
                k = len(eliminated_indices)  # 被淘汰人数
                method = get_voting_method(swd.season)
                non_eliminated_indices = [
                    i for i in range(n) if i not in eliminated_indices
                ]

                if method == "percentage":
                    judge_pct = swd.judge_scores / swd.judge_scores.sum()

                    for iteration in range(max_iterations):
                        # 对于所有被淘汰者，确保他们的综合得分低于所有未被淘汰者
                        for e_idx in eliminated_indices:
                            for i in non_eliminated_indices:
                                margin = judge_pct[i] - judge_pct[e_idx]
                                if shares[e_idx] - shares[i] >= margin - 0.002:
                                    adjustment = (
                                        shares[e_idx] - shares[i] - margin
                                    ) / 2 + 0.005
                                    shares[e_idx] = max(
                                        0.001, shares[e_idx] - adjustment
                                    )
                                    shares[i] += adjustment

                        shares = np.maximum(shares, 0.001)
                        shares = shares / shares.sum()

                        is_sat, _, _ = check_elimination_constraint(
                            shares, swd.judge_scores, eliminated_indices, swd.season
                        )
                        if is_sat:
                            break
                    else:
                        # ========== 异常检测 ==========
                        diagnostics["is_anomaly"] = True
                        diagnostics["anomaly_type"] = "SURPRISING_ELIMINATION"
                        diagnostics["eliminated"] = [
                            swd.contestant_ids[e] for e in eliminated_indices
                        ]
                        diagnostics["judge_pct"] = judge_pct.tolist()

                        # 分析：被淘汰者的评委分数排名（取所有被淘汰者中最高的）
                        elim_judge_ranks = [
                            n - np.argsort(np.argsort(swd.judge_scores))[e_idx]
                            for e_idx in eliminated_indices
                        ]
                        diagnostics["eliminated_judge_ranks"] = [
                            int(r) for r in elim_judge_ranks
                        ]
                        diagnostics["best_eliminated_judge_rank"] = int(
                            min(elim_judge_ranks)
                        )
                        diagnostics["analysis"] = self._analyze_anomaly(
                            swd, eliminated_indices[0]
                        )

                        self.detected_anomalies.append(diagnostics.copy())
                        return None, diagnostics

                else:  # rank or rank_bottom2
                    # 给所有被淘汰者分配最低的份额
                    for e_idx in eliminated_indices:
                        shares[e_idx] = min(shares[e_idx], 0.002 / k)
                    shares = shares / shares.sum()
        # 无人淘汰时，不添加任何淘汰约束，保持先验份额

        # 决赛约束
        if swd.is_finale and swd.final_placements:
            method = get_voting_method(swd.season)
            placement_list = [
                (swd.contestant_ids.index(cid), place)
                for cid, place in swd.final_placements.items()
                if cid in swd.contestant_ids
            ]
            placement_list.sort(key=lambda x: x[1])

            if method == "percentage":
                judge_pct = swd.judge_scores / swd.judge_scores.sum()

                for _ in range(max_iterations):
                    for rank_pos in range(len(placement_list) - 1):
                        better_idx = placement_list[rank_pos][0]
                        worse_idx = placement_list[rank_pos + 1][0]
                        needed_diff = (
                            judge_pct[worse_idx] - judge_pct[better_idx]
                        ) + 0.005
                        current_diff = shares[better_idx] - shares[worse_idx]
                        if current_diff < needed_diff:
                            adjustment = (needed_diff - current_diff) / 2 + 0.003
                            shares[better_idx] += adjustment
                            shares[worse_idx] = max(
                                0.001, shares[worse_idx] - adjustment
                            )

                    shares = np.maximum(shares, 0.001)
                    shares = shares / shares.sum()

                    is_sat, _, _ = check_finale_constraint(
                        shares,
                        swd.judge_scores,
                        swd.final_placements,
                        swd.contestant_ids,
                        swd.season,
                    )
                    if is_sat:
                        break
            else:
                base_share = 1.0 / n
                for idx, place in placement_list:
                    bonus = (n - place + 1) * 0.05
                    shares[idx] = base_share + bonus
                shares = np.maximum(shares, 0.001)
                shares = shares / shares.sum()

        return shares, diagnostics

    def _analyze_anomaly(self, swd: SeasonWeekData, elim_idx: int) -> dict:
        """深度分析异常案例。"""
        n = len(swd.judge_scores)
        judge_pct = swd.judge_scores / swd.judge_scores.sum()
        elim_pct = judge_pct[elim_idx]

        # 被淘汰者需要多少粉丝支持才能存活？
        other_pcts = [judge_pct[i] for i in range(n) if i != elim_idx]
        min_other_pct = min(other_pcts)

        # 如果 elim_pct > min_other_pct，说明评委分数已经足够高
        # 被淘汰只能是因为极低的粉丝支持
        analysis = {
            "eliminated_judge_pct": float(elim_pct),
            "min_other_judge_pct": float(min_other_pct),
            "judge_advantage": float(elim_pct - min_other_pct),
        }

        if elim_pct > min_other_pct:
            analysis["required_negative_fan_gap"] = float(elim_pct - min_other_pct)

        return analysis

    def _sample_with_constraint(
        self, swd: SeasonWeekData, current: np.ndarray, proposal_std: float
    ) -> tuple[np.ndarray, bool]:
        """在约束域内采样。"""
        n = len(current)

        log_current = np.log(current + 1e-10)
        noise = self.rng.normal(0, proposal_std, n)
        log_proposal = log_current + noise

        proposal = np.exp(log_proposal - np.max(log_proposal))
        proposal = proposal / proposal.sum()
        proposal = np.maximum(proposal, 1e-6)
        proposal = proposal / proposal.sum()

        # 检查淘汰约束
        if swd.eliminated_ids:
            eliminated_indices = [
                swd.contestant_ids.index(eid)
                for eid in swd.eliminated_ids
                if eid in swd.contestant_ids
            ]
            is_sat, _, _ = check_elimination_constraint(
                proposal, swd.judge_scores, eliminated_indices, swd.season
            )
            if not is_sat:
                return current, False

        # 检查决赛约束
        if swd.is_finale and swd.final_placements:
            is_sat, _, _ = check_finale_constraint(
                proposal,
                swd.judge_scores,
                swd.final_placements,
                swd.contestant_ids,
                swd.season,
            )
            if not is_sat:
                return current, False

        return proposal, True

    def _compute_log_posterior(
        self, state: dict[int, np.ndarray], season: int
    ) -> float:
        """计算对数后验（降低了 smoothness_weight）。"""
        log_post = 0.0
        weeks_data = self.seasons_data[season]
        prev_state = None
        prev_swd = None

        for swd in weeks_data:
            popularities = state[swd.week]
            n = len(swd.contestant_ids)

            # 先验项
            for i, cid in enumerate(swd.contestant_ids):
                mean = self.prior_means.get(cid, DEFAULT_PRIOR_MEAN) / n
                log_post -= 0.5 * ((popularities[i] - mean) / (PRIOR_STD / n)) ** 2

            # 平滑性项（权重已降低到 0.5）
            if prev_state is not None and prev_swd is not None:
                for i, cid in enumerate(swd.contestant_ids):
                    if cid in prev_swd.contestant_ids:
                        prev_idx = prev_swd.contestant_ids.index(cid)
                        diff = popularities[i] - prev_state[prev_idx]
                        log_post -= self.config.smoothness_weight * diff**2

            prev_state = popularities
            prev_swd = swd

        return log_post

    def sample_season(
        self,
        season: int,
        return_samples: bool = False,
        use_cache: bool = True,
        save_warmup_every: int | None = None,
    ) -> tuple[
        dict[int, np.ndarray],
        MCMCDiagnostics,
        list | None,
        list | None,
        list | None,
    ]:
        """
        对单个赛季进行采样，支持缓存和稀疏采样。

        Parameters
        ----------
        season : int
            要采样的赛季编号。
        return_samples : bool
            是否返回完整的样本链。
        use_cache : bool
            是否使用缓存结果（如果可用）。
        save_warmup_every : int or None
            若为整数，则在 warmup 阶段每该步保存一次状态，用于轨迹图（返回第 4、5 项）。

        Returns
        -------
        tuple
            (后验均值, 诊断信息, 样本链或None, warmup样本或None, warmup迭代下标或None)
        """
        empty_extra: tuple[list | None, list | None] = (None, None)
        if season not in self.seasons_data:
            return {}, MCMCDiagnostics(0, 0, 0, 0, []), None, None, None

        # 尝试从缓存加载（带 warmup 的请求不读缓存）
        if use_cache and save_warmup_every is None:
            cached = self._load_cache(season)
            if cached is not None:
                posterior_mean, diagnostics, samples = cached
                if not return_samples:
                    samples = None
                return posterior_mean, diagnostics, samples, None, None

        weeks_data = self.seasons_data[season]

        # 初始化
        state: dict[int, np.ndarray] = {}
        anomalies_this_season = []

        for swd in weeks_data:
            feasible, diag = self._find_feasible_solution(swd)
            if feasible is None:
                n = len(swd.contestant_ids)
                state[swd.week] = np.ones(n) / n
                anomalies_this_season.append(diag)
            else:
                state[swd.week] = feasible

        n_total = self.config.n_warmup + self.config.n_samples
        samples: list[dict[int, np.ndarray]] = []
        warmup_samples: list[dict[int, np.ndarray]] = []
        warmup_iterations: list[int] = []

        proposal_std = self.config.initial_proposal_std
        n_accepted_window = 0
        n_total_window = 0

        current_log_post = self._compute_log_posterior(state, season)
        total_accepted = 0
        total_proposals = 0

        for i in range(n_total):
            for swd in weeks_data:
                new_shares, was_feasible = self._sample_with_constraint(
                    swd, state[swd.week], proposal_std
                )

                if was_feasible:
                    old_shares = state[swd.week]
                    state[swd.week] = new_shares

                    new_log_post = self._compute_log_posterior(state, season)
                    log_alpha = new_log_post - current_log_post

                    if np.log(self.rng.random()) < log_alpha:
                        current_log_post = new_log_post
                        total_accepted += 1
                        n_accepted_window += 1
                    else:
                        state[swd.week] = old_shares

                total_proposals += 1
                n_total_window += 1

            # 自适应调整（激进）
            if (i + 1) % self.config.adapt_interval == 0 and i < self.config.n_warmup:
                current_rate = n_accepted_window / max(n_total_window, 1)

                if current_rate > self.config.target_acceptance + 0.05:
                    proposal_std *= 1.3
                elif current_rate < self.config.target_acceptance - 0.05:
                    proposal_std *= 0.7

                proposal_std = np.clip(proposal_std, 0.02, 0.6)
                n_accepted_window = 0
                n_total_window = 0

            # warmup 阶段按 save_warmup_every 保存（用于轨迹图展示过程）
            if (
                save_warmup_every is not None
                and i < self.config.n_warmup
                and i % save_warmup_every == 0
            ):
                warmup_samples.append({k: v.copy() for k, v in state.items()})
                warmup_iterations.append(i)

            # 稀疏采样：预热后每 `thinning` 步保存一个
            if (
                i >= self.config.n_warmup
                and (i - self.config.n_warmup) % self.config.thinning == 0
            ):
                samples.append({k: v.copy() for k, v in state.items()})

        # 后验均值
        posterior_mean: dict[int, np.ndarray] = {}
        for week in state.keys():
            stacked = np.stack([s[week] for s in samples])
            posterior_mean[week] = np.mean(stacked, axis=0)

        # ESS 估计
        ess = self._estimate_ess(samples)
        acceptance_rate = total_accepted / max(total_proposals, 1)

        diagnostics = MCMCDiagnostics(
            acceptance_rate=acceptance_rate,
            effective_sample_size=ess,
            proposal_std_final=proposal_std,
            n_anomalies=len(anomalies_this_season),
            anomaly_cases=anomalies_this_season,
        )

        # 检查 ESS 是否过低，标记需要重采样
        diagnostics.low_ess_warning = ess < self.config.min_ess_threshold
        diagnostics.multimodal_suspected = ess < 20  # 极低 ESS 可能暗示多峰分布

        # 保存到缓存（仅 3 元组，不含 warmup）
        if use_cache:
            self._save_cache(season, (posterior_mean, diagnostics, samples))

        w_samp = warmup_samples if warmup_samples else None
        w_iter = warmup_iterations if warmup_iterations else None
        if return_samples:
            return posterior_mean, diagnostics, samples, w_samp, w_iter
        return posterior_mean, diagnostics, None, w_samp, w_iter

    def sample_season_with_ess_check(
        self,
        season: int,
        return_samples: bool = False,
        use_cache: bool = True,
        verbose: bool = False,
    ) -> tuple[dict[int, np.ndarray], MCMCDiagnostics, list | None]:
        """
        对单个赛季进行采样，并在 ESS 过低时自动增强采样。

        这是 sample_season 的增强版本，会检测低 ESS 并自动重采样。

        Parameters
        ----------
        season : int
            要采样的赛季编号
        return_samples : bool
            是否返回完整的样本链
        use_cache : bool
            是否使用缓存结果
        verbose : bool
            是否打印详细信息

        Returns
        -------
        tuple
            (后验均值, 诊断信息, 样本链或None)
        """
        # 首次采样
        posterior, diagnostics, samples, _, _ = self.sample_season(
            season, return_samples=True, use_cache=use_cache
        )

        # 检查 ESS
        if diagnostics.effective_sample_size >= self.config.min_ess_threshold:
            if not return_samples:
                samples = None
            return posterior, diagnostics, samples

        # ESS 过低，尝试增强采样
        if verbose:
            print(
                f"    [!] Season {season}: Low ESS ({diagnostics.effective_sample_size:.0f}), "
                f"attempting enhanced sampling..."
            )

        for attempt in range(self.config.max_resampling_attempts):
            multiplier = self.config.resampling_multiplier ** (attempt + 1)
            enhanced_config = self.config.get_enhanced_config(multiplier)

            # 使用增强配置创建临时采样器（使用独立缓存）
            temp_sampler = AdaptiveMCMCSampler(
                self.data,
                self.contestant_info,
                enhanced_config,
                cache_dir=self.cache_dir,  # 使用缓存（增强配置有不同的 hash）
                use_data_driven_priors=True,
            )
            temp_sampler.industry_priors = self.industry_priors.copy()
            temp_sampler.prior_means = self.prior_means.copy()

            # 仅对这个赛季重新采样（启用缓存）
            new_posterior, new_diagnostics, new_samples, _, _ = temp_sampler.sample_season(
                season, return_samples=True, use_cache=use_cache
            )

            if verbose:
                print(
                    f"      Attempt {attempt + 1}: ESS = {new_diagnostics.effective_sample_size:.0f}"
                )

            if new_diagnostics.effective_sample_size >= self.config.min_ess_threshold:
                # 更新诊断信息
                new_diagnostics.resampling_attempts = attempt + 1
                new_diagnostics.original_ess = diagnostics.effective_sample_size
                if not return_samples:
                    new_samples = None
                return new_posterior, new_diagnostics, new_samples

        # 所有尝试都失败，标记为多峰分布嫌疑
        if verbose:
            print(
                f"    [!] Season {season}: ESS remains low after {self.config.max_resampling_attempts} "
                f"attempts. Possible multimodal distribution."
            )

        diagnostics.multimodal_suspected = True
        diagnostics.resampling_attempts = self.config.max_resampling_attempts
        if not return_samples:
            samples = None
        return posterior, diagnostics, samples

    def _estimate_ess(self, samples: list[dict[int, np.ndarray]]) -> float:
        """估计有效样本量（改进版本）。"""
        if len(samples) < 10:
            return float(len(samples))

        first_week = list(samples[0].keys())[0]
        chain = np.array([s[first_week][0] for s in samples])

        n = len(chain)
        mean = np.mean(chain)
        var = np.var(chain)

        if var < 1e-10:
            return float(n)

        # 改进的 ESS：Geyer 的初始单调序列估计器
        max_lag = min(n // 2, 200)
        autocorr = np.zeros(max_lag)

        for lag in range(max_lag):
            autocorr[lag] = (
                np.mean((chain[: n - lag] - mean) * (chain[lag:] - mean)) / var
            )

        # 找到第一个负自相关
        autocorr_sum = 0.0
        for lag in range(1, max_lag):
            if autocorr[lag] < 0:
                break
            autocorr_sum += autocorr[lag]

        ess = n / (1 + 2 * autocorr_sum)
        return max(ess, 1.0)

    def sample_all_seasons(
        self,
        verbose: bool = True,
        return_samples: bool = False,
        use_cache: bool = True,
        adaptive_ess: bool = True,
    ) -> tuple[
        dict[tuple[int, int], np.ndarray],
        dict[int, MCMCDiagnostics],
        dict | None,
    ]:
        """
        对所有赛季进行采样，可选缓存和自适应 ESS 检查。

        Parameters
        ----------
        verbose : bool
            是否打印进度。
        return_samples : bool
            是否返回完整样本链。
        use_cache : bool
            是否使用缓存结果（如果可用）。
        adaptive_ess : bool
            是否对低 ESS 赛季自动增强采样。

        Returns
        -------
        tuple
            (所有后验分布, 所有诊断信息, 所有样本或None)
        """
        all_posteriors: dict[tuple[int, int], np.ndarray] = {}
        all_diagnostics: dict[int, MCMCDiagnostics] = {}
        all_samples = {} if return_samples else None

        seasons = sorted(self.seasons_data.keys())

        for season in seasons:
            if verbose:
                print(
                    f"  Sampling Season {season}/{seasons[-1]}...", end=" ", flush=True
                )

            # 选择采样方法：普通采样或带 ESS 检查的自适应采样
            if adaptive_ess:
                posterior, diagnostics, samples = self.sample_season_with_ess_check(
                    season, return_samples, use_cache, verbose=False
                )
            else:
                posterior, diagnostics, samples, _, _ = self.sample_season(
                    season, return_samples, use_cache
                )

            for week, pop in posterior.items():
                all_posteriors[(season, week)] = pop

            all_diagnostics[season] = diagnostics

            if return_samples and samples is not None:
                all_samples[season] = samples

            if verbose:
                cached_marker = "[cached]" if self._load_cache(season) else ""
                print(
                    f"accept={diagnostics.acceptance_rate:.1%}, "
                    f"σ={diagnostics.proposal_std_final:.3f}, "
                    f"ESS={diagnostics.effective_sample_size:.0f} {cached_marker}"
                )

        return all_posteriors, all_diagnostics, all_samples
