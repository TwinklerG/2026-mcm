"""
DWTS 粉丝投票估计的数据结构和配置。

包含所有核心数据类和 MCMC 配置。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class SeasonWeekData:
    """单个赛季-周的数据。"""

    season: int
    week: int
    contestant_ids: list[str]
    judge_scores: np.ndarray
    eliminated_ids: list[str]
    is_finale: bool = False
    final_placements: dict[str, int] | None = None


@dataclass
class ContestantInfo:
    """选手元信息。"""

    contestant_id: str
    industry: str
    age: float
    partner: str


@dataclass
class MCMCConfig:
    """MCMC 配置 - 针对 ESS 优化。"""

    n_warmup: int = 3000
    n_samples: int = 10000
    thinning: int = 5  # 每 5 个样本保存一个，减少自相关
    initial_proposal_std: float = 0.15
    target_acceptance: float = 0.35  # 接近最优的 23.4%
    adapt_interval: int = 50
    smoothness_weight: float = 0.5  # 从 1.2 降低以改善混合
    random_seed: int = 42

    # 低 ESS 赛季的自适应配置
    min_ess_threshold: int = 100  # ESS 低于此阈值触发重采样
    max_resampling_attempts: int = 3  # 最大重采样次数
    resampling_multiplier: float = 2.0  # 重采样时增加的样本数倍数

    def to_hash(self) -> str:
        """生成用于缓存的哈希值。"""
        config_str = f"{self.n_warmup}_{self.n_samples}_{self.thinning}_{self.smoothness_weight}_{self.random_seed}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def get_enhanced_config(self, multiplier: float = 2.0) -> "MCMCConfig":
        """返回增强配置（更多样本）用于低 ESS 赛季。"""
        return MCMCConfig(
            n_warmup=int(self.n_warmup * multiplier),
            n_samples=int(self.n_samples * multiplier),
            thinning=max(1, self.thinning // 2),  # 减少稀疏化以保留更多样本
            initial_proposal_std=self.initial_proposal_std * 0.8,
            target_acceptance=self.target_acceptance,
            adapt_interval=self.adapt_interval,
            smoothness_weight=self.smoothness_weight * 0.7,  # 进一步降低以改善混合
            random_seed=self.random_seed + 1,
            min_ess_threshold=self.min_ess_threshold,
            max_resampling_attempts=self.max_resampling_attempts,
            resampling_multiplier=self.resampling_multiplier,
        )


@dataclass
class MCMCDiagnostics:
    """MCMC 诊断指标。"""

    acceptance_rate: float
    effective_sample_size: float
    proposal_std_final: float
    n_anomalies: int  # 改名: 不再是 "infeasible"，而是 "anomalies"
    anomaly_cases: list[dict]

    # 新增诊断字段
    low_ess_warning: bool = False
    multimodal_suspected: bool = False
    resampling_attempts: int = 0
    original_ess: float | None = None
