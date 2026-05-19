"""
Plot MCMC trace (chain) for one or more fan-share parameters.

1. Default: post-warmup only (from cache or run one season).
2. include_warmup=True: run a short chain that saves warmup, plot full trace
   (warmup + post) so the "process" (initial phase -> convergence) is visible.
Output: figures/mcmc_trace.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
CACHE_DIR = SCRIPT_DIR / "cache"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Same config as run_inference so cache key matches
N_WARMUP = 3000
N_SAMPLES = 10000
THINNING = 5

# Short run for "trace with warmup" figure (faster, shows process)
SHORT_WARMUP = 1500
SHORT_SAMPLES = 1500
SHORT_THINNING = 5
SAVE_WARMUP_EVERY = 75


def _load_sampler_and_samples(
    season: int,
    include_warmup: bool = False,
):
    """Build sampler and get samples (from cache or run). If include_warmup, run short chain with warmup saved."""
    from model import AdaptiveMCMCSampler, MCMCConfig
    from run_inference import load_data

    season_week_data, contestant_info, _, _ = load_data()
    if include_warmup:
        config = MCMCConfig(
            n_warmup=SHORT_WARMUP,
            n_samples=SHORT_SAMPLES,
            thinning=SHORT_THINNING,
            initial_proposal_std=0.15,
            target_acceptance=0.35,
            adapt_interval=50,
            smoothness_weight=0.5,
            random_seed=42,
        )
        use_cache = False
    else:
        config = MCMCConfig(
            n_warmup=N_WARMUP,
            n_samples=N_SAMPLES,
            thinning=THINNING,
            initial_proposal_std=0.15,
            target_acceptance=0.35,
            adapt_interval=50,
            smoothness_weight=0.5,
            random_seed=42,
        )
        use_cache = True

    sampler = AdaptiveMCMCSampler(
        season_week_data,
        contestant_info,
        config,
        cache_dir=CACHE_DIR,
        use_data_driven_priors=True,
    )
    _, _, samples, warmup_samples, warmup_iterations = sampler.sample_season(
        season,
        return_samples=True,
        use_cache=use_cache,
        save_warmup_every=SAVE_WARMUP_EVERY if include_warmup else None,
    )
    n_warmup = SHORT_WARMUP if include_warmup else N_WARMUP
    thin = SHORT_THINNING if include_warmup else THINNING
    return sampler, config, samples, warmup_samples, warmup_iterations, n_warmup, thin


def plot_mcmc_trace(
    season: int = 5,
    week_idx: int = 0,
    contestant_indices: tuple[int, ...] = (0, 1, 2),
    output_path: Path | None = None,
    include_warmup: bool = True,
) -> None:
    """
    Plot MCMC chains for selected (season, week, contestants).
    If include_warmup, run a short chain with warmup saved so the trace
    shows the process (initial phase -> convergence).

    Parameters
    ----------
    season : int
        Season number.
    week_idx : int
        Index of week in that season (0 = first week).
    contestant_indices : tuple of int
        Which contestant indices to plot (same week).
    output_path : Path or None
        Where to save the figure.
    include_warmup : bool
        If True, run short MCMC with warmup saved and plot full trace (warmup + post).
    """
    sampler, config, samples, warmup_samples, warmup_iterations, n_warmup, thin = (
        _load_sampler_and_samples(season, include_warmup=include_warmup)
    )
    if samples is None or len(samples) == 0:
        raise RuntimeError(
            f"No samples for season {season}. Run full inference first or ensure cache exists."
        )

    weeks_data = sampler.seasons_data[season]
    week_numbers = sorted(samples[0].keys())
    week = week_numbers[week_idx]
    swd = next(sw for sw in weeks_data if sw.week == week)
    n_contestants = len(swd.contestant_ids)
    indices = [i for i in contestant_indices if i < n_contestants]
    if not indices:
        indices = [0, min(1, n_contestants - 1), n_contestants - 1]
        indices = [i for i in indices if i >= 0][:3]

    if include_warmup and warmup_samples and warmup_iterations:
        iter_warmup = np.array(warmup_iterations)
        iter_post = n_warmup + np.arange(len(samples)) * thin
        iteration = np.concatenate([iter_warmup, iter_post])
        has_warmup = True
    else:
        iteration = n_warmup + np.arange(len(samples)) * thin
        has_warmup = False

    n_plots = len(indices)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.2 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.28)

    colors = ["#1565C0", "#C62828", "#2E7D32"]
    for p, (i, cidx) in enumerate(zip(indices, range(len(indices)))):
        if has_warmup and warmup_samples is not None:
            chain_warmup = np.array([s[week][i] for s in warmup_samples])
            chain_post = np.array([s[week][i] for s in samples])
            chain = np.concatenate([chain_warmup, chain_post])
        else:
            chain = np.array([s[week][i] for s in samples])
        ax = axes[p]
        ax.plot(iteration, chain, color=colors[cidx % len(colors)], alpha=0.85, linewidth=0.6)
        if has_warmup:
            ax.axvline(n_warmup, color="#757575", linestyle="--", linewidth=1.2, label="End warmup")
            ax.legend(loc="upper right", fontsize=9)
        ax.set_ylabel("Fan share", fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_xlim(iteration[0], iteration[-1])
        cid = swd.contestant_ids[i]
        short_name = cid.replace("_S" + str(season), "")[:20]
        ax.set_title(f"Season {season}, Week {week}: {short_name}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    axes[-1].set_xlabel(
        "Iteration (warmup → post-warmup)" if has_warmup else "Iteration (after warmup)",
        fontsize=11,
    )
    fig.suptitle(
        "MCMC trace: fan-share chains (warmup + post-warmup)"
        if has_warmup
        else "MCMC trace: fan-share chains (thinned)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    if output_path is None:
        output_path = FIGURES_DIR / "mcmc_trace.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    plot_mcmc_trace(
        season=5,
        week_idx=0,
        contestant_indices=(0, 1, 2),
        output_path=FIGURES_DIR / "mcmc_trace.png",
        include_warmup=True,
    )
    print(f"Figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
