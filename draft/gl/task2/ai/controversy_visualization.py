"""
Controversial Contestants Analysis and Visualization

This module provides comprehensive analysis and visualization for controversial contestants
like Jerry Rice, Billy Ray Cyrus, Bristol Palin, and Bobby Bones.

Key analysis focuses on:
- Elimination probability differences between Percentage vs Rank methods
- Whether they benefit significantly under Rank method (high DiffProb)
- Whether Judges' Save rule alters their fate (high JudgeImpact)
- Whether their survival depends on long-tail samples (low stability)
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class ControversialContestantAnalyzer:
    """Analyzer for controversial contestants with comprehensive metrics and visualizations."""

    def __init__(self, ranks_file: str, output_dir: str = "res"):
        """
        Initialize the analyzer.

        Args:
            ranks_file: Path to the ranks.csv file
            output_dir: Directory to save visualizations
        """
        self.ranks_file = ranks_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load data
        self.df_ranks = pl.read_csv(ranks_file)

        # Define controversial contestants
        self.controversial_contestants = [
            "Jerry Rice",
            "Billy Ray Cyrus",
            "Bristol Palin",
            "Bobby Bones",
        ]

        # Define methods to analyze
        self.methods = ["fan_rank", "judge_rank", "rank_rank", "percent_rank"]

    def analyze_contestant_elimination_probabilities(
        self, contestant_name: str, n_simulations: int = 100, noise_level: float = 0.1
    ) -> Dict[str, pl.DataFrame]:
        """
        Analyze elimination probabilities for a specific contestant across all methods.

        Args:
            contestant_name: Name of the contestant to analyze
            n_simulations: Number of simulation runs
            noise_level: Level of noise for uncertainty

        Returns:
            Dictionary with elimination probabilities for each method
        """
        # Filter data for this contestant
        contestant_data = self.df_ranks.filter(
            pl.col("contestant_id").str.contains(contestant_name)
        )

        if len(contestant_data) == 0:
            print(f"No data found for contestant: {contestant_name}")
            return {}

        method_probs = {}

        for method in self.methods:
            prob_results = []

            for row in contestant_data.iter_rows(named=True):
                season, week = row["season"], row["week"]

                # Get all contestants for this week
                week_data = self.df_ranks.filter(
                    (pl.col("season") == season) & (pl.col("week") == week)
                )

                if len(week_data) == 0:
                    continue

                # Calculate base probabilities
                base_probs = week_data.with_columns(
                    [(pl.col("week_left") - pl.col(method) + 1).alias("base_prob")]
                ).with_columns([pl.col("base_prob") / pl.col("base_prob").sum()])

                # Run simulations
                elimination_probs = []

                for sim in range(n_simulations):
                    np.random.seed(sim + season * 1000 + week * 10000)
                    noise = np.random.normal(0, noise_level, len(base_probs))

                    noisy_probs = base_probs.with_columns(
                        [
                            (pl.col("base_prob") + pl.lit(noise))
                            .clip(0, 1)
                            .alias("noisy_prob")
                        ]
                    ).with_columns([pl.col("noisy_prob") / pl.col("noisy_prob").sum()])

                    # Get elimination probability for this contestant
                    contestant_prob = (
                        noisy_probs.filter(
                            pl.col("contestant_id").str.contains(contestant_name)
                        )["noisy_prob"].sum()
                        / n_simulations
                    )

                    elimination_probs.append(contestant_prob)

                prob_results.append(
                    {
                        "season": season,
                        "week": week,
                        "contestant": contestant_name,
                        "method": method,
                        "elimination_prob": np.mean(elimination_probs),
                        "elimination_prob_std": np.std(elimination_probs),
                    }
                )

            method_probs[method] = pl.DataFrame(prob_results)

        return method_probs

    def analyze_method_differences_for_contestant(
        self,
        contestant_name: str,
        method1: str = "rank_rank",
        method2: str = "percent_rank",
        n_simulations: int = 100,
    ) -> pl.DataFrame:
        """
        Analyze method differences for a specific contestant.

        Args:
            contestant_name: Name of the contestant
            method1: First method to compare
            method2: Second method to compare
            n_simulations: Number of simulations

        Returns:
            DataFrame with method difference analysis
        """
        contestant_data = self.df_ranks.filter(
            pl.col("contestant_id").str.contains(contestant_name)
        )

        if len(contestant_data) == 0:
            return pl.DataFrame()

        difference_results = []

        for row in contestant_data.iter_rows(named=True):
            season, week = row["season"], row["week"]

            week_data = self.df_ranks.filter(
                (pl.col("season") == season) & (pl.col("week") == week)
            )

            if len(week_data) == 0:
                continue

            # Calculate probabilities for both methods
            method1_probs = week_data.with_columns(
                [(pl.col("week_left") - pl.col(method1) + 1).alias("base_prob1")]
            ).with_columns([pl.col("base_prob1") / pl.col("base_prob1").sum()])

            method2_probs = week_data.with_columns(
                [(pl.col("week_left") - pl.col(method2) + 1).alias("base_prob2")]
            ).with_columns([pl.col("base_prob2") / pl.col("base_prob2").sum()])

            # Get contestant's probability under each method
            contestant_prob1 = method1_probs.filter(
                pl.col("contestant_id").str.contains(contestant_name)
            )["base_prob1"].sum()

            contestant_prob2 = method2_probs.filter(
                pl.col("contestant_id").str.contains(contestant_name)
            )["base_prob2"].sum()

            difference_results.append(
                {
                    "season": season,
                    "week": week,
                    "contestant": contestant_name,
                    "method1": method1,
                    "method2": method2,
                    "prob_method1": contestant_prob1,
                    "prob_method2": contestant_prob2,
                    "prob_difference": contestant_prob1 - contestant_prob2,
                    "benefit_from_method1": contestant_prob1 > contestant_prob2,
                }
            )

        return pl.DataFrame(difference_results)

    def analyze_judge_impact_for_contestant(
        self,
        contestant_name: str,
        method: str = "rank_rank",
        n_simulations: int = 100,
        save_season_start: int = 28,
    ) -> pl.DataFrame:
        """
        Analyze judge save impact for a specific contestant.

        Args:
            contestant_name: Name of the contestant
            method: Method to use for base prediction
            n_simulations: Number of simulations
            save_season_start: First season with judge save rule

        Returns:
            DataFrame with judge impact analysis
        """
        contestant_data = self.df_ranks.filter(
            (pl.col("contestant_id").str.contains(contestant_name))
            & (pl.col("season") >= save_season_start)
        )

        if len(contestant_data) == 0:
            return pl.DataFrame()

        impact_results = []

        for row in contestant_data.iter_rows(named=True):
            season, week = row["season"], row["week"]

            week_data = self.df_ranks.filter(
                (pl.col("season") == season) & (pl.col("week") == week)
            )

            if len(week_data) == 0:
                continue

            # Calculate base probabilities
            base_probs = (
                week_data.with_columns(
                    [(pl.col("week_left") - pl.col(method) + 1).alias("base_prob")]
                )
                .with_columns([pl.col("base_prob") / pl.col("base_prob").sum()])
                .select(["contestant_id", "base_prob", "judge_rank"])
            )

            # Run simulations
            impacts = 0
            theoretical_eliminations = 0
            actual_eliminations = 0

            for sim in range(n_simulations):
                np.random.seed(sim + season * 1000 + week * 10000)
                noise = np.random.normal(0, 0.1, len(base_probs))

                noisy_probs = base_probs.with_columns(
                    [
                        (pl.col("base_prob") + pl.lit(noise))
                        .clip(0, 1)
                        .alias("noisy_prob")
                    ]
                ).with_columns([pl.col("noisy_prob") / pl.col("noisy_prob").sum()])

                # Check if this contestant would be eliminated theoretically
                theoretical_eliminated = (
                    noisy_probs.sort("noisy_prob", descending=True)
                    .select("contestant_id")
                    .row(0)[0]
                )
                if theoretical_eliminated and contestant_name in theoretical_eliminated:
                    theoretical_eliminations += 1

                # Apply judge save rule
                bottom_two = noisy_probs.sort("noisy_prob", descending=True).head(2)
                saved_contestant = (
                    bottom_two.sort("judge_rank").select("contestant_id").row(0)[0]
                )
                actual_eliminated = (
                    bottom_two.filter(pl.col("contestant_id") != saved_contestant)
                    .select("contestant_id")
                    .row(0)[0]
                )

                if actual_eliminated and contestant_name in actual_eliminated:
                    actual_eliminations += 1

                # Check if judge save changed the outcome for this contestant
                if (
                    theoretical_eliminated
                    and contestant_name in theoretical_eliminated
                    and actual_eliminated
                    and contestant_name not in actual_eliminated
                ):
                    impacts += 1

            impact_prob = impacts / n_simulations
            theoretical_prob = theoretical_eliminations / n_simulations
            actual_prob = actual_eliminations / n_simulations

            impact_results.append(
                {
                    "season": season,
                    "week": week,
                    "contestant": contestant_name,
                    "judge_impact_prob": impact_prob,
                    "theoretical_elimination_prob": theoretical_prob,
                    "actual_elimination_prob": actual_prob,
                    "protected_by_judges": theoretical_prob > actual_prob,
                }
            )

        return pl.DataFrame(impact_results)

    def calculate_stability_for_contestant(
        self, contestant_name: str, method: str = "rank_rank", n_simulations: int = 100
    ) -> pl.DataFrame:
        """
        Calculate stability metrics for a specific contestant.

        Args:
            contestant_name: Name of the contestant
            method: Method to analyze
            n_simulations: Number of simulations

        Returns:
            DataFrame with stability analysis
        """
        contestant_data = self.df_ranks.filter(
            pl.col("contestant_id").str.contains(contestant_name)
        )

        if len(contestant_data) == 0:
            return pl.DataFrame()

        stability_results = []

        for row in contestant_data.iter_rows(named=True):
            season, week = row["season"], row["week"]

            week_data = self.df_ranks.filter(
                (pl.col("season") == season) & (pl.col("week") == week)
            )

            if len(week_data) == 0:
                continue

            # Calculate elimination probabilities across simulations
            base_probs = week_data.with_columns(
                [(pl.col("week_left") - pl.col(method) + 1).alias("base_prob")]
            ).with_columns([pl.col("base_prob") / pl.col("base_prob").sum()])

            elimination_probs = []

            for sim in range(n_simulations):
                np.random.seed(sim + season * 1000 + week * 10000)
                noise = np.random.normal(0, 0.1, len(base_probs))

                noisy_probs = base_probs.with_columns(
                    [
                        (pl.col("base_prob") + pl.lit(noise))
                        .clip(0, 1)
                        .alias("noisy_prob")
                    ]
                ).with_columns([pl.col("noisy_prob") / pl.col("noisy_prob").sum()])

                # Get contestant's elimination probability
                contestant_prob = noisy_probs.filter(
                    pl.col("contestant_id").str.contains(contestant_name)
                )["noisy_prob"].sum()

                elimination_probs.append(contestant_prob)

            # Calculate stability metrics
            max_prob = np.max(elimination_probs)
            mean_prob = np.mean(elimination_probs)
            std_prob = np.std(elimination_probs)

            # Stability score: how concentrated the probability mass is
            stability_score = max_prob  # Higher max prob = more stable

            stability_results.append(
                {
                    "season": season,
                    "week": week,
                    "contestant": contestant_name,
                    "method": method,
                    "stability_score": stability_score,
                    "mean_elimination_prob": mean_prob,
                    "std_elimination_prob": std_prob,
                    "max_elimination_prob": max_prob,
                    "min_elimination_prob": np.min(elimination_probs),
                    "depends_on_long_tail": std_prob
                    > 0.1,  # High variance suggests long-tail dependence
                }
            )

        return pl.DataFrame(stability_results)

    def plot_posterior_probability_bars(
        self, contestant_name: str, save_path: Optional[str] = None
    ) -> None:
        """
        Create posterior probability bar charts for a contestant.

        Args:
            contestant_name: Name of the contestant
            save_path: Path to save the plot
        """
        # Get elimination probabilities for all methods
        method_probs = self.analyze_contestant_elimination_probabilities(
            contestant_name
        )

        if not method_probs:
            print(f"No data found for contestant: {contestant_name}")
            return

        # Combine all data
        all_data = []
        for method, df in method_probs.items():
            if len(df) > 0:
                df_copy = df.clone()
                all_data.append(df_copy)

        if not all_data:
            print(f"No probability data found for contestant: {contestant_name}")
            return

        combined_df = pl.concat(all_data)

        # Create the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"Elimination Probabilities for {contestant_name}",
            fontsize=16,
            fontweight="bold",
        )

        methods = ["fan_rank", "judge_rank", "rank_rank", "percent_rank"]
        method_labels = ["Fan Rank", "Judge Rank", "Combined Rank", "Percentage"]

        for i, (method, label) in enumerate(zip(methods, method_labels)):
            ax = axes[i // 2, i % 2]

            method_data = combined_df.filter(pl.col("method") == method)

            if len(method_data) > 0:
                # Create bar plot
                weeks = [
                    f"S{row['season']}W{row['week']}"
                    for row in method_data.iter_rows(named=True)
                ]
                probs = method_data["elimination_prob"].to_list()
                std_errs = method_data["elimination_prob_std"].to_list()

                bars = ax.bar(weeks, probs, yerr=std_errs, capsize=5, alpha=0.7)
                ax.set_title(f"{label} Method", fontweight="bold")
                ax.set_ylabel("Elimination Probability")
                ax.tick_params(axis="x", rotation=45)

                # Color bars based on probability level
                for j, (bar, prob) in enumerate(zip(bars, probs)):
                    if prob > 0.3:
                        bar.set_color("red")
                    elif prob > 0.15:
                        bar.set_color("orange")
                    else:
                        bar.set_color("green")

        plt.tight_layout()

        if save_path is None:
            save_path = (
                self.output_dir
                / f"{contestant_name.replace(' ', '_')}_posterior_probs.png"
            )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Posterior probability plot saved to: {save_path}")

    def plot_method_differences_heatmap(
        self, contestant_name: str, save_path: Optional[str] = None
    ) -> None:
        """
        Create heatmap showing method differences for a contestant.

        Args:
            contestant_name: Name of the contestant
            save_path: Path to save the plot
        """
        # Get method difference data
        diff_data = self.analyze_method_differences_for_contestant(contestant_name)

        if len(diff_data) == 0:
            print(f"No difference data found for contestant: {contestant_name}")
            return

        # Create pivot table for heatmap
        heatmap_data = diff_data.pivot(
            values="prob_difference", index="week", on="season"
        ).fill_nan(0)

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(
            heatmap_data.to_pandas(),
            annot=True,
            cmap="RdBu_r",
            center=0,
            fmt=".3f",
            cbar_kws={"label": "Probability Difference (Rank - Percentage)"},
        )

        plt.title(
            f"Method Differences Heatmap for {contestant_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Season")
        plt.ylabel("Week")
        plt.tight_layout()

        if save_path is None:
            save_path = (
                self.output_dir
                / f"{contestant_name.replace(' ', '_')}_method_differences.png"
            )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Method differences heatmap saved to: {save_path}")

    def plot_radar_chart_comparison(
        self, contestant_names: List[str] = None, save_path: Optional[str] = None
    ) -> None:
        """
        Create radar chart comparing multiple contestants across three dimensions:
        stability, method divergence, and judge impact.

        Args:
            contestant_names: List of contestant names to compare
            save_path: Path to save the plot
        """
        if contestant_names is None:
            contestant_names = self.controversial_contestants

        # Calculate metrics for each contestant
        contestant_metrics = []

        for contestant in contestant_names:
            # Stability analysis
            stability_data = self.calculate_stability_for_contestant(contestant)
            avg_stability = (
                stability_data["stability_score"].mean()
                if len(stability_data) > 0
                else 0
            )

            # Method difference analysis
            diff_data = self.analyze_method_differences_for_contestant(contestant)
            avg_diff = (
                abs(diff_data["prob_difference"].mean()) if len(diff_data) > 0 else 0
            )

            # Judge impact analysis
            impact_data = self.analyze_judge_impact_for_contestant(contestant)
            avg_impact = (
                impact_data["judge_impact_prob"].mean() if len(impact_data) > 0 else 0
            )

            contestant_metrics.append(
                {
                    "contestant": contestant,
                    "stability": avg_stability,
                    "method_divergence": avg_diff,
                    "judge_impact": avg_impact,
                }
            )

        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        # Set up the angles for the radar chart
        categories = ["Stability", "Method Divergence", "Judge Impact"]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Plot each contestant
        for metrics in contestant_metrics:
            values = [
                metrics["stability"],
                metrics["method_divergence"],
                metrics["judge_impact"],
            ]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, "o-", linewidth=2, label=metrics["contestant"])
            ax.fill(angles, values, alpha=0.25)

        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(
            "Controversial Contestants Comparison\n(Stability, Method Divergence, Judge Impact)",
            size=14,
            fontweight="bold",
            pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "controversial_contestants_radar.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Radar chart comparison saved to: {save_path}")

    def generate_comprehensive_report(
        self, contestant_name: str, output_prefix: Optional[str] = None
    ) -> None:
        """
        Generate comprehensive analysis report for a contestant.

        Args:
            contestant_name: Name of the contestant
            output_prefix: Prefix for output files
        """
        if output_prefix is None:
            output_prefix = contestant_name.replace(" ", "_")

        print(f"\n=== Comprehensive Analysis for {contestant_name} ===")

        # 1. Posterior probability analysis
        print("1. Analyzing elimination probabilities...")
        self.plot_posterior_probability_bars(
            contestant_name,
            save_path=self.output_dir / f"{output_prefix}_posterior_probs.png",
        )

        # 2. Method differences
        print("2. Analyzing method differences...")
        self.plot_method_differences_heatmap(
            contestant_name,
            save_path=self.output_dir / f"{output_prefix}_method_differences.png",
        )

        # 3. Stability analysis
        print("3. Analyzing stability...")
        stability_data = self.calculate_stability_for_contestant(contestant_name)
        if len(stability_data) > 0:
            avg_stability = stability_data["stability_score"].mean()
            long_tail_weeks = stability_data["depends_on_long_tail"].sum()
            print(f"   Average stability: {avg_stability:.3f}")
            print(
                f"   Weeks depending on long-tail: {long_tail_weeks}/{len(stability_data)}"
            )

            # Save stability data
            stability_data.write_csv(self.output_dir / f"{output_prefix}_stability.csv")

        # 4. Method differences analysis
        print("4. Analyzing method differences...")
        diff_data = self.analyze_method_differences_for_contestant(contestant_name)
        if len(diff_data) > 0:
            avg_diff = diff_data["prob_difference"].mean()
            benefit_weeks = diff_data["benefit_from_method1"].sum()
            print(
                f"   Average probability difference (Rank - Percentage): {avg_diff:.3f}"
            )
            print(
                f"   Weeks benefiting from Rank method: {benefit_weeks}/{len(diff_data)}"
            )

            # Save difference data
            diff_data.write_csv(
                self.output_dir / f"{output_prefix}_method_differences.csv"
            )

        # 5. Judge impact analysis
        print("5. Analyzing judge impact...")
        impact_data = self.analyze_judge_impact_for_contestant(contestant_name)
        if len(impact_data) > 0:
            avg_impact = impact_data["judge_impact_prob"].mean()
            protected_weeks = impact_data["protected_by_judges"].sum()
            print(f"   Average judge impact: {avg_impact:.3f}")
            print(f"   Weeks protected by judges: {protected_weeks}/{len(impact_data)}")

            # Save impact data
            impact_data.write_csv(self.output_dir / f"{output_prefix}_judge_impact.csv")

        print(f"\nAll analysis files saved to: {self.output_dir}")

    def run_all_controversial_contestants_analysis(self) -> None:
        """
        Run comprehensive analysis for all controversial contestants.
        """
        print("=== Controversial Contestants Analysis ===")

        # Generate individual reports
        for contestant in self.controversial_contestants:
            self.generate_comprehensive_report(contestant)

        # Generate comparison radar chart
        print("\n6. Creating comparison radar chart...")
        self.plot_radar_chart_comparison()

        print("\n=== Analysis Complete ===")
        print(f"All results saved to: {self.output_dir}")


def main():
    """Main function to run controversial contestants analysis."""
    from pathlib import Path

    # Setup paths
    current_dir = Path(__file__).parent
    ranks_file = current_dir.parent / "res" / "ranks.csv"
    output_dir = current_dir / "res" / "controversial_analysis"

    print("=== Controversial Contestants Analysis ===")
    print(f"Reading data from: {ranks_file}")

    # Check if ranks file exists
    if not ranks_file.exists():
        print(f"Error: Ranks file not found at {ranks_file}")
        print("Please run prepare.py first to generate ranks.csv")
        return

    # Initialize analyzer
    analyzer = ControversialContestantAnalyzer(str(ranks_file), str(output_dir))

    # Run comprehensive analysis
    analyzer.run_all_controversial_contestants_analysis()


if __name__ == "__main__":
    main()
