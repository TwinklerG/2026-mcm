"""
Monte Carlo Simulation for "Dancing with the Stars" Scoring Rules

This script compares two scoring rules:
- Rule A (Rank-based): Total Score = rank(Judges' Score) + rank(Fan Vote %)
- Rule B (Percentage-based): Total Score = Judges' Score + Fan Vote %

Author: MCM Team
Date: January 31, 2026
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")


def load_data(
    judges_file: str, fan_votes_file: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load judges scores and fan vote posterior samples from CSV files.

    Args:
        judges_file: Path to judges_scores.csv
        fan_votes_file: Path to fan_vote_posterior_samples.csv

    Returns:
        Tuple of (judges_df, fan_votes_df)
    """
    print("Loading data files...")

    # Load judges scores
    judges_df = pd.read_csv(judges_file)
    print(f"Loaded judges scores: {len(judges_df)} records")
    print(f"Judges data columns: {list(judges_df.columns)}")

    # Load fan vote posterior samples
    fan_votes_df = pd.read_csv(fan_votes_file)
    print(f"Loaded fan vote samples: {len(fan_votes_df)} records")
    print(f"Fan votes data columns: {list(fan_votes_df.columns)}")

    return judges_df, fan_votes_df


def merge_data_for_simulation(
    judges_df: pd.DataFrame, fan_votes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge judges scores with fan vote samples for simulation.

    Args:
        judges_df: DataFrame with judges scores
        fan_votes_df: DataFrame with fan vote samples

    Returns:
        Merged DataFrame ready for simulation
    """
    print("Merging data for simulation...")

    # Merge judges scores with fan vote samples
    merged_df = pd.merge(
        judges_df, fan_votes_df, on=["season", "week", "contestant_id"], how="inner"
    )

    print(f"Merged data: {len(merged_df)} records")
    print(f"Unique contestants: {merged_df['contestant_id'].nunique()}")
    print(f"Simulation runs per contestant: {merged_df['simulation_run'].nunique()}")

    return merged_df


def run_simulation_for_week(week_data: pd.DataFrame, n_simulations: int) -> list[tuple]:
    """
    Run Monte Carlo simulation for a single week.

    Args:
        week_data: DataFrame containing data for all contestants in a specific week
        n_simulations: Number of simulation runs

    Returns:
        List of tuples (eliminated_A, eliminated_B) for each simulation
    """
    results = []

    # Get unique contestants for this week
    contestants = week_data["contestant_id"].unique()

    for sim_run in range(1, n_simulations + 1):
        # Get data for current simulation run
        sim_data = week_data[week_data["simulation_run"] == sim_run].copy()

        if len(sim_data) == 0:
            continue

        # Extract scores and vote shares
        judges_scores = sim_data["judges_score"].values
        fan_votes = sim_data["vote_share"].values
        contestant_ids = sim_data["contestant_id"].values

        # Rule A (Rank-based): Total Score = rank(Judges) + rank(Fan Votes)
        # Lower rank is better, so we rank in descending order (higher scores get lower ranks)
        rank_judges = rankdata(-judges_scores, method="min")  # Lower rank = better
        rank_votes = rankdata(-fan_votes, method="min")  # Lower rank = better
        total_score_A = rank_judges + rank_votes

        # Rule B (Percentage-based): Total Score = Judges Score + Fan Vote %
        total_score_B = judges_scores + fan_votes

        # Find eliminated contestant (highest score for Rule A, lowest for Rule B)
        # For Rule A: Higher total score = worse performance = eliminated
        eliminated_idx_A = np.argmax(total_score_A)
        eliminated_A = contestant_ids[eliminated_idx_A]

        # For Rule B: Lower total score = worse performance = eliminated
        eliminated_idx_B = np.argmin(total_score_B)
        eliminated_B = contestant_ids[eliminated_idx_B]

        results.append((eliminated_A, eliminated_B))

    return results


def calculate_metrics(simulation_results: list[tuple]) -> dict:
    """
    Calculate outcome discrepancy probability and elimination stability metrics.

    Args:
        simulation_results: List of tuples (eliminated_A, eliminated_B)

    Returns:
        Dictionary with calculated metrics
    """
    if not simulation_results:
        return {}

    # Extract eliminated contestants for each rule
    eliminated_A = [result[0] for result in simulation_results]
    eliminated_B = [result[1] for result in simulation_results]

    # Calculate Outcome Discrepancy Probability (ODP)
    discrepancies = sum(1 for a, b in simulation_results if a != b)
    odp = discrepancies / len(simulation_results)

    # Calculate elimination stability for Rule A
    from collections import Counter

    counter_A = Counter(eliminated_A)
    most_common_A = counter_A.most_common(1)[0]
    stability_A = {
        "contestant": most_common_A[0],
        "prob": most_common_A[1] / len(simulation_results),
    }

    # Calculate elimination stability for Rule B
    counter_B = Counter(eliminated_B)
    most_common_B = counter_B.most_common(1)[0]
    stability_B = {
        "contestant": most_common_B[0],
        "prob": most_common_B[1] / len(simulation_results),
    }

    # Additional metrics
    unique_eliminated_A = len(set(eliminated_A))
    unique_eliminated_B = len(set(eliminated_B))

    metrics = {
        "ODP": odp,
        "Stability_A": stability_A,
        "Stability_B": stability_B,
        "Unique_Contestants_Eliminated_A": unique_eliminated_A,
        "Unique_Contestants_Eliminated_B": unique_eliminated_B,
        "Total_Simulations": len(simulation_results),
    }

    return metrics


def run_full_simulation(
    judges_file: str, fan_votes_file: str, n_simulations: int = 10000
) -> pd.DataFrame:
    """
    Run complete Monte Carlo simulation across all weeks and seasons.

    Args:
        judges_file: Path to judges_scores.csv
        fan_votes_file: Path to fan_vote_posterior_samples.csv
        n_simulations: Number of simulation runs

    Returns:
        DataFrame with results for each week
    """
    print(f"Starting Monte Carlo simulation with {n_simulations} runs...")

    # Load and merge data
    judges_df, fan_votes_df = load_data(judges_file, fan_votes_file)
    merged_df = merge_data_for_simulation(judges_df, fan_votes_df)

    # Get all unique season-week combinations
    season_weeks = (
        merged_df[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    )

    all_results = []

    print(f"Processing {len(season_weeks)} season-week combinations...")

    for idx, (_, row) in enumerate(season_weeks.iterrows()):
        season, week = row["season"], row["week"]

        print(
            f"Processing Season {season}, Week {week} ({idx + 1}/{len(season_weeks)})"
        )

        # Get data for this week
        week_data = merged_df[
            (merged_df["season"] == season) & (merged_df["week"] == week)
        ].copy()

        if len(week_data) == 0:
            print(f"  No data found for Season {season}, Week {week}")
            continue

        # Run simulation for this week
        simulation_results = run_simulation_for_week(week_data, n_simulations)

        if not simulation_results:
            print(f"  No simulation results for Season {season}, Week {week}")
            continue

        # Calculate metrics
        metrics = calculate_metrics(simulation_results)

        # Store results
        result_row = {
            "season": season,
            "week": week,
            "n_contestants": week_data["contestant_id"].nunique(),
            "n_simulations": len(simulation_results),
            "ODP": metrics["ODP"],
            "Stability_A_contestant": metrics["Stability_A"]["contestant"],
            "Stability_A_prob": metrics["Stability_A"]["prob"],
            "Stability_B_contestant": metrics["Stability_B"]["contestant"],
            "Stability_B_prob": metrics["Stability_B"]["prob"],
            "Unique_Eliminated_A": metrics["Unique_Contestants_Eliminated_A"],
            "Unique_Eliminated_B": metrics["Unique_Contestants_Eliminated_B"],
        }

        all_results.append(result_row)

        # Print summary for this week
        print(f"  ODP: {metrics['ODP']:.3f}")
        print(
            f"  Rule A most eliminated: {metrics['Stability_A']['contestant']} ({metrics['Stability_A']['prob']:.1%})"
        )
        print(
            f"  Rule B most eliminated: {metrics['Stability_B']['contestant']} ({metrics['Stability_B']['prob']:.1%})"
        )

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    print(f"\nSimulation completed! Processed {len(results_df)} weeks.")

    return results_df


def save_detailed_results(results_df: pd.DataFrame, output_dir: str = "res") -> None:
    """
    Save simulation results and generate summary statistics.

    Args:
        results_df: DataFrame with simulation results
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save main results
    results_file = output_path / "monte_carlo_simulation_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")

    # Generate and save summary statistics
    summary_stats = {
        "total_weeks": len(results_df),
        "mean_ODP": results_df["ODP"].mean(),
        "std_ODP": results_df["ODP"].std(),
        "min_ODP": results_df["ODP"].min(),
        "max_ODP": results_df["ODP"].max(),
        "mean_stability_A": results_df["Stability_A_prob"].mean(),
        "mean_stability_B": results_df["Stability_B_prob"].mean(),
        "high_discrepancy_weeks": (results_df["ODP"] > 0.5).sum(),
        "high_stability_weeks_A": (results_df["Stability_A_prob"] > 0.8).sum(),
        "high_stability_weeks_B": (results_df["Stability_B_prob"] > 0.8).sum(),
    }

    # Save summary
    summary_file = output_path / "simulation_summary.csv"
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total weeks analyzed: {summary_stats['total_weeks']}")
    print(f"Mean Outcome Discrepancy Probability: {summary_stats['mean_ODP']:.3f}")
    print(f"Std Dev of ODP: {summary_stats['std_ODP']:.3f}")
    print(f"ODP Range: {summary_stats['min_ODP']:.3f} - {summary_stats['max_ODP']:.3f}")
    print(
        f"Weeks with high discrepancy (ODP > 0.5): {summary_stats['high_discrepancy_weeks']}"
    )
    print(f"Mean stability Rule A: {summary_stats['mean_stability_A']:.3f}")
    print(f"Mean stability Rule B: {summary_stats['mean_stability_B']:.3f}")
    print(
        f"High stability weeks Rule A (>80%): {summary_stats['high_stability_weeks_A']}"
    )
    print(
        f"High stability weeks Rule B (>80%): {summary_stats['high_stability_weeks_B']}"
    )
    print("=" * 60)


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("MONTE CARLO SIMULATION - DANCING WITH THE STARS SCORING RULES")
    print("=" * 80)

    # Configuration
    N_SIMULATIONS = 10000

    # File paths (using the prepared data files)
    judges_file = "res/judges_scores.csv"
    fan_votes_file = "res/fan_vote_posterior_samples.csv"

    # Check if input files exist
    if not Path(judges_file).exists():
        print(f"Error: Judges scores file not found: {judges_file}")
        print("Please ensure the file exists and update the path in the script.")
        return

    if not Path(fan_votes_file).exists():
        print(f"Error: Fan votes file not found: {fan_votes_file}")
        print("Please ensure the file exists and update the path in the script.")
        return

    try:
        # Run simulation
        results_df = run_full_simulation(judges_file, fan_votes_file, N_SIMULATIONS)

        if len(results_df) == 0:
            print("No simulation results generated. Please check input data.")
            return

        # Save results
        save_detailed_results(results_df)

        print(f"\nSimulation completed successfully!")
        print(f"Check the 'res' directory for detailed results and summary statistics.")

    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
