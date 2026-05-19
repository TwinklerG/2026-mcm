"""
Monte Carlo Simulation Data Preparation

This script prepares the input files needed for the Monte Carlo simulation:
1. judges_scores.csv - Contains actual judges' scores for each contestant in each week
2. fan_vote_posterior_samples.csv - Contains simulated fan vote shares from Bayesian model

Author: MCM Team
Date: January 31, 2026
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def prepare_judges_scores(scores_file: str, output_file: str) -> pd.DataFrame:
    """
    Prepare judges scores file from the long-format scores data.

    Args:
        scores_file: Path to scores_long.csv
        output_file: Path to save judges_scores.csv

    Returns:
        DataFrame with judges scores
    """
    print("Preparing judges scores data...")

    # Load scores data
    scores_df = pd.read_csv(scores_file)
    print(f"Loaded scores data: {len(scores_df)} records")

    # Calculate total judges score for each contestant per week
    judges_scores = (
        scores_df.groupby(["season", "week", "contestant_id"])["score"]
        .sum()
        .reset_index()
    )
    judges_scores = judges_scores.rename(columns={"score": "judges_score"})

    print(
        f"Calculated judges scores for {len(judges_scores)} contestant-week combinations"
    )
    print(f"Unique contestants: {judges_scores['contestant_id'].nunique()}")
    print(f"Seasons: {sorted(judges_scores['season'].unique())}")
    print(
        f"Weeks per season: {judges_scores.groupby('season')['week'].nunique().to_dict()}"
    )

    # Save to CSV
    judges_scores.to_csv(output_file, index=False)
    print(f"Judges scores saved to: {output_file}")

    return judges_scores


def prepare_fan_vote_posterior_samples(
    ranks_file: str, output_file: str, n_samples: int = 1000, noise_level: float = 0.05
) -> pd.DataFrame:
    """
    Prepare fan vote posterior samples from the ranks data.

    This function generates posterior samples by adding noise to the observed fan percentages
    to simulate uncertainty in fan voting patterns.

    Args:
        ranks_file: Path to ranks.csv
        output_file: Path to save fan_vote_posterior_samples.csv
        n_samples: Number of posterior samples to generate per contestant-week
        noise_level: Standard deviation of noise to add (as fraction of vote share)

    Returns:
        DataFrame with fan vote posterior samples
    """
    print("Preparing fan vote posterior samples...")

    # Load ranks data
    ranks_df = pd.read_csv(ranks_file)
    print(f"Loaded ranks data: {len(ranks_df)} records")

    # Extract fan percentages
    fan_data = ranks_df[["season", "week", "contestant_id", "fan_percent"]].copy()

    print(f"Generating {n_samples} posterior samples per contestant-week...")

    all_samples = []

    for _, row in fan_data.iterrows():
        season, week, contestant_id, fan_percent = (
            row["season"],
            row["week"],
            row["contestant_id"],
            row["fan_percent"],
        )

        # Generate posterior samples by adding noise
        np.random.seed(
            hash(f"{season}_{week}_{contestant_id}") % 2**32
        )  # Reproducible samples

        # Add noise proportional to the vote share
        noise = np.random.normal(0, noise_level * fan_percent, n_samples)
        samples = fan_percent + noise

        # Ensure samples are non-negative and sum to reasonable values
        samples = np.maximum(samples, 0.001)  # Minimum 0.1% vote share

        # Create DataFrame for this contestant-week
        for i, sample in enumerate(samples):
            all_samples.append(
                {
                    "season": season,
                    "week": week,
                    "contestant_id": contestant_id,
                    "simulation_run": i + 1,
                    "vote_share": sample,
                }
            )

    # Create DataFrame
    posterior_samples = pd.DataFrame(all_samples)

    print(f"Generated {len(posterior_samples)} posterior samples")
    print(f"Unique contestants: {posterior_samples['contestant_id'].nunique()}")
    print(f"Simulation runs per contestant-week: {n_samples}")

    # Save to CSV
    posterior_samples.to_csv(output_file, index=False)
    print(f"Fan vote posterior samples saved to: {output_file}")

    return posterior_samples


def validate_input_files(judges_file: str, fan_votes_file: str) -> bool:
    """
    Validate that the prepared input files have the correct structure.

    Args:
        judges_file: Path to judges_scores.csv
        fan_votes_file: Path to fan_vote_posterior_samples.csv

    Returns:
        True if files are valid, False otherwise
    """
    print("Validating input files...")

    # Check judges scores file
    if not Path(judges_file).exists():
        print(f"Error: Judges scores file not found: {judges_file}")
        return False

    judges_df = pd.read_csv(judges_file)
    required_judges_cols = ["season", "week", "contestant_id", "judges_score"]

    if not all(col in judges_df.columns for col in required_judges_cols):
        print(
            f"Error: Judges scores file missing required columns: {required_judges_cols}"
        )
        print(f"Found columns: {list(judges_df.columns)}")
        return False

    print(f"✓ Judges scores file valid: {len(judges_df)} records")

    # Check fan votes file
    if not Path(fan_votes_file).exists():
        print(f"Error: Fan votes file not found: {fan_votes_file}")
        return False

    fan_votes_df = pd.read_csv(fan_votes_file)
    required_fan_cols = [
        "season",
        "week",
        "contestant_id",
        "simulation_run",
        "vote_share",
    ]

    if not all(col in fan_votes_df.columns for col in required_fan_cols):
        print(f"Error: Fan votes file missing required columns: {required_fan_cols}")
        print(f"Found columns: {list(fan_votes_df.columns)}")
        return False

    print(f"✓ Fan votes file valid: {len(fan_votes_df)} records")

    # Check data consistency
    judges_season_weeks = set(zip(judges_df["season"], judges_df["week"]))
    fan_season_weeks = set(zip(fan_votes_df["season"], fan_votes_df["week"]))

    if judges_season_weeks != fan_season_weeks:
        missing_in_fan = judges_season_weeks - fan_season_weeks
        missing_in_judges = fan_season_weeks - judges_season_weeks

        if missing_in_fan:
            print(f"Warning: Missing fan data for season-weeks: {missing_in_fan}")
        if missing_in_judges:
            print(f"Warning: Missing judges data for season-weeks: {missing_in_judges}")

    # Check simulation runs consistency
    fan_runs_per_week = fan_votes_df.groupby(["season", "week", "contestant_id"])[
        "simulation_run"
    ].nunique()
    if len(fan_runs_per_week.unique()) > 1:
        print(
            f"Warning: Inconsistent number of simulation runs across contestant-weeks"
        )
        print(f"Run counts: {fan_runs_per_week.unique()}")
    else:
        print(
            f"✓ Consistent simulation runs: {fan_runs_per_week.iloc[0]} per contestant-week"
        )

    print("✓ All input files validated successfully!")
    return True


def generate_data_summary(
    judges_file: str, fan_votes_file: str, output_dir: str = "res"
) -> None:
    """
    Generate a summary of the prepared data.

    Args:
        judges_file: Path to judges_scores.csv
        fan_votes_file: Path to fan_vote_posterior_samples.csv
        output_dir: Directory to save summary
    """
    print("Generating data summary...")

    # Load data
    judges_df = pd.read_csv(judges_file)
    fan_votes_df = pd.read_csv(fan_votes_file)

    # Create summary with proper type conversion
    summary = {
        "judges_scores": {
            "total_records": int(len(judges_df)),
            "unique_contestants": int(judges_df["contestant_id"].nunique()),
            "seasons": [int(x) for x in sorted(judges_df["season"].unique())],
            "weeks_per_season": {
                int(k): int(v)
                for k, v in judges_df.groupby("season")["week"]
                .nunique()
                .to_dict()
                .items()
            },
            "contestants_per_week": {
                f"{k[0]}_{k[1]}": int(v)
                for k, v in judges_df.groupby(["season", "week"])["contestant_id"]
                .nunique()
                .to_dict()
                .items()
            },
            "judges_score_stats": {
                "mean": float(judges_df["judges_score"].mean()),
                "std": float(judges_df["judges_score"].std()),
                "min": float(judges_df["judges_score"].min()),
                "max": float(judges_df["judges_score"].max()),
            },
        },
        "fan_vote_samples": {
            "total_records": int(len(fan_votes_df)),
            "unique_contestants": int(fan_votes_df["contestant_id"].nunique()),
            "simulation_runs": int(fan_votes_df["simulation_run"].nunique()),
            "vote_share_stats": {
                "mean": float(fan_votes_df["vote_share"].mean()),
                "std": float(fan_votes_df["vote_share"].std()),
                "min": float(fan_votes_df["vote_share"].min()),
                "max": float(fan_votes_df["vote_share"].max()),
            },
        },
    }

    # Save summary as JSON
    import json

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    summary_file = output_path / "monte_carlo_data_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Data summary saved to: {summary_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Judges Scores:")
    print(f"  • Total records: {summary['judges_scores']['total_records']}")
    print(f"  • Unique contestants: {summary['judges_scores']['unique_contestants']}")
    print(f"  • Seasons: {summary['judges_scores']['seasons']}")
    print(
        f"  • Score range: {summary['judges_scores']['judges_score_stats']['min']:.1f} - {summary['judges_scores']['judges_score_stats']['max']:.1f}"
    )

    print(f"\nFan Vote Samples:")
    print(f"  • Total records: {summary['fan_vote_samples']['total_records']}")
    print(
        f"  • Unique contestants: {summary['fan_vote_samples']['unique_contestants']}"
    )
    print(
        f"  • Simulation runs per contestant-week: {summary['fan_vote_samples']['simulation_runs']}"
    )
    print(
        f"  • Vote share range: {summary['fan_vote_samples']['vote_share_stats']['min']:.3f} - {summary['fan_vote_samples']['vote_share_stats']['max']:.3f}"
    )
    print("=" * 60)


def main():
    """
    Main function to prepare all input files for Monte Carlo simulation.
    """
    print("=" * 80)
    print("MONTE CARLO SIMULATION DATA PREPARATION")
    print("=" * 80)

    # Configuration
    N_SAMPLES = 1000  # Number of posterior samples per contestant-week
    NOISE_LEVEL = 0.05  # Noise level for generating posterior samples

    # Input file paths
    scores_file = Path(__file__).parent / "../processed/scores_long.csv"
    ranks_file = Path(__file__).parent / "../res/ranks.csv"

    # Output file paths
    output_dir = "res"
    judges_file = f"{output_dir}/judges_scores.csv"
    fan_votes_file = f"{output_dir}/fan_vote_posterior_samples.csv"

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    try:
        # Step 1: Prepare judges scores
        print("\n1. Preparing judges scores...")
        judges_df = prepare_judges_scores(scores_file, judges_file)

        # Step 2: Prepare fan vote posterior samples
        print("\n2. Preparing fan vote posterior samples...")
        fan_votes_df = prepare_fan_vote_posterior_samples(
            ranks_file, fan_votes_file, N_SAMPLES, NOISE_LEVEL
        )

        # Step 3: Validate input files
        print("\n3. Validating input files...")
        if validate_input_files(judges_file, fan_votes_file):
            print("✓ All files validated successfully!")
        else:
            print("✗ File validation failed!")
            return

        # Step 4: Generate data summary
        print("\n4. Generating data summary...")
        generate_data_summary(judges_file, fan_votes_file, output_dir)

        print(f"\n" + "=" * 80)
        print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Input files ready for Monte Carlo simulation:")
        print(f"  • Judges scores: {judges_file}")
        print(f"  • Fan vote samples: {fan_votes_file}")
        print(f"  • Data summary: {output_dir}/monte_carlo_data_summary.json")
        print(f"\nNext step: Run monte_carlo_simulation.py")
        print("=" * 80)

    except Exception as e:
        print(f"Error during data preparation: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
