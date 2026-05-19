#!/usr/bin/env python3
"""
Task 2 Analysis Runner

This script demonstrates how to run the complete Task 2 analysis pipeline.
It includes data preparation, analysis execution, and results interpretation.

Usage:
    python run_task2_analysis.py

Author: MCM Team
Date: February 1, 2026
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from task2_complete import Task2Analyzer
from monte_carlo_prepare import prepare_judges_scores, prepare_fan_vote_posterior_samples
import pandas as pd


def check_data_availability():
    """Check if required data files are available."""
    print("Checking data availability...")
    
    data_dir = Path("res")
    required_files = [
        "judges_scores.csv",
        "fan_vote_posterior_samples.csv",
        "ranks.csv"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = data_dir / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            print(f"✓ Found: {file}")
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        print("Please run monte_carlo_prepare.py first to generate the required data files.")
        return False
    
    print("✓ All required data files are available!")
    return True


def run_data_preparation():
    """Run data preparation if needed."""
    print("\nRunning data preparation...")
    
    try:
        # Prepare judges scores
        judges_df = prepare_judges_scores(
            scores_file="../processed/scores_long.csv",
            output_file="res/judges_scores.csv"
        )
        
        # Prepare fan vote posterior samples
        fan_votes_df = prepare_fan_vote_posterior_samples(
            ranks_file="res/ranks.csv",
            output_file="res/fan_vote_posterior_samples.csv",
            n_samples=1000,
            noise_level=0.05
        )
        
        print("✓ Data preparation completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        return False


def run_task2_analysis():
    """Run the complete Task 2 analysis."""
    print("\n" + "="*80)
    print("STARTING TASK 2 ANALYSIS")
    print("="*80)
    
    try:
        # Initialize analyzer
        analyzer = Task2Analyzer(
            data_dir="res",
            output_dir="task2_results"
        )
        
        # Run complete analysis
        analyzer.run_complete_analysis(n_samples=1000)
        
        print("\n✓ Task 2 analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Task 2 analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def display_results_summary():
    """Display a summary of the analysis results."""
    print("\n" + "="*80)
    print("ANALYSIS RESULTS SUMMARY")
    print("="*80)
    
    results_dir = Path("task2_results")
    
    if not results_dir.exists():
        print("No results directory found. Analysis may not have completed successfully.")
        return
    
    # Check for key result files
    result_files = {
        "simulation_results.csv": "Raw simulation results",
        "stability_analysis.csv": "Stability analysis results",
        "discrepancy_analysis.csv": "Method discrepancy analysis",
        "recommendations.json": "Final recommendations",
        "analysis_summary.md": "Analysis summary report",
        "task2_analysis_plots.png": "Visualization plots"
    }
    
    print("\nGenerated result files:")
    for file, description in result_files.items():
        file_path = results_dir / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✓ {file} ({size:,} bytes) - {description}")
        else:
            print(f"✗ {file} - {description}")
    
    # Display key findings if available
    recommendations_file = results_dir / "recommendations.json"
    if recommendations_file.exists():
        print("\n" + "-"*60)
        print("KEY RECOMMENDATIONS:")
        print("-"*60)
        
        import json
        with open(recommendations_file, 'r') as f:
            recommendations = json.load(f)
        
        print(f"Aggregation Method: {recommendations['aggregation_method']['recommended']}")
        print(f"Confidence: {recommendations['aggregation_method']['confidence']}")
        print(f"Judge-Save Rule: {recommendations['judges_save_rule']['recommended']}")
        
        evidence = recommendations['quantitative_evidence']
        print(f"\nStability Comparison:")
        print(f"  Rank-Based: {evidence['stability_comparison']['rank_based']:.3f}")
        print(f"  Percentage-Based: {evidence['stability_comparison']['percentage_based']:.3f}")
        print(f"  Method Discrepancy: {evidence['method_discrepancy']:.3f}")


def main():
    """Main execution function."""
    print("="*80)
    print("TASK 2 ANALYSIS PIPELINE")
    print("="*80)
    print("This script runs the complete Task 2 analysis pipeline:")
    print("1. Data preparation and validation")
    print("2. Posterior-predictive elimination simulation")
    print("3. Stability and sensitivity metrics calculation")
    print("4. Controversial contestants analysis")
    print("5. Recommendation framework")
    print("6. Results visualization and reporting")
    print("="*80)
    
    # Step 1: Check data availability
    if not check_data_availability():
        print("\nAttempting to run data preparation...")
        if not run_data_preparation():
            print("\nPlease ensure you have the required input data files.")
            return False
    
    # Step 2: Run Task 2 analysis
    if not run_task2_analysis():
        return False
    
    # Step 3: Display results summary
    display_results_summary()
    
    print("\n" + "="*80)
    print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the generated results in 'task2_results/' directory")
    print("2. Check the analysis summary in 'task2_results/analysis_summary.md'")
    print("3. Examine the visualization plots in 'task2_results/task2_analysis_plots.png'")
    print("4. Read the detailed recommendations in 'task2_results/recommendations.json'")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)