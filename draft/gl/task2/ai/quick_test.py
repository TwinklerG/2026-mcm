#!/usr/bin/env python3
"""
Quick Test of Task 2 Core Functionality

This script provides a quick test of the Task 2 analysis framework
with a small subset of data to verify everything is working.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from task2_complete import Task2Analyzer


def quick_test():
    """Run a quick test with a small subset of data."""
    print("="*60)
    print("QUICK TASK 2 FUNCTIONALITY TEST")
    print("="*60)
    
    try:
        # Initialize analyzer
        analyzer = Task2Analyzer(data_dir="res", output_dir="quick_test_results")
        print("✓ Analyzer initialized")
        
        # Load data
        analyzer.load_data()
        print("✓ Data loaded successfully")
        
        # Check data
        if analyzer.judges_df is not None:
            print(f"  - Judges data: {len(analyzer.judges_df)} records")
        if analyzer.fan_votes_df is not None:
            print(f"  - Fan votes data: {len(analyzer.fan_votes_df)} records")
        
        # Test with a small subset
        print("\nTesting with Season 1, Week 1 data...")
        
        # Get small subset of data
        week1_data = analyzer.judges_df[
            (analyzer.judges_df['season'] == 1) & 
            (analyzer.judges_df['week'] == 1)
        ].copy()
        
        if len(week1_data) > 0:
            print(f"  - Found {len(week1_data)} contestants for Season 1, Week 1")
            
            # Test aggregation score calculation
            sample_data = week1_data.merge(
                analyzer.fan_votes_df[
                    (analyzer.fan_votes_df['season'] == 1) & 
                    (analyzer.fan_votes_df['week'] == 1) &
                    (analyzer.fan_votes_df['simulation_run'] == 1)
                ],
                on=['season', 'week', 'contestant_id']
            )
            
            if len(sample_data) > 0:
                print(f"  - Merged data: {len(sample_data)} records")
                
                # Test score calculation
                rank_scores, percentage_scores = analyzer.calculate_aggregation_scores(sample_data, 1)
                
                if len(rank_scores) > 0 and len(percentage_scores) > 0:
                    print("✓ Aggregation score calculation works")
                    print(f"  - Rank method eliminated: {rank_scores.loc[rank_scores['total_score'].idxmax(), 'contestant_id']}")
                    print(f"  - Percentage method eliminated: {percentage_scores.loc[percentage_scores['total_score'].idxmin(), 'contestant_id']}")
                else:
                    print("✗ Aggregation score calculation failed")
                    return False
            else:
                print("✗ Failed to merge data")
                return False
        else:
            print("✗ No data found for Season 1, Week 1")
            return False
        
        print("\n✓ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    success = quick_test()
    
    if success:
        print("\n🎉 Task 2 framework is working correctly!")
        print("\nNext steps:")
        print("1. Run full analysis: python run_task2_analysis.py")
        print("2. Or run individual components for detailed testing")
    else:
        print("\n🔧 Please check the errors above before running full analysis")
    
    return success


if __name__ == "__main__":
    main()