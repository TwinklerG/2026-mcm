#!/usr/bin/env python3
"""
Task 2 Components Test Script

This script tests the individual components of the Task 2 analysis pipeline
to ensure everything is working correctly before running the full analysis.

Usage:
    python test_task2_components.py

Author: MCM Team
Date: February 1, 2026
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from task2_complete import Task2Analyzer


def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    
    try:
        # Create a temporary analyzer
        analyzer = Task2Analyzer(data_dir="res", output_dir="test_results")
        
        # Test data loading
        analyzer.load_data()
        
        # Check if data was loaded
        if analyzer.judges_df is not None:
            print(f"✓ Judges data loaded: {len(analyzer.judges_df)} records")
        else:
            print("✗ Judges data not loaded")
            return False
        
        if analyzer.fan_votes_df is not None:
            print(f"✓ Fan votes data loaded: {len(analyzer.fan_votes_df)} records")
        else:
            print("✗ Fan votes data not loaded")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False


def test_aggregation_score_calculation():
    """Test aggregation score calculation."""
    print("\nTesting aggregation score calculation...")
    
    try:
        # Create sample data
        sample_data = pd.DataFrame({
            'contestant_id': ['Contestant_A', 'Contestant_B', 'Contestant_C'],
            'judges_score': [25.0, 30.0, 20.0],
            'vote_share': [0.3, 0.4, 0.3],
            'simulation_run': [1, 1, 1]
        })
        
        # Create analyzer
        analyzer = Task2Analyzer()
        
        # Test score calculation
        rank_scores, percentage_scores = analyzer.calculate_aggregation_scores(sample_data, 1)
        
        # Check results
        if len(rank_scores) > 0 and len(percentage_scores) > 0:
            print("✓ Aggregation scores calculated successfully")
            print(f"  Rank method results: {len(rank_scores)} contestants")
            print(f"  Percentage method results: {len(percentage_scores)} contestants")
            
            # Check if scores are reasonable
            if rank_scores['total_score'].min() > 0 and percentage_scores['total_score'].min() > 0:
                print("✓ Score values are reasonable")
                return True
            else:
                print("✗ Score values are not reasonable")
                return False
        else:
            print("✗ Failed to calculate aggregation scores")
            return False
            
    except Exception as e:
        print(f"✗ Aggregation score calculation test failed: {e}")
        return False


def test_stability_calculation():
    """Test elimination stability calculation."""
    print("\nTesting elimination stability calculation...")
    
    try:
        # Create sample simulation results
        sample_results = pd.DataFrame({
            'season': [1, 1, 1, 1, 1] * 20,  # 100 samples for week 1
            'week': [1] * 100,
            'eliminated_rank': ['Contestant_A'] * 80 + ['Contestant_B'] * 20,
            'eliminated_percentage': ['Contestant_A'] * 95 + ['Contestant_C'] * 5
        })
        
        # Create analyzer
        analyzer = Task2Analyzer()
        
        # Test stability calculation
        stability_results = analyzer.calculate_elimination_stability(sample_results)
        
        if len(stability_results) > 0:
            print("✓ Stability calculation completed successfully")
            print(f"  Results for {len(stability_results)} weeks")
            
            # Check if stability values are reasonable
            row = stability_results.iloc[0]
            if 0 <= row['rank_stability'] <= 1 and 0 <= row['percentage_stability'] <= 1:
                print(f"✓ Stability values are reasonable: Rank={row['rank_stability']:.3f}, Percentage={row['percentage_stability']:.3f}")
                return True
            else:
                print("✗ Stability values are not in valid range [0,1]")
                return False
        else:
            print("✗ No stability results generated")
            return False
            
    except Exception as e:
        print(f"✗ Stability calculation test failed: {e}")
        return False


def test_discrepancy_calculation():
    """Test method discrepancy probability calculation."""
    print("\nTesting method discrepancy calculation...")
    
    try:
        # Create sample simulation results with known discrepancy
        sample_results = pd.DataFrame({
            'season': [1] * 100,
            'week': [1] * 100,
            'eliminated_rank': ['Contestant_A'] * 60 + ['Contestant_B'] * 40,
            'eliminated_percentage': ['Contestant_A'] * 40 + ['Contestant_C'] * 60
        })
        
        # Create analyzer
        analyzer = Task2Analyzer()
        
        # Test discrepancy calculation
        discrepancy_results = analyzer.calculate_method_discrepancy_probability(sample_results)
        
        if len(discrepancy_results) > 0:
            print("✓ Discrepancy calculation completed successfully")
            
            # Check if discrepancy value is reasonable
            row = discrepancy_results.iloc[0]
            if 0 <= row['discrepancy_prob'] <= 1:
                print(f"✓ Discrepancy value is reasonable: {row['discrepancy_prob']:.3f}")
                return True
            else:
                print("✗ Discrepancy value is not in valid range [0,1]")
                return False
        else:
            print("✗ No discrepancy results generated")
            return False
            
    except Exception as e:
        print(f"✗ Discrepancy calculation test failed: {e}")
        return False


def test_recommendation_generation():
    """Test recommendation generation."""
    print("\nTesting recommendation generation...")
    
    try:
        # Create sample analysis results
        stability_results = pd.DataFrame({
            'season': [1, 1, 2, 2],
            'week': [1, 2, 1, 2],
            'rank_stability': [0.8, 0.7, 0.9, 0.6],
            'percentage_stability': [0.9, 0.8, 0.95, 0.7]
        })
        
        discrepancy_results = pd.DataFrame({
            'season': [1, 1, 2, 2],
            'week': [1, 2, 1, 2],
            'discrepancy_prob': [0.3, 0.4, 0.2, 0.5]
        })
        
        judge_impact_results = pd.DataFrame({
            'season': [28, 28],
            'week': [1, 2],
            'judge_impact_prob': [0.4, 0.3]
        })
        
        # Create analyzer
        analyzer = Task2Analyzer()
        
        # Test recommendation generation
        recommendations = analyzer.generate_recommendations(
            stability_results, discrepancy_results, judge_impact_results
        )
        
        if recommendations and 'aggregation_method' in recommendations:
            print("✓ Recommendations generated successfully")
            print(f"  Recommended method: {recommendations['aggregation_method']['recommended']}")
            print(f"  Confidence: {recommendations['aggregation_method']['confidence']}")
            return True
        else:
            print("✗ Failed to generate recommendations")
            return False
            
    except Exception as e:
        print(f"✗ Recommendation generation test failed: {e}")
        return False


def test_controversial_contestants_analysis():
    """Test controversial contestants analysis."""
    print("\nTesting controversial contestants analysis...")
    
    try:
        # Create sample simulation results with controversial contestants
        sample_results = pd.DataFrame({
            'season': [1, 1, 1, 2, 2, 2],
            'week': [1, 1, 1, 1, 1, 1],
            'eliminated_rank': ['Jerry Rice', 'Contestant_B', 'Contestant_C', 'Jerry Rice', 'Contestant_D', 'Contestant_E'],
            'eliminated_percentage': ['Contestant_B', 'Jerry Rice', 'Contestant_C', 'Contestant_D', 'Jerry Rice', 'Contestant_E']
        })
        
        # Create analyzer
        analyzer = Task2Analyzer()
        
        # Test controversial contestants analysis
        controversial_analysis = analyzer.analyze_controversial_contestants(
            sample_results, ["Jerry Rice"]
        )
        
        if "Jerry Rice" in controversial_analysis:
            print("✓ Controversial contestants analysis completed successfully")
            analysis = controversial_analysis["Jerry Rice"]
            print(f"  Analysis for Jerry Rice: {len(analysis)} records")
            return True
        else:
            print("✗ No analysis generated for controversial contestants")
            return False
            
    except Exception as e:
        print(f"✗ Controversial contestants analysis test failed: {e}")
        return False


def run_all_tests():
    """Run all component tests."""
    print("="*80)
    print("TASK 2 COMPONENTS TEST SUITE")
    print("="*80)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Aggregation Score Calculation", test_aggregation_score_calculation),
        ("Stability Calculation", test_stability_calculation),
        ("Discrepancy Calculation", test_discrepancy_calculation),
        ("Recommendation Generation", test_recommendation_generation),
        ("Controversial Contestants Analysis", test_controversial_contestants_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-'*60}")
        print(f"Running test: {test_name}")
        print(f"{'-'*60}")
        
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "="*80)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*80)
    
    if passed == total:
        print("🎉 All tests passed! The Task 2 analysis pipeline is ready to run.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        return False


def main():
    """Main test execution function."""
    print("Task 2 Components Test Suite")
    print("This script tests individual components of the Task 2 analysis pipeline.")
    print("="*80)
    
    success = run_all_tests()
    
    if success:
        print("\n🚀 Ready to run full Task 2 analysis!")
        print("Execute: python run_task2_analysis.py")
    else:
        print("\n🔧 Please fix the failing tests before running the full analysis.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)