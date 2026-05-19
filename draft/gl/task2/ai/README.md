# Task 2: Simulation-Based Comparative Analysis

This directory contains the complete implementation of Task 2 for the MCM 2026 problem, comparing Rank-based and Percentage-based aggregation rules for "Dancing with the Stars" elimination process.

## 📁 Files Overview

### Core Implementation
- **`task2_complete.py`** - Complete Task 2 analysis framework
- **`monte_carlo_prepare.py`** - Data preparation script
- **`monte_carlo_simulation.py`** - Monte Carlo simulation engine

### Execution Scripts
- **`run_task2_analysis.py`** - Main execution script with full pipeline
- **`test_task2_components.py`** - Component testing script

### Results and Documentation
- **`TASK2_最终报告.md`** - Chinese final report
- **`TASK2_FINAL_REPORT.md`** - English final report
- **`MONTE_CARLO_RESULTS.md`** - Monte Carlo simulation results

## 🚀 Quick Start

### Option 1: Run Complete Analysis (Recommended)
```bash
python run_task2_analysis.py
```

This script will:
1. Check data availability
2. Run data preparation if needed
3. Execute the complete Task 2 analysis
4. Generate visualizations and reports
5. Display results summary

### Option 2: Test Components First
```bash
# Test individual components
python test_task2_components.py

# If all tests pass, run full analysis
python run_task2_analysis.py
```

### Option 3: Manual Execution
```bash
# Step 1: Prepare data
python monte_carlo_prepare.py

# Step 2: Run analysis
python task2_complete.py
```

## 📊 Input Data Requirements

The analysis requires the following data files in the `res/` directory:

1. **`judges_scores.csv`** - Judges' scores for each contestant
   - Columns: `season`, `week`, `contestant_id`, `judges_score`

2. **`fan_vote_posterior_samples.csv`** - Posterior samples of fan votes
   - Columns: `season`, `week`, `contestant_id`, `simulation_run`, `vote_share`

3. **`ranks.csv`** - Additional ranking data (optional)
   - Columns: `season`, `week`, `contestant_id`, `fan_percent`, `judge_percent`, etc.

## 📈 Output Files

The analysis generates the following output files in `task2_results/`:

### Core Analysis Results
- **`simulation_results.csv`** - Raw simulation results
- **`stability_analysis.csv`** - Elimination stability metrics
- **`discrepancy_analysis.csv`** - Method discrepancy probabilities
- **`judge_impact_analysis.csv`** - Judge-Save impact analysis

### Controversial Contestants Analysis
- **`controversial_jerry_rice_analysis.csv`**
- **`controversial_billy_ray_cyrus_analysis.csv`**
- **`controversial_bristol_palin_analysis.csv`**
- **`controversial_bobby_bones_analysis.csv`**

### Reports and Visualizations
- **`recommendations.json`** - Final recommendations in JSON format
- **`analysis_summary.md`** - Human-readable analysis summary
- **`task2_analysis_plots.png`** - Visualization plots

## 🔬 Analysis Components

### 1. Posterior-Predictive Elimination Simulation
- Simulates elimination outcomes using MCMC samples from Task 1
- Compares Rank-based vs Percentage-based aggregation rules
- Generates posterior distributions of elimination outcomes

### 2. Stability and Sensitivity Metrics
- **Elimination Stability**: `Stab_t^(method) = max_{i in C_t} P(e_t=i|method)`
- **Method Discrepancy Probability**: `DiffProb_t = 1/S * sum_{s=1}^S 1{e_t^(rank,(s)) ≠ e_t^(pct,(s))}`
- **Judge-Save Impact Probability**: Measures impact of Judge-Save rule

### 3. Controversial Contestants Analysis
- Detailed analysis of Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby Bones
- Compares elimination probabilities under both methods
- Identifies method-specific benefits and biases

### 4. Recommendation Framework
- Generates actionable recommendations based on quantitative metrics
- Compares aggregation methods and Judge-Save rule effectiveness
- Provides confidence levels and reasoning

## 🎯 Key Features

### Mathematical Rigor
- Implements exact mathematical formulations from the problem statement
- Uses proper statistical inference with posterior samples
- Handles tie-breaking according to official DWTS rules

### Computational Efficiency
- Optimized for large-scale simulation (1000+ samples per week)
- Memory-efficient data processing
- Parallel processing capabilities where applicable

### Comprehensive Analysis
- Multi-dimensional comparison of aggregation methods
- Statistical significance testing
- Robustness analysis under uncertainty

### Professional Visualization
- High-quality plots for publication
- Interactive analysis capabilities
- Clear presentation of results

## 📋 Mathematical Formulations

### Rank-Based Method (Seasons 1-2, 28-34)
```
R_(i,t)^(T,(s)) = R_(i,t)^J + R_(i,t)^(F,(s))
e_t^(rank,(s)) = argmax_(i in C_t) R_(i,t)^(T,(s))
```

### Percentage-Based Method (Seasons 3-27)
```
p_(i,t)^J = J_(i,t) / sum_(j in C_t) J_(j,t)
T_(i,t)^((s)) = p_(i,t)^J + f_(i,t)^((s))
e_t^(pct,(s)) = argmin_(i in C_t) T_(i,t)^((s))
```

### Key Metrics
- **Stability**: Measures consistency of elimination outcomes
- **Discrepancy**: Measures difference between methods
- **Judge Impact**: Measures effect of Judge-Save rule

## 🔧 Configuration Options

### Simulation Parameters
```python
# Number of posterior samples to use
n_samples = 1000

# Noise level for fan vote uncertainty
noise_level = 0.05

# Season when Judge-Save rule starts
save_season_start = 28
```

### Analysis Options
```python
# Controversial contestants to analyze
controversial_contestants = [
    "Jerry Rice", "Billy Ray Cyrus", 
    "Bristol Palin", "Bobby Bones"
]

# Output directory
output_dir = "task2_results"
```

## 📊 Expected Results

Based on the analysis framework, you should expect:

1. **Stability Comparison**: Percentage-based method typically shows higher stability
2. **Method Discrepancy**: Significant differences in 40%+ of weeks
3. **Judge Impact**: Measurable impact in seasons with Judge-Save rule
4. **Controversial Cases**: Method-specific benefits for different contestants

## 🐛 Troubleshooting

### Common Issues

1. **Missing Data Files**
   - Ensure all required CSV files are in the `res/` directory
   - Run `monte_carlo_prepare.py` to generate missing files

2. **Memory Issues**
   - Reduce `n_samples` parameter for large datasets
   - Use data chunking for very large fan vote files

3. **Visualization Errors**
   - Ensure matplotlib and seaborn are installed
   - Check display settings for headless environments

### Getting Help

1. **Run Component Tests**: `python test_task2_components.py`
2. **Check Data Format**: Verify CSV file structure matches requirements
3. **Review Error Messages**: Detailed error information is provided

## 📚 References

- Task 2 Problem Statement (MCM 2026)
- Official DWTS Rules and Scoring Methods
- Statistical Inference for Elimination Processes
- Posterior-Predictive Simulation Techniques

## 👥 Authors

MCM Team 2026
- Implementation: Complete Task 2 analysis framework
- Validation: Comprehensive testing suite
- Documentation: Detailed guides and reports

---

*For questions or issues, please refer to the component test results or check the generated analysis summary for detailed insights.*