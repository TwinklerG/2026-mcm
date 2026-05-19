"""
Recommendation Framework for Elimination Prediction Methods

This module provides an actionable recommendation framework based on quantitative metrics:
- DiffProb: Method discrepancy probability
- Stab: Elimination stability
- JudgeImpact: Judge save impact probability

The framework generates formal recommendations for:
1. Whether to adopt percentage-based or rank-based aggregation
2. Whether to retain the Judges' Save rule
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class RecommendationFramework:
    """Framework for generating actionable recommendations based on quantitative metrics."""

    def __init__(self, output_dir: str = "res"):
        """
        Initialize the recommendation framework.

        Args:
            output_dir: Directory containing analysis results
        """
        self.output_dir = Path(output_dir)
        self.thresholds = {
            "diffprob_low": 0.2,  # DiffProb < 0.2: methods are consistent
            "diffprob_high": 0.5,  # DiffProb > 0.5: methods are inconsistent
            "stability_high": 0.7,  # Stability > 0.7: highly stable
            "stability_medium": 0.5,  # Stability 0.5-0.7: moderately stable
            "judge_impact_high": 0.3,  # JudgeImpact > 0.3: significant impact
            "judge_impact_medium": 0.15,  # JudgeImpact 0.15-0.3: moderate impact
        }

    def load_analysis_results(self) -> Dict[str, pl.DataFrame]:
        """
        Load all analysis results from previous metrics.

        Returns:
            Dictionary containing all analysis DataFrames
        """
        results = {}

        # Load stability analysis
        stability_file = self.output_dir / "stability_summary.csv"
        if stability_file.exists():
            results["stability_summary"] = pl.read_csv(stability_file)

        # Load method discrepancy analysis
        discrepancy_file = self.output_dir / "discrepancy_statistics.csv"
        if discrepancy_file.exists():
            results["discrepancy_statistics"] = pl.read_csv(discrepancy_file)

        # Load judge impact analysis
        judge_impact_file = self.output_dir / "method_judge_impact_comparison.csv"
        if judge_impact_file.exists():
            results["judge_impact_comparison"] = pl.read_csv(judge_impact_file)

        # Load detailed results
        stability_scores_file = self.output_dir / "stability_scores.csv"
        if stability_scores_file.exists():
            results["stability_scores"] = pl.read_csv(stability_scores_file)

        discrepancy_file = self.output_dir / "method_discrepancy.csv"
        if discrepancy_file.exists():
            results["method_discrepancy"] = pl.read_csv(discrepancy_file)

        judge_impact_file = self.output_dir / "judge_impact_analysis.csv"
        if judge_impact_file.exists():
            results["judge_impact_analysis"] = pl.read_csv(judge_impact_file)

        return results

    def analyze_method_consistency(
        self, discrepancy_stats: pl.DataFrame
    ) -> Dict[str, float]:
        """
        Analyze consistency between Rank and Percentage methods.

        Args:
            discrepancy_stats: DataFrame with discrepancy statistics

        Returns:
            Dictionary with consistency metrics
        """
        # Focus on rank_rank vs percent_rank comparison
        rank_vs_percent = discrepancy_stats.filter(
            pl.col("method_pair") == "rank_rank_vs_percent_rank"
        )

        if len(rank_vs_percent) == 0:
            return {"consistency_score": 0.5, "consistency_level": "unknown"}

        mean_discrepancy = rank_vs_percent["mean_discrepancy"][0]
        std_discrepancy = rank_vs_percent["std_discrepancy"][0]
        max_discrepancy = rank_vs_percent["max_discrepancy"][0]

        # Calculate consistency score (1 - mean_discrepancy)
        consistency_score = 1 - mean_discrepancy

        # Determine consistency level
        if mean_discrepancy < self.thresholds["diffprob_low"]:
            consistency_level = "highly_consistent"
        elif mean_discrepancy < self.thresholds["diffprob_high"]:
            consistency_level = "moderately_consistent"
        else:
            consistency_level = "highly_inconsistent"

        return {
            "consistency_score": consistency_score,
            "consistency_level": consistency_level,
            "mean_discrepancy": mean_discrepancy,
            "std_discrepancy": std_discrepancy,
            "max_discrepancy": max_discrepancy,
        }

    def analyze_method_stability(
        self, stability_summary: pl.DataFrame
    ) -> Dict[str, Dict]:
        """
        Analyze stability of different methods.

        Args:
            stability_summary: DataFrame with stability summary

        Returns:
            Dictionary with stability analysis for each method
        """
        stability_analysis = {}

        for row in stability_summary.iter_rows(named=True):
            method = row["method"]
            mean_stability = row["mean_stability"]
            std_stability = row["std_stability"]

            # Determine stability level
            if mean_stability >= self.thresholds["stability_high"]:
                stability_level = "highly_stable"
            elif mean_stability >= self.thresholds["stability_medium"]:
                stability_level = "moderately_stable"
            else:
                stability_level = "low_stability"

            stability_analysis[method] = {
                "mean_stability": mean_stability,
                "std_stability": std_stability,
                "stability_level": stability_level,
                "stability_score": mean_stability,  # Use mean as the score
            }

        return stability_analysis

    def analyze_judge_impact(
        self, judge_impact_comparison: pl.DataFrame
    ) -> Dict[str, float]:
        """
        Analyze the impact of Judges' Save rule.

        Args:
            judge_impact_comparison: DataFrame with judge impact comparison

        Returns:
            Dictionary with judge impact metrics
        """
        if len(judge_impact_comparison) == 0:
            return {"judge_impact_score": 0.0, "impact_level": "no_data"}

        # Calculate overall judge impact
        mean_impact = judge_impact_comparison["mean_impact_prob"].mean()
        max_impact = judge_impact_comparison["max_impact_prob"].max()
        std_impact = judge_impact_comparison["std_impact_prob"].mean()

        # Determine impact level
        if mean_impact >= self.thresholds["judge_impact_high"]:
            impact_level = "high_impact"
        elif mean_impact >= self.thresholds["judge_impact_medium"]:
            impact_level = "moderate_impact"
        else:
            impact_level = "low_impact"

        return {
            "judge_impact_score": mean_impact,
            "impact_level": impact_level,
            "max_impact": max_impact,
            "std_impact": std_impact,
            "pct_weeks_with_impact": judge_impact_comparison[
                "pct_weeks_with_impact"
            ].mean(),
        }

    def generate_method_recommendation(
        self,
        consistency_analysis: Dict[str, float],
        stability_analysis: Dict[str, Dict],
    ) -> Dict[str, str]:
        """
        Generate recommendation for aggregation method.

        Args:
            consistency_analysis: Results from method consistency analysis
            stability_analysis: Results from stability analysis

        Returns:
            Dictionary with method recommendation and reasoning
        """
        consistency_level = consistency_analysis["consistency_level"]
        mean_discrepancy = consistency_analysis["mean_discrepancy"]

        # Extract stability scores for rank_rank and percent_rank
        rank_stability = stability_analysis.get("rank_rank", {}).get(
            "stability_score", 0
        )
        percent_stability = stability_analysis.get("percent_rank", {}).get(
            "stability_score", 0
        )

        recommendation = {
            "recommended_method": "",
            "confidence": "",
            "reasoning": [],
            "key_findings": [],
        }

        # Case 1: Methods are highly consistent
        if consistency_level == "highly_consistent":
            recommendation["recommended_method"] = "either"
            recommendation["confidence"] = "high"
            recommendation["reasoning"].append(
                f"Rank and Percentage methods show high consistency (mean discrepancy: {mean_discrepancy:.3f})"
            )
            recommendation["reasoning"].append(
                "Both methods provide similar interpretations of fan voting behavior"
            )
            recommendation["key_findings"].append(
                "Method choice is not sensitive - either approach can be adopted"
            )

        # Case 2: Methods are inconsistent - choose based on stability
        elif consistency_level in ["moderately_consistent", "highly_inconsistent"]:
            stability_diff = abs(rank_stability - percent_stability)

            if percent_stability > rank_stability:
                recommendation["recommended_method"] = "percentage_based"
                recommendation["confidence"] = (
                    "high" if stability_diff > 0.1 else "medium"
                )
                recommendation["reasoning"].append(
                    f"Percentage method shows higher stability ({percent_stability:.3f} vs {rank_stability:.3f})"
                )
                recommendation["reasoning"].append(
                    "Percentage-based aggregation is more robust to uncertainty in fan voting"
                )
                recommendation["key_findings"].append(
                    "Percentage method demonstrates better consistency across posterior samples"
                )

            elif rank_stability > percent_stability:
                recommendation["recommended_method"] = "rank_based"
                recommendation["confidence"] = (
                    "high" if stability_diff > 0.1 else "medium"
                )
                recommendation["reasoning"].append(
                    f"Rank method shows higher stability ({rank_stability:.3f} vs {percent_stability:.3f})"
                )
                recommendation["reasoning"].append(
                    "Rank-based aggregation provides more consistent elimination predictions"
                )
                recommendation["key_findings"].append(
                    "Rank method demonstrates better robustness to fan vote uncertainty"
                )

            else:
                recommendation["recommended_method"] = "either"
                recommendation["confidence"] = "low"
                recommendation["reasoning"].append(
                    "Both methods show similar stability levels"
                )
                recommendation["reasoning"].append(
                    f"Methods show moderate inconsistency (mean discrepancy: {mean_discrepancy:.3f})"
                )
                recommendation["key_findings"].append(
                    "Further analysis may be needed to distinguish between methods"
                )

        # Add consistency findings
        if mean_discrepancy > 0.5:
            recommendation["key_findings"].append(
                f"High method discrepancy ({mean_discrepancy:.3f}) suggests structural differences"
            )
        elif mean_discrepancy > 0.2:
            recommendation["key_findings"].append(
                f"Moderate method discrepancy ({mean_discrepancy:.3f}) indicates some interpretational differences"
            )

        return recommendation

    def generate_judge_save_recommendation(
        self, judge_impact_analysis: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Generate recommendation for Judges' Save rule.

        Args:
            judge_impact_analysis: Results from judge impact analysis

        Returns:
            Dictionary with judge save recommendation and reasoning
        """
        impact_level = judge_impact_analysis["impact_level"]
        mean_impact = judge_impact_analysis["judge_impact_score"]
        pct_weeks_with_impact = judge_impact_analysis.get("pct_weeks_with_impact", 0)

        recommendation = {
            "recommended_action": "",
            "confidence": "",
            "reasoning": [],
            "key_findings": [],
        }

        # Case 1: High impact
        if impact_level == "high_impact":
            recommendation["recommended_action"] = "retain"
            recommendation["confidence"] = "high"
            recommendation["reasoning"].append(
                f"Judges' Save significantly impacts outcomes (mean impact: {mean_impact:.3f})"
            )
            recommendation["reasoning"].append(
                f"Rule affects {pct_weeks_with_impact:.1f}% of weeks with high-stakes eliminations"
            )
            recommendation["key_findings"].append(
                "Judges' Save effectively prevents erroneous eliminations"
            )
            recommendation["key_findings"].append(
                "Rule provides valuable safety net for unexpected fan voting patterns"
            )

        # Case 2: Moderate impact
        elif impact_level == "moderate_impact":
            recommendation["recommended_action"] = "retain_with_conditions"
            recommendation["confidence"] = "medium"
            recommendation["reasoning"].append(
                f"Judges' Save has moderate impact (mean impact: {mean_impact:.3f})"
            )
            recommendation["reasoning"].append(
                "Rule occasionally prevents questionable eliminations"
            )
            recommendation["key_findings"].append(
                "Judges' Save provides occasional but meaningful intervention"
            )
            recommendation["key_findings"].append(
                "Consider refining rule application criteria"
            )

        # Case 3: Low impact
        else:
            recommendation["recommended_action"] = "consider_removal"
            recommendation["confidence"] = "medium"
            recommendation["reasoning"].append(
                f"Judges' Save has minimal impact (mean impact: {mean_impact:.3f})"
            )
            recommendation["reasoning"].append(
                "Rule rarely changes elimination outcomes"
            )
            recommendation["key_findings"].append(
                "Judges' Save may be unnecessary given current voting patterns"
            )
            recommendation["key_findings"].append(
                "Consider removing rule to simplify elimination process"
            )

        return recommendation

    def generate_comprehensive_recommendation(
        self,
        method_recommendation: Dict[str, str],
        judge_recommendation: Dict[str, str],
        consistency_analysis: Dict[str, float],
        stability_analysis: Dict[str, Dict],
        judge_impact_analysis: Dict[str, float],
    ) -> Dict[str, any]:
        """
        Generate comprehensive recommendation report.

        Args:
            method_recommendation: Method recommendation results
            judge_recommendation: Judge save recommendation results
            consistency_analysis: Consistency analysis results
            stability_analysis: Stability analysis results
            judge_impact_analysis: Judge impact analysis results

        Returns:
            Comprehensive recommendation report
        """
        report = {
            "executive_summary": {},
            "detailed_recommendations": {
                "aggregation_method": method_recommendation,
                "judges_save_rule": judge_recommendation,
            },
            "quantitative_evidence": {
                "method_consistency": consistency_analysis,
                "method_stability": stability_analysis,
                "judge_impact": judge_impact_analysis,
            },
            "implementation_guidance": {},
            "risk_assessment": {},
            "monitoring_recommendations": {},
        }

        # Executive Summary
        if method_recommendation["recommended_method"] == "either":
            method_summary = "Either aggregation method can be used - both provide consistent results"
        elif method_recommendation["recommended_method"] == "percentage_based":
            method_summary = (
                "Recommend percentage-based aggregation for better stability"
            )
        else:
            method_summary = "Recommend rank-based aggregation for better stability"

        if judge_recommendation["recommended_action"] == "retain":
            judge_summary = "Strongly recommend retaining Judges' Save rule"
        elif judge_recommendation["recommended_action"] == "retain_with_conditions":
            judge_summary = (
                "Recommend retaining Judges' Save with potential refinements"
            )
        else:
            judge_summary = "Consider removing Judges' Save rule due to minimal impact"

        report["executive_summary"] = {
            "primary_recommendation": f"{method_summary}. {judge_summary}",
            "confidence_level": method_recommendation["confidence"],
            "key_metrics": {
                "method_consistency": f"{consistency_analysis['consistency_score']:.3f}",
                "best_stability_method": max(
                    stability_analysis.keys(),
                    key=lambda x: stability_analysis[x]["stability_score"],
                ),
                "judge_impact_level": judge_impact_analysis["impact_level"],
            },
        }

        # Implementation Guidance
        if method_recommendation["recommended_method"] == "percentage_based":
            report["implementation_guidance"]["aggregation_method"] = {
                "recommended": "percentage_based",
                "implementation_steps": [
                    "Use percentage-based aggregation for elimination predictions",
                    "Monitor stability metrics across seasons",
                    "Compare results with rank-based method for validation",
                ],
                "expected_benefits": [
                    "Higher robustness to fan voting uncertainty",
                    "More consistent elimination predictions",
                    "Better handling of extreme voting patterns",
                ],
            }
        elif method_recommendation["recommended_method"] == "rank_based":
            report["implementation_guidance"]["aggregation_method"] = {
                "recommended": "rank_based",
                "implementation_steps": [
                    "Use rank-based aggregation for elimination predictions",
                    "Monitor stability metrics across seasons",
                    "Compare results with percentage-based method for validation",
                ],
                "expected_benefits": [
                    "Higher robustness to fan voting uncertainty",
                    "More consistent elimination predictions",
                    "Better handling of extreme voting patterns",
                ],
            }
        else:
            report["implementation_guidance"]["aggregation_method"] = {
                "recommended": "either_method",
                "implementation_steps": [
                    "Either aggregation method can be used",
                    "Choose based on implementation preferences",
                    "Monitor consistency metrics to ensure continued alignment",
                ],
                "expected_benefits": [
                    "Both methods provide consistent results",
                    "Choice is not sensitive to method selection",
                    "Either approach will yield similar outcomes",
                ],
            }

        # Judge Save Implementation
        if judge_recommendation["recommended_action"] == "retain":
            report["implementation_guidance"]["judges_save_rule"] = {
                "recommended": "retain",
                "implementation_steps": [
                    "Continue applying Judges' Save rule",
                    "Monitor impact metrics each season",
                    "Document cases where rule prevents erroneous eliminations",
                ],
                "expected_benefits": [
                    "Prevents significant erroneous eliminations",
                    "Provides safety net for unexpected voting patterns",
                    "Maintains fairness in elimination process",
                ],
            }
        elif judge_recommendation["recommended_action"] == "retain_with_conditions":
            report["implementation_guidance"]["judges_save_rule"] = {
                "recommended": "retain_with_refinements",
                "implementation_steps": [
                    "Continue applying Judges' Save rule",
                    "Consider refining application criteria",
                    "Monitor impact metrics more closely",
                    "Evaluate rule effectiveness each season",
                ],
                "potential_improvements": [
                    "Define clearer criteria for rule application",
                    "Consider limiting rule to specific circumstances",
                    "Implement more nuanced judge evaluation process",
                ],
            }
        else:
            report["implementation_guidance"]["judges_save_rule"] = {
                "recommended": "consider_removal",
                "implementation_steps": [
                    "Evaluate rule removal in pilot season",
                    "Monitor elimination outcomes without rule",
                    "Compare results with historical data",
                    "Make final decision based on pilot results",
                ],
                "considerations": [
                    "Rule may be unnecessary given current voting patterns",
                    "Removal would simplify elimination process",
                    "Monitor for any increase in questionable eliminations",
                ],
            }

        # Risk Assessment
        report["risk_assessment"] = {
            "current_system_risks": [
                f"Method discrepancy of {consistency_analysis['mean_discrepancy']:.3f} may indicate inconsistent interpretations",
                f"Low stability scores suggest sensitivity to fan voting uncertainty",
                f"Judge impact of {judge_impact_analysis['judge_impact_score']:.3f} may affect elimination fairness",
            ],
            "recommended_changes_risks": [],
            "mitigation_strategies": [
                "Implement gradual changes with monitoring",
                "Maintain backup systems during transition",
                "Regular review of metrics and outcomes",
            ],
        }

        # Add specific risks based on recommendations
        if method_recommendation["recommended_method"] == "percentage_based":
            report["risk_assessment"]["recommended_changes_risks"].append(
                "Switching to percentage-based method may require system updates"
            )
        elif method_recommendation["recommended_method"] == "rank_based":
            report["risk_assessment"]["recommended_changes_risks"].append(
                "Switching to rank-based method may require system updates"
            )

        if judge_recommendation["recommended_action"] == "consider_removal":
            report["risk_assessment"]["recommended_changes_risks"].append(
                "Removing Judges' Save rule may lead to more controversial eliminations"
            )

        # Monitoring Recommendations
        report["monitoring_recommendations"] = {
            "key_metrics_to_track": [
                "Method consistency (DiffProb) - should remain low",
                "Method stability (Stab) - should remain high",
                "Judge impact (JudgeImpact) - should be monitored for changes",
            ],
            "monitoring_frequency": "Seasonal review with annual comprehensive analysis",
            "alert_thresholds": {
                "method_discrepancy_high": 0.5,
                "stability_low": 0.3,
                "judge_impact_significant": 0.3,
            },
            "review_process": [
                "Monthly metric collection during active seasons",
                "End-of-season comprehensive analysis",
                "Annual framework review and threshold adjustment",
            ],
        }

        return report

    def save_recommendation_report(
        self, report: Dict[str, any], output_file: str = "recommendation_report.json"
    ) -> None:
        """
        Save recommendation report to file.

        Args:
            report: Comprehensive recommendation report
            output_file: Output file path
        """
        output_path = self.output_dir / output_file

        # Convert to JSON-serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {
                    key: convert_to_serializable(value) for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_report = convert_to_serializable(report)

        with open(output_path, "w") as f:
            json.dump(serializable_report, f, indent=2)

        print(f"Recommendation report saved to: {output_path}")

    def print_executive_summary(self, report: Dict[str, any]) -> None:
        """
        Print executive summary of recommendations.

        Args:
            report: Comprehensive recommendation report
        """
        print("\n" + "=" * 80)
        print("EXECUTIVE SUMMARY - ELIMINATION PREDICTION METHOD RECOMMENDATIONS")
        print("=" * 80)

        summary = report["executive_summary"]
        print(f"\nPRIMARY RECOMMENDATION:")
        print(f"{summary['primary_recommendation']}")

        print(f"\nCONFIDENCE LEVEL: {summary['confidence_level']}")

        print(f"\nKEY METRICS:")
        for key, value in summary["key_metrics"].items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")

        print(f"\nDETAILED RECOMMENDATIONS:")

        # Method recommendation
        method_rec = report["detailed_recommendations"]["aggregation_method"]
        print(f"\n1. AGGREGATION METHOD:")
        print(f"   Recommended: {method_rec['recommended_method']}")
        print(f"   Confidence: {method_rec['confidence']}")
        print(f"   Key Reasoning:")
        for reason in method_rec["reasoning"]:
            print(f"     • {reason}")

        # Judge save recommendation
        judge_rec = report["detailed_recommendations"]["judges_save_rule"]
        print(f"\n2. JUDGES' SAVE RULE:")
        print(f"   Recommended Action: {judge_rec['recommended_action']}")
        print(f"   Confidence: {judge_rec['confidence']}")
        print(f"   Key Reasoning:")
        for reason in judge_rec["reasoning"]:
            print(f"     • {reason}")

        print("\n" + "=" * 80)

    def run_full_analysis(self) -> Dict[str, any]:
        """
        Run complete recommendation framework analysis.

        Returns:
            Comprehensive recommendation report
        """
        print("=== Recommendation Framework Analysis ===")

        # Load analysis results
        print("1. Loading analysis results...")
        results = self.load_analysis_results()

        if not results:
            print("No analysis results found. Please run metric.py first.")
            return {}

        # Perform analyses
        print("2. Analyzing method consistency...")
        consistency_analysis = self.analyze_method_consistency(
            results.get("discrepancy_statistics", pl.DataFrame())
        )

        print("3. Analyzing method stability...")
        stability_analysis = self.analyze_method_stability(
            results.get("stability_summary", pl.DataFrame())
        )

        print("4. Analyzing judge impact...")
        judge_impact_analysis = self.analyze_judge_impact(
            results.get("judge_impact_comparison", pl.DataFrame())
        )

        # Generate recommendations
        print("5. Generating method recommendation...")
        method_recommendation = self.generate_method_recommendation(
            consistency_analysis, stability_analysis
        )

        print("6. Generating judge save recommendation...")
        judge_recommendation = self.generate_judge_save_recommendation(
            judge_impact_analysis
        )

        print("7. Generating comprehensive report...")
        comprehensive_report = self.generate_comprehensive_recommendation(
            method_recommendation,
            judge_recommendation,
            consistency_analysis,
            stability_analysis,
            judge_impact_analysis,
        )

        # Save and display results
        print("8. Saving recommendation report...")
        self.save_recommendation_report(comprehensive_report)

        print("9. Displaying executive summary...")
        self.print_executive_summary(comprehensive_report)

        print(f"\n=== Analysis Complete ===")
        print(f"Full report saved to: {self.output_dir / 'recommendation_report.json'}")

        return comprehensive_report


def main():
    """Main function to run recommendation framework analysis."""
    from pathlib import Path

    # Setup paths
    current_dir = Path(__file__).parent
    output_dir = current_dir / "res"

    print("=== Recommendation Framework ===")
    print(f"Loading results from: {output_dir}")

    # Check if analysis results exist
    required_files = [
        "stability_summary.csv",
        "discrepancy_statistics.csv",
        "method_judge_impact_comparison.csv",
    ]

    missing_files = []
    for file in required_files:
        if not (output_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"Error: Missing required analysis files: {missing_files}")
        print("Please run metric.py first to generate the required analysis results.")
        return

    # Initialize framework and run analysis
    framework = RecommendationFramework(str(output_dir))
    report = framework.run_full_analysis()

    if report:
        print(f"\nRecommendation framework analysis completed successfully!")
        print(f"Check {output_dir} for detailed results and visualizations.")


if __name__ == "__main__":
    main()
