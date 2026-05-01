"""
main.py
-------
Entry point for the program synthesis comparison study.

This script runs the full evaluation pipeline:
1. Runs all four synthesizers on all 20 benchmarks
2. Saves raw results to results/raw_results.csv
3. Generates comparison tables and charts

Usage:
    python main.py                    # Run full evaluation
    python main.py --skip-llms        # Only run enumerative (faster, for testing)
    python main.py --synthesizers Enumerative GPT-4o  # Run specific synthesizers
    python main.py --analyze-only     # Only generate charts from existing results
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API keys are present
def check_api_keys(skip_llms: bool = False) -> bool:
    """Check that required API keys are set."""
    if skip_llms:
        return True
    
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.getenv("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    
    if missing:
        print(f"Error: Missing API keys: {', '.join(missing)}")
        print("Please add them to your .env file.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Program Synthesis Comparison Study - CSE 590"
    )
    parser.add_argument(
        "--skip-llms",
        action="store_true",
        help="Only run enumerative synthesizer (no API calls, faster)"
    )
    parser.add_argument(
        "--synthesizers",
        nargs="+",
        choices=["Enumerative", "GPT-4o", "Claude", "Gemini"],
        help="Which synthesizers to run (default: all four)"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip synthesis, only generate analysis from existing results"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Program Synthesis Comparison Study")
    print("CSE 590 - University of Michigan")
    print("=" * 60)

    if not args.analyze_only:
        # Check API keys before running
        if not check_api_keys(skip_llms=args.skip_llms):
            sys.exit(1)

        # Run evaluation
        from evaluation.run_eval import run_all_evaluations, save_results, print_summary

        results = run_all_evaluations(
            skip_llms=args.skip_llms,
            synthesizers_to_run=args.synthesizers
        )

        save_results(results)
        print_summary(results)

    # Generate analysis
    from evaluation.results import (
        load_results,
        generate_summary_table,
        generate_candidates_table,
        generate_time_table,
        generate_detailed_breakdown,
        generate_charts
    )

    results_path = "results/raw_results.csv"
    if not os.path.exists(results_path):
        print(f"No results found at {results_path}")
        if args.analyze_only:
            print("Run without --analyze-only first to generate results.")
        sys.exit(1)

    print("\nGenerating analysis...")
    results = load_results(results_path)

    generate_summary_table(results)
    generate_candidates_table(results)
    generate_time_table(results)
    generate_detailed_breakdown(results)
    generate_charts(results)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("Results saved to results/ folder")
    print("=" * 60)


if __name__ == "__main__":
    main()


def run_extended_analysis():
    """Run error analysis and ablation study."""
    print("\n" + "=" * 60)
    print("Running Extended Analysis...")
    print("=" * 60)
    
    from evaluation.error_analysis import (
        analyze_enumerative_failures,
        generate_failure_report
    )
    from evaluation.ablation_study import (
        run_difficulty_scaling_analysis,
        save_ablation_results
    )
    
    analyze_enumerative_failures()
    generate_failure_report()
    run_difficulty_scaling_analysis()
    save_ablation_results()
    
    print("\nExtended analysis complete.")
