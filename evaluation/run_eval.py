"""
run_eval.py
-----------
Runs all four synthesizers on all 20 benchmarks and collects results.

For each synthesizer on each benchmark, records:
- success: did it find a correct program (True/False)
- held_out_pass: does the found program pass the held-out test
- candidates: number of candidates explored
- time: seconds to solution or failure
- program: string representation of found program (or None)

Results are saved to results/raw_results.csv for analysis.
"""

import os
import sys
import csv
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmarks import ALL_BENCHMARKS, Benchmark
from synthesizer import enumerative
from synthesizer import llm_gpt
from synthesizer import llm_claude
from synthesizer import llm_gemini
from synthesizer.verifier import verify_held_out


def run_single(
    synthesizer_name: str,
    synthesizer_module,
    benchmark: Benchmark
) -> Dict[str, Any]:
    """
    Run a single synthesizer on a single benchmark.
    
    Args:
        synthesizer_name: display name for the synthesizer
        synthesizer_module: the synthesizer module with a synthesize() function
        benchmark: the benchmark to run on
        
    Returns:
        Dictionary with results
    """
    print(f"  Running {synthesizer_name} on {benchmark.name}...", end=" ", flush=True)
    
    try:
        program, candidates, elapsed = synthesizer_module.synthesize(
            examples=benchmark.examples,
            variables=benchmark.variables
        )
        
        success = program is not None
        
        # Check held-out test if synthesis succeeded
        held_out_pass = False
        program_str = None
        if success:
            held_out_pass = verify_held_out(program, benchmark.held_out)
            program_str = str(program)
        
        result = {
            "synthesizer": synthesizer_name,
            "benchmark": benchmark.name,
            "difficulty": benchmark.difficulty,
            "success": success,
            "held_out_pass": held_out_pass,
            "candidates": candidates,
            "time_seconds": round(elapsed, 3),
            "program_found": program_str,
            "expected_program": benchmark.expected_program
        }
        
        status = "✓" if (success and held_out_pass) else ("partial" if success else "✗")
        print(f"{status} ({candidates} candidates, {elapsed:.2f}s)")
        
    except Exception as e:
        print(f"ERROR: {e}")
        result = {
            "synthesizer": synthesizer_name,
            "benchmark": benchmark.name,
            "difficulty": benchmark.difficulty,
            "success": False,
            "held_out_pass": False,
            "candidates": 0,
            "time_seconds": 0.0,
            "program_found": None,
            "expected_program": benchmark.expected_program
        }
    
    return result


def run_all_evaluations(
    skip_llms: bool = False,
    synthesizers_to_run: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Run all synthesizers on all benchmarks.
    
    Args:
        skip_llms: if True, only run enumerative synthesizer (for testing)
        synthesizers_to_run: list of synthesizer names to run
                             (default: all four)
    
    Returns:
        List of result dictionaries
    """
    # Define synthesizers
    all_synthesizers = [
        ("Enumerative", enumerative),
        ("GPT-4o", llm_gpt),
        ("Claude", llm_claude),
        ("Gemini", llm_gemini),
    ]
    
    if skip_llms:
        all_synthesizers = [("Enumerative", enumerative)]
    elif synthesizers_to_run:
        all_synthesizers = [
            (name, mod) for name, mod in all_synthesizers
            if name in synthesizers_to_run
        ]
    
    all_results = []
    
    for synth_name, synth_module in all_synthesizers:
        print(f"\n{'='*60}")
        print(f"Running synthesizer: {synth_name}")
        print(f"{'='*60}")
        
        for benchmark in ALL_BENCHMARKS:
            result = run_single(synth_name, synth_module, benchmark)
            all_results.append(result)
    
    return all_results


def save_results(results: List[Dict[str, Any]], output_path: str = None) -> str:
    """
    Save results to a CSV file.
    
    Args:
        results: list of result dictionaries
        output_path: path to save CSV (default: results/raw_results.csv)
        
    Returns:
        Path where results were saved
    """
    if output_path is None:
        os.makedirs("results", exist_ok=True)
        output_path = "results/raw_results.csv"
    
    if not results:
        print("No results to save.")
        return output_path
    
    fieldnames = list(results[0].keys())
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {output_path}")
    return output_path


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a quick summary table of results."""
    if not results:
        return
    
    synthesizers = list(dict.fromkeys(r["synthesizer"] for r in results))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Synthesizer':<15} {'Success':<10} {'Held-out':<10} {'Avg Candidates':<18} {'Avg Time'}")
    print("-" * 60)
    
    for synth in synthesizers:
        synth_results = [r for r in results if r["synthesizer"] == synth]
        
        successes = sum(1 for r in synth_results if r["success"])
        held_out_passes = sum(1 for r in synth_results if r["held_out_pass"])
        total = len(synth_results)
        
        successful_results = [r for r in synth_results if r["success"]]
        avg_candidates = (
            sum(r["candidates"] for r in successful_results) / len(successful_results)
            if successful_results else 0
        )
        avg_time = (
            sum(r["time_seconds"] for r in synth_results) / total
            if total > 0 else 0
        )
        
        print(
            f"{synth:<15} {successes}/{total:<7} {held_out_passes}/{total:<7} "
            f"{avg_candidates:<18.1f} {avg_time:.2f}s"
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run synthesis evaluation")
    parser.add_argument(
        "--skip-llms",
        action="store_true",
        help="Only run enumerative synthesizer (faster, for testing)"
    )
    parser.add_argument(
        "--synthesizers",
        nargs="+",
        choices=["Enumerative", "GPT-4o", "Claude", "Gemini"],
        help="Which synthesizers to run (default: all)"
    )
    args = parser.parse_args()
    
    print("Starting evaluation...")
    print(f"Running on {len(ALL_BENCHMARKS)} benchmarks")
    
    results = run_all_evaluations(
        skip_llms=args.skip_llms,
        synthesizers_to_run=args.synthesizers
    )
    
    save_results(results)
    print_summary(results)
