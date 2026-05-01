"""
results.py
----------
Generates comparison tables and charts from evaluation results.

Reads raw_results.csv and produces:
1. A formatted comparison table (also saved as results_table.csv)
2. Bar charts comparing synthesizers across difficulty levels
3. A detailed breakdown by benchmark

Run this after run_eval.py has completed.
"""

import os
import csv
import sys
from typing import List, Dict, Any
from collections import defaultdict

# Check if matplotlib is available
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping charts")


def load_results(csv_path: str = "results/raw_results.csv") -> List[Dict[str, Any]]:
    """Load results from CSV file."""
    results = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert types
            row["success"] = row["success"] == "True"
            row["held_out_pass"] = row["held_out_pass"] == "True"
            row["candidates"] = int(row["candidates"])
            row["time_seconds"] = float(row["time_seconds"])
            results.append(row)
    return results


def generate_summary_table(results: List[Dict[str, Any]]) -> None:
    """Generate and print a summary table."""
    synthesizers = list(dict.fromkeys(r["synthesizer"] for r in results))
    difficulties = ["simple_arithmetic", "hard_arithmetic", "simple_string", "hard_string"]
    
    print("\n" + "="*80)
    print("RESULTS TABLE: Success Rate by Synthesizer and Difficulty")
    print("="*80)
    
    header = f"{'Synthesizer':<15}"
    for diff in difficulties:
        short = diff.replace("_", " ").title()[:16]
        header += f" {short:<16}"
    header += f" {'Overall':<10}"
    print(header)
    print("-" * 80)
    
    table_data = []
    
    for synth in synthesizers:
        row_data = {"synthesizer": synth}
        row_str = f"{synth:<15}"
        
        total_success = 0
        total_count = 0
        
        for diff in difficulties:
            diff_results = [
                r for r in results
                if r["synthesizer"] == synth and r["difficulty"] == diff
            ]
            
            if diff_results:
                success = sum(1 for r in diff_results if r["held_out_pass"])
                count = len(diff_results)
                pct = (success / count) * 100
                row_str += f" {success}/{count} ({pct:.0f}%)      "
                row_data[diff] = f"{success}/{count}"
                total_success += success
                total_count += count
            else:
                row_str += f" N/A              "
                row_data[diff] = "N/A"
        
        overall_pct = (total_success / total_count * 100) if total_count > 0 else 0
        row_str += f" {total_success}/{total_count} ({overall_pct:.0f}%)"
        row_data["overall"] = f"{total_success}/{total_count}"
        
        print(row_str)
        table_data.append(row_data)
    
    print("="*80)
    
    # Save table to CSV
    os.makedirs("results", exist_ok=True)
    with open("results/results_table.csv", "w", newline="") as f:
        fieldnames = ["synthesizer"] + difficulties + ["overall"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_data)
    
    print("Summary table saved to results/results_table.csv")


def generate_candidates_table(results: List[Dict[str, Any]]) -> None:
    """Generate table showing average candidates explored."""
    synthesizers = list(dict.fromkeys(r["synthesizer"] for r in results))
    
    print("\n" + "="*80)
    print("CANDIDATES EXPLORED: Average per Synthesizer (successful runs only)")
    print("="*80)
    print(f"{'Synthesizer':<15} {'Simple Arith':<15} {'Hard Arith':<15} {'Simple Str':<15} {'Hard Str':<15} {'Overall'}")
    print("-" * 80)
    
    difficulties = ["simple_arithmetic", "hard_arithmetic", "simple_string", "hard_string"]
    
    for synth in synthesizers:
        row_str = f"{synth:<15}"
        
        total_candidates = 0
        total_count = 0
        
        for diff in difficulties:
            diff_results = [
                r for r in results
                if r["synthesizer"] == synth
                and r["difficulty"] == diff
                and r["success"]
            ]
            
            if diff_results:
                avg = sum(r["candidates"] for r in diff_results) / len(diff_results)
                row_str += f" {avg:<14.1f}"
                total_candidates += sum(r["candidates"] for r in diff_results)
                total_count += len(diff_results)
            else:
                row_str += f" {'N/A':<14}"
        
        overall_avg = (total_candidates / total_count) if total_count > 0 else 0
        row_str += f" {overall_avg:.1f}"
        print(row_str)
    
    print("="*80)


def generate_time_table(results: List[Dict[str, Any]]) -> None:
    """Generate table showing average time to solution."""
    synthesizers = list(dict.fromkeys(r["synthesizer"] for r in results))
    
    print("\n" + "="*80)
    print("TIME TO SOLUTION: Average seconds per Synthesizer (successful runs only)")
    print("="*80)
    print(f"{'Synthesizer':<15} {'Simple Arith':<15} {'Hard Arith':<15} {'Simple Str':<15} {'Hard Str':<15} {'Overall'}")
    print("-" * 80)
    
    difficulties = ["simple_arithmetic", "hard_arithmetic", "simple_string", "hard_string"]
    
    for synth in synthesizers:
        row_str = f"{synth:<15}"
        
        total_time = 0
        total_count = 0
        
        for diff in difficulties:
            diff_results = [
                r for r in results
                if r["synthesizer"] == synth
                and r["difficulty"] == diff
                and r["success"]
            ]
            
            if diff_results:
                avg = sum(r["time_seconds"] for r in diff_results) / len(diff_results)
                row_str += f" {avg:<14.3f}"
                total_time += sum(r["time_seconds"] for r in diff_results)
                total_count += len(diff_results)
            else:
                row_str += f" {'N/A':<14}"
        
        overall_avg = (total_time / total_count) if total_count > 0 else 0
        row_str += f" {overall_avg:.3f}s"
        print(row_str)
    
    print("="*80)


def generate_charts(results: List[Dict[str, Any]]) -> None:
    """Generate bar charts comparing synthesizers."""
    if not HAS_MATPLOTLIB:
        print("Skipping charts (matplotlib not available)")
        return
    
    os.makedirs("results", exist_ok=True)
    
    synthesizers = list(dict.fromkeys(r["synthesizer"] for r in results))
    difficulties = ["simple_arithmetic", "hard_arithmetic", "simple_string", "hard_string"]
    diff_labels = ["Simple\nArithmetic", "Hard\nArithmetic", "Simple\nString", "Hard\nString"]
    
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    
    # Chart 1: Success rate by difficulty
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Success rate chart
    ax1 = axes[0]
    x = range(len(difficulties))
    width = 0.2
    
    for i, (synth, color) in enumerate(zip(synthesizers, colors)):
        success_rates = []
        for diff in difficulties:
            diff_results = [
                r for r in results
                if r["synthesizer"] == synth and r["difficulty"] == diff
            ]
            if diff_results:
                rate = sum(1 for r in diff_results if r["held_out_pass"]) / len(diff_results) * 100
            else:
                rate = 0
            success_rates.append(rate)
        
        offset = (i - len(synthesizers)/2 + 0.5) * width
        bars = ax1.bar([xi + offset for xi in x], success_rates, width, 
                      label=synth, color=color, alpha=0.85)
    
    ax1.set_xlabel("Difficulty Level")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Success Rate by Synthesizer and Difficulty")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(diff_labels)
    ax1.set_ylim(0, 110)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Average candidates chart (log scale for enumerative)
    ax2 = axes[1]
    
    for i, (synth, color) in enumerate(zip(synthesizers, colors)):
        avg_candidates = []
        for diff in difficulties:
            diff_results = [
                r for r in results
                if r["synthesizer"] == synth
                and r["difficulty"] == diff
                and r["success"]
            ]
            if diff_results:
                avg = sum(r["candidates"] for r in diff_results) / len(diff_results)
            else:
                avg = 0
            avg_candidates.append(avg)
        
        offset = (i - len(synthesizers)/2 + 0.5) * width
        ax2.bar([xi + offset for xi in x], avg_candidates, width,
               label=synth, color=color, alpha=0.85)
    
    ax2.set_xlabel("Difficulty Level")
    ax2.set_ylabel("Average Candidates Explored")
    ax2.set_title("Candidates Explored by Synthesizer and Difficulty")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(diff_labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    chart_path = "results/comparison_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison chart saved to {chart_path}")
    
    # Chart 2: Per-benchmark success heatmap
    benchmarks = list(dict.fromkeys(r["benchmark"] for r in results))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    heatmap_data = []
    for synth in synthesizers:
        row = []
        for bench in benchmarks:
            bench_results = [
                r for r in results
                if r["synthesizer"] == synth and r["benchmark"] == bench
            ]
            if bench_results:
                # 1 = success + held out pass, 0.5 = success only, 0 = fail
                r = bench_results[0]
                if r["held_out_pass"]:
                    row.append(1.0)
                elif r["success"]:
                    row.append(0.5)
                else:
                    row.append(0.0)
            else:
                row.append(0.0)
        heatmap_data.append(row)
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(synthesizers)))
    ax.set_yticklabels(synthesizers)
    ax.set_title("Per-Benchmark Results (Green=Pass, Red=Fail, Yellow=Partial)")
    
    plt.colorbar(im, ax=ax, label="Result (1=Full Pass, 0.5=Training Pass, 0=Fail)")
    plt.tight_layout()
    
    heatmap_path = "results/benchmark_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Benchmark heatmap saved to {heatmap_path}")


def generate_detailed_breakdown(results: List[Dict[str, Any]]) -> None:
    """Print detailed per-benchmark results."""
    synthesizers = list(dict.fromkeys(r["synthesizer"] for r in results))
    benchmarks = list(dict.fromkeys(r["benchmark"] for r in results))
    
    print("\n" + "="*80)
    print("DETAILED BENCHMARK RESULTS")
    print("="*80)
    
    for bench in benchmarks:
        bench_results = {r["synthesizer"]: r for r in results if r["benchmark"] == bench}
        
        if bench_results:
            first = list(bench_results.values())[0]
            print(f"\n{bench} [{first['difficulty']}]")
            print(f"  Expected: {first['expected_program']}")
            
            for synth in synthesizers:
                if synth in bench_results:
                    r = bench_results[synth]
                    status = "PASS" if r["held_out_pass"] else ("PARTIAL" if r["success"] else "FAIL")
                    prog = r["program_found"] or "None"
                    if len(prog) > 50:
                        prog = prog[:47] + "..."
                    print(f"  {synth:<12}: {status:<8} | {r['candidates']:>6} candidates | "
                          f"{r['time_seconds']:.2f}s | {prog}")


if __name__ == "__main__":
    # Load results
    results_path = "results/raw_results.csv"
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run run_eval.py first.")
        sys.exit(1)
    
    print(f"Loading results from {results_path}...")
    results = load_results(results_path)
    print(f"Loaded {len(results)} result entries")
    
    # Generate all outputs
    generate_summary_table(results)
    generate_candidates_table(results)
    generate_time_table(results)
    generate_detailed_breakdown(results)
    generate_charts(results)
    
    print("\nAll analysis complete. Check the results/ folder.")
