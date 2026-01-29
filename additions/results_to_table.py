"""
Convert benchmark_results.json into comprehensive tables for analysis.

Usage: python results_to_table.py
       python results_to_table.py --export-csv results.csv
"""

import argparse
import json
import csv
from pathlib import Path


def load_results(json_path="additions/benchmark_results.json"):
    """Load benchmark results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def print_full_results_table(data):
    """Print comprehensive results table."""
    results = data["results"]
    
    print(f"\n{'='*140}")
    print(f"COMPLETE DATASET - ALL EXPERIMENTS")
    print(f"Platform: {data['platform']}, Batch Size: {data['batch_size']}")
    print(f"{'='*140}\n")
    
    # Header
    header = (f"{'Model':<15} {'Category':<18} {'Optimizer':<22} "
              f"{'Runtime(s)':<12} {'Throughput':<15} {'Latency(us)':<15} "
              f"{'Parts':<6} {'DSP':<7} {'BRAM':<7} {'LUT':<8} {'FF':<8} {'Status':<8}")
    print(header)
    print("=" * 140)
    
    # Sort by model category, then model name, then optimizer
    sorted_results = sorted(results, key=lambda x: (
        x.get("model_category", ""),
        x.get("model", ""),
        x.get("optimizer", "")
    ))
    
    prev_category = None
    prev_model = None
    
    for r in sorted_results:
        model = r.get("model", "")
        category = r.get("model_category", "")
        optimizer = r.get("optimizer", "")
        
        # Add separator between categories
        if category != prev_category and prev_category is not None:
            print("-" * 140)
        
        model_display = model.upper() if model != prev_model else ""
        category_display = category.capitalize() if model != prev_model else ""
        
        if r.get("success"):
            throughput = r.get("throughput", 0)
            latency = r.get("latency_us", 0)
            runtime = r.get("runtime_seconds", 0)
            parts = r.get("num_partitions", 0)
            resources = r.get("resources", {})
            feasible = "OK" if r.get("feasible") else "INFEAS"
            
            print(f"{model_display:<15} "
                  f"{category_display:<18} "
                  f"{optimizer:<22} "
                  f"{runtime:<12.2f} "
                  f"{throughput:<15.6f} "
                  f"{latency:>14,.0f} "
                  f"{parts:<6} "
                  f"{resources.get('DSP', 0):<7} "
                  f"{resources.get('BRAM', 0):<7} "
                  f"{resources.get('LUT', 0):<8} "
                  f"{resources.get('FF', 0):<8} "
                  f"{feasible:<8}")
        else:
            error = r.get("error", "Unknown error")[:30]
            print(f"{model_display:<15} "
                  f"{category_display:<18} "
                  f"{optimizer:<22} "
                  f"{'FAILED':<12} "
                  f"{error:<15} "
                  f"{'-':<15} {'-':<6} {'-':<7} {'-':<7} {'-':<8} {'-':<8} {'ERROR':<8}")
        
        prev_category = category
        prev_model = model
    
    print("=" * 140)


def print_summary_by_model(data):
    """Print summary grouped by model."""
    results = data["results"]
    
    print(f"\n{'='*120}")
    print("SUMMARY BY MODEL - Best Performance")
    print(f"{'='*120}\n")
    
    # Group by model
    by_model = {}
    for r in results:
        model = r.get("model")
        if model:
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(r)
    
    print(f"{'Model':<15} {'Category':<18} {'Best Optimizer':<22} {'Best Throughput':<18} {'Avg Runtime(s)':<16} {'Experiments':<12}")
    print("-" * 120)
    
    for model in sorted(by_model.keys()):
        model_results = by_model[model]
        successful = [r for r in model_results if r.get("success") and r.get("feasible")]
        
        if successful:
            category = successful[0].get("model_category", "").capitalize()
            best = max(successful, key=lambda x: x.get("throughput", 0))
            avg_runtime = sum(r.get("runtime_seconds", 0) for r in successful) / len(successful)
            
            print(f"{model.upper():<15} "
                  f"{category:<18} "
                  f"{best['optimizer']:<22} "
                  f"{best['throughput']:<18.6f} "
                  f"{avg_runtime:<16.2f} "
                  f"{len(model_results):<12}")
        else:
            print(f"{model.upper():<15} {'N/A':<18} {'FAILED':<22} {'-':<18} {'-':<16} {len(model_results):<12}")
    
    print("=" * 120)


def print_summary_by_optimizer(data):
    """Print summary grouped by optimizer."""
    results = data["results"]
    
    print(f"\n{'='*120}")
    print("SUMMARY BY OPTIMIZER - Performance Comparison")
    print(f"{'='*120}\n")
    
    # Group by optimizer
    by_optimizer = {}
    for r in results:
        opt = r.get("optimizer")
        if opt and r.get("success") and r.get("feasible"):
            if opt not in by_optimizer:
                by_optimizer[opt] = []
            by_optimizer[opt].append(r)
    
    print(f"{'Optimizer':<25} {'Models Tested':<15} {'Avg Runtime(s)':<18} {'Avg Throughput':<18} {'Best Throughput':<18}")
    print("-" * 120)
    
    for optimizer in sorted(by_optimizer.keys()):
        opt_results = by_optimizer[optimizer]
        
        avg_runtime = sum(r.get("runtime_seconds", 0) for r in opt_results) / len(opt_results)
        avg_throughput = sum(r.get("throughput", 0) for r in opt_results) / len(opt_results)
        best_throughput = max(r.get("throughput", 0) for r in opt_results)
        
        print(f"{optimizer:<25} "
              f"{len(opt_results):<15} "
              f"{avg_runtime:<18.2f} "
              f"{avg_throughput:<18.6f} "
              f"{best_throughput:<18.6f}")
    
    print("=" * 120)


def export_to_csv(data, csv_path):
    """Export all results to CSV file."""
    results = data["results"]
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'model', 'model_category', 'optimizer', 'success', 'feasible',
            'runtime_seconds', 'throughput', 'latency_us', 
            'num_partitions', 'dsp', 'bram', 'lut', 'ff',
            'platform', 'batch_size', 'initial_feasible', 'time_to_first_feasible', 'error'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            resources = r.get('resources', {})
            row = {
                'model': r.get('model', ''),
                'model_category': r.get('model_category', ''),
                'optimizer': r.get('optimizer', ''),
                'success': r.get('success', False),
                'feasible': r.get('feasible', False),
                'runtime_seconds': r.get('runtime_seconds', 0),
                'throughput': r.get('throughput', 0),
                'latency_us': r.get('latency_us', 0),
                'num_partitions': r.get('num_partitions', 0),
                'dsp': resources.get('DSP', 0),
                'bram': resources.get('BRAM', 0),
                'lut': resources.get('LUT', 0),
                'ff': resources.get('FF', 0),
                'platform': data.get('platform', ''),
                'batch_size': data.get('batch_size', 0),
                'initial_feasible': r.get('initial_feasible', True),
                'time_to_first_feasible': r.get('time_to_first_feasible', ''),
                'error': r.get('error', '')
            }
            writer.writerow(row)
    
    print(f"\nExported to CSV: {csv_path}")


def print_category_analysis(data):
    """Analyze performance by model category."""
    results = data["results"]
    
    print(f"\n{'='*120}")
    print("ANALYSIS BY MODEL CATEGORY - Scaling Behavior")
    print(f"{'='*120}\n")
    
    # Group by category
    by_category = {}
    for r in results:
        if r.get("success") and r.get("feasible"):
            category = r.get("model_category", "unknown")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(r)
    
    print(f"{'Category':<20} {'Models':<10} {'Avg Runtime(s)':<18} {'Avg Throughput':<18} {'Runtime Range':<25}")
    print("-" * 120)
    
    for category in sorted(by_category.keys()):
        cat_results = by_category[category]
        
        # Count unique models in this category
        unique_models = len(set(r.get("model") for r in cat_results))
        
        avg_runtime = sum(r.get("runtime_seconds", 0) for r in cat_results) / len(cat_results)
        avg_throughput = sum(r.get("throughput", 0) for r in cat_results) / len(cat_results)
        
        min_runtime = min(r.get("runtime_seconds", 0) for r in cat_results)
        max_runtime = max(r.get("runtime_seconds", 0) for r in cat_results)
        runtime_range = f"{min_runtime:.1f}s - {max_runtime:.1f}s"
        
        print(f"{category.capitalize():<20} "
              f"{unique_models:<10} "
              f"{avg_runtime:<18.2f} "
              f"{avg_throughput:<18.6f} "
              f"{runtime_range:<25}")
    
    print("=" * 120)


def print_infeasibility_recovery_analysis(data):
    """Analyze how optimizers handle initially infeasible designs."""
    results = data["results"]
    
    print(f"\n{'='*120}")
    print("INFEASIBILITY RECOVERY ANALYSIS - Optimizer Convergence")
    print(f"{'='*120}\n")
    
    # Find results that started infeasible
    infeasible_starts = [r for r in results if r.get("success") and not r.get("initial_feasible", True)]
    
    if not infeasible_starts:
        print("No experiments started with infeasible designs.\n")
        print("=" * 120)
        return
    
    print(f"Total experiments starting infeasible: {len(infeasible_starts)}\n")
    
    # Group by optimizer
    by_optimizer = {}
    for r in infeasible_starts:
        opt = r.get("optimizer")
        if opt:
            if opt not in by_optimizer:
                by_optimizer[opt] = {"recovered": [], "failed": []}
            
            if r.get("feasible"):
                by_optimizer[opt]["recovered"].append(r)
            else:
                by_optimizer[opt]["failed"].append(r)
    
    print(f"{'Optimizer':<25} {'Started Infeas.':<18} {'Recovered':<12} {'Recovery Rate':<15} {'Avg Time-to-Feasible':<22}")
    print("-" * 120)
    
    for optimizer in sorted(by_optimizer.keys()):
        opt_data = by_optimizer[optimizer]
        recovered = opt_data["recovered"]
        failed = opt_data["failed"]
        total = len(recovered) + len(failed)
        
        recovery_rate = 100.0 * len(recovered) / total if total > 0 else 0
        
        if recovered:
            avg_time = sum(r.get("time_to_first_feasible", 0) for r in recovered) / len(recovered)
            time_str = f"{avg_time:.2f}s"
        else:
            time_str = "N/A"
        
        print(f"{optimizer:<25} "
              f"{total:<18} "
              f"{len(recovered):<12} "
              f"{recovery_rate:<14.1f}% "
              f"{time_str:<22}")
    
    # Show detailed breakdown by model
    print(f"\nDetailed Recovery by Model:")
    print("-" * 120)
    
    model_recovery = {}
    for r in infeasible_starts:
        model = r.get("model")
        if model not in model_recovery:
            model_recovery[model] = []
        model_recovery[model].append(r)
    
    for model in sorted(model_recovery.keys()):
        model_results = model_recovery[model]
        recovered = [r for r in model_results if r.get("feasible")]
        
        print(f"\n  {model.upper()}: {len(recovered)}/{len(model_results)} recovered")
        for r in model_results:
            opt = r.get("optimizer")
            feasible = r.get("feasible")
            time_to_feas = r.get("time_to_first_feasible")
            
            if feasible and time_to_feas:
                print(f"    - {opt:<22} OK in ≤{time_to_feas:.2f}s")
            elif feasible:
                print(f"    - {opt:<22} OK")
            else:
                print(f"    - {opt:<22} Still infeasible")
    
    print("\n" + "=" * 120)
    

def main():
    parser = argparse.ArgumentParser(description="Convert benchmark results to tables")
    parser.add_argument("--input", default="additions/benchmark_results.json",
                       help="Input JSON file")
    parser.add_argument("--export-csv", 
                       help="Export to CSV file")
    args = parser.parse_args()
    
    # Load results
    try:
        data = load_results(args.input)
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Print various tables
    print_full_results_table(data)
    print_summary_by_model(data)
    print_summary_by_optimizer(data)
    print_category_analysis(data)
    print_infeasibility_recovery_analysis(data)  # New: analyze optimizer convergence
    
    # Export to CSV if requested
    if args.export_csv:
        export_to_csv(data, args.export_csv)
    
    print(f"\n\nTotal experiments: {len(data['results'])}")
    print(f"Successful: {sum(1 for r in data['results'] if r.get('success'))}")
    print(f"Feasible: {sum(1 for r in data['results'] if r.get('success') and r.get('feasible'))}\n")


if __name__ == "__main__":
    main()
