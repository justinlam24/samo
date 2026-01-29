"""
Benchmark script to compare SAMO optimizers across a model set.
Tests 3 models (small, depth stress, heterogeneity stress) with 3 optimizers.

Usage: python benchmark_model_set.py --platform platforms/u250_1slr.json
"""

import argparse
import json
import time
import sys
import copy
import logging
import random
from pathlib import Path
import importlib

import numpy as np

# Import SAMO components
import samo.backend.fpgaconvnet.parser as fpgaconvnet_parser
from samo.optimiser.annealing import SimulatedAnnealing
from samo.optimiser.rule import RuleBased
from samo.optimiser.brute import BruteForce


# Model set definition - Comprehensive model coverage for dataset collection
MODEL_SET = {
    # Small baseline models
    "simple": {
        "path": "models/simple.onnx",
        "description": "Simplest baseline - Minimal CNN",
        "category": "small"
    },
    "lenet": {
        "path": "models/lenet.onnx",
        "description": "Small/Sanity Check - Classic LeNet CNN",
        "category": "small"
    },
    
    # Fully connected networks (homogeneous layers)
    "sfc": {
        "path": "models/sfc.onnx",
        "description": "Small Fully Connected - Dense layers only",
        "category": "fully_connected"
    },
    "tfc": {
        "path": "models/tfc.onnx",
        "description": "Medium Fully Connected - Deeper dense network",
        "category": "fully_connected"
    },
    "lfc": {
        "path": "models/lfc.onnx",
        "description": "Large Fully Connected - Deep dense network",
        "category": "fully_connected"
    },
    
    # Depth stress (many sequential layers)
    "mpcnn": {
        "path": "models/mpcnn.onnx",
        "description": "Depth Stress - Multi-layer CNN with pooling",
        "category": "depth"
    },
    "vgg11": {
        "path": "models/vgg11.onnx",
        "description": "Deep Depth Stress - VGG-11 architecture",
        "category": "depth"
    },
    
    # Heterogeneity stress (varied layer types)
    "cnv": {
        "path": "models/cnv.onnx",
        "description": "Heterogeneity Stress - CNV with varied conv layers",
        "category": "heterogeneity"
    },
    "alexnet": {
        "path": "models/alexnet.onnx",
        "description": "Heterogeneity Stress - AlexNet varied kernels",
        "category": "heterogeneity"
    },
    
    # Efficient/optimized architectures
    "mobilenetv1": {
        "path": "models/mobilenetv1.onnx",
        "description": "Efficient Architecture - MobileNet depthwise separable",
        "category": "efficient"
    },
    "alexnet_fpgaconvnet": {
        "path": "models/alexnet_fpgaconvnet.onnx",
        "description": "FPGA-Optimized - AlexNet fpgaConvNet variant",
        "category": "fpga_optimized"
    },
    "vgg16_fpgaconvnet": {
        "path": "models/vgg16_fpgaconvnet.onnx",
        "description": "FPGA-Optimized - VGG16 fpgaConvNet variant",
        "category": "fpga_optimized"
    }
}


def run_optimizer(optimizer_class, network, optimizer_name, model_name, **kwargs):
    """
    Run a single optimizer and return results focused on throughput.
    """
    print(f"  Running {optimizer_name}...", end=" ", flush=True)
    
    # Create optimizer instance with deep copy
    opt = optimizer_class(copy.deepcopy(network), **kwargs)
    
    # CRITICAL: Split network completely first (like CLI does)
    # Brute Force requires a single partition
    can_split = optimizer_name != "Brute Force"
    while can_split:
        can_split = False
        for i in range(len(opt.network.partitions)):
            valid_splits = opt.network.valid_splits(i)
            network_copy = copy.deepcopy(opt.network)
            if valid_splits:
                can_split = True
                prev = opt.network.check_constraints()
                opt.network.split(i, valid_splits[0])
                if prev and not opt.network.check_constraints():
                    can_split = False
                    opt.network = network_copy
    
    # Check initial design feasibility (but continue anyway - optimizer may fix it)
    initial_feasible = bool(opt.network.check_constraints())  # Convert numpy.bool_ to Python bool for JSON
    if not initial_feasible:
        print(f"(Initial infeasible, letting optimizer attempt to fix)...", end=" ", flush=True)
    
    # Time the optimization
    start_time = time.time()
    opt.start_time = start_time  # CRITICAL: Set start_time for the optimizer
    
    # Create individual log files for each optimizer run
    import os
    log_dir = f"additions/logs/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # The annealing optimizer writes to outputs/log.csv, so create that directory
    os.makedirs("outputs", exist_ok=True)
    
    try:
        opt.optimise()
        elapsed_time = time.time() - start_time
        
        # Copy the log.csv file if it exists (created by annealing)
        if os.path.exists("outputs/log.csv") and optimizer_name == "Simulated Annealing":
            import shutil
            shutil.copy("outputs/log.csv", f"{log_dir}/annealing_log.csv")
        
        # Collect results - FOCUS ON THROUGHPUT
        throughput = opt.network.eval_throughput()
        latency_us = opt.network.eval_latency()
        feasible = opt.network.check_constraints()
        num_partitions = len(opt.network.partitions)
        
        # Calculate time-to-first-feasible if started infeasible
        # Note: This is an upper bound - actual time may be less, but we can't measure
        # without instrumenting the optimizer internals
        time_to_first_feasible = None
        if not initial_feasible and feasible:
            time_to_first_feasible = float(elapsed_time)  # Conservative estimate
        
        # Collect total resource utilization across all partitions
        total_resources = {"DSP": 0, "BRAM": 0, "LUT": 0, "FF": 0}
        for partition in opt.network.partitions:
            rsc = partition.eval_resource()
            for key in total_resources:
                total_resources[key] += rsc.get(key, 0)
        
        results = {
            "optimizer": optimizer_name,
            "success": True,
            "runtime_seconds": float(elapsed_time),
            "throughput": float(throughput),
            "latency_us": float(latency_us),
            "num_partitions": int(num_partitions),
            "resources": {k: int(v) for k, v in total_resources.items()},
            "feasible": bool(feasible),
            "initial_feasible": initial_feasible,  # Track whether initial design was feasible
            "time_to_first_feasible": time_to_first_feasible,  # Time to converge from infeasible to feasible
        }
        
        status_msg = f"DONE ({elapsed_time:.1f}s)"
        if not initial_feasible and feasible:
            status_msg += f" - Optimizer fixed infeasible design in ≤{elapsed_time:.1f}s!"
        elif not feasible:
            status_msg += " - Still infeasible"
        print(status_msg)
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"FAILED ({str(e)[:50]}...)")
        
        results = {
            "optimizer": optimizer_name,
            "success": False,
            "runtime_seconds": float(elapsed_time),
            "error": str(e),
        }
    
    return results


def run_model_benchmark(model_name, model_info, platform, batch_size, seed=None, include_brute_force=False):
    """
    Run all optimizers on a single model.
    """
    print(f"\n{'='*70}")
    print(f"Model: {model_name.upper()} - {model_info['description']}")
    print(f"{'='*70}")
    
    # Set random seed for reproducibility
    if seed is None:
        seed = random.randint(0, 2**32-1)
    random.seed(seed)
    np.random.seed(seed)
    
    # Parse network
    try:
        network = fpgaconvnet_parser.parse(
            model_info["path"], 
            platform, 
            batch_size
        )
        network.objective = "throughput"  # Focus on throughput
        network.enable_reconf = True
        
        # Initialize network
        for partition in network.partitions:
            partition.reset()
            
    except Exception as e:
        print(f"✗ Failed to parse model: {e}")
        return None
    
    # Define optimizers - Brute Force optional via flag
    optimizer_configs = []
    
    if include_brute_force:
        optimizer_configs.append({
            "class": BruteForce,
            "name": "Brute Force",
            "kwargs": {}
        })
    
    optimizer_configs.extend([
        {
            "class": RuleBased,
            "name": "Rule-Based",
            "kwargs": {}
        },
        {
            "class": SimulatedAnnealing,
            "name": "Simulated Annealing",
            "kwargs": {
                "T": 5.0,        # Lower starting temperature
                "T_min": 0.5,    # Higher minimum (fewer steps)
                "cool": 0.95,    # Faster cooling (5% per step)
                "iterations": 20, # Fewer iterations per temp
            }
        }
    ])
    
    # Run all optimizers
    results = []
    for config in optimizer_configs:
        result = run_optimizer(
            config["class"],
            network,
            config["name"],
            model_name,  # Pass model name for logging
            **config["kwargs"]
        )
        result["model"] = model_name
        result["model_category"] = model_info["category"]
        results.append(result)
    
    return results


def print_summary_table(all_results):
    """
    Print high-level summary of benchmark run.
    For detailed analysis, use: python additions/results_to_table.py
    """
    print(f"\n\n{'='*100}")
    print("BENCHMARK SUMMARY - High Level Overview")
    print(f"{'='*100}\n")
    
    # Overall statistics
    total = len(all_results)
    successful = sum(1 for r in all_results if r.get("success"))
    feasible = sum(1 for r in all_results if r.get("success") and r.get("feasible"))
    
    print(f"Total experiments: {total}")
    print(f"Successful runs: {successful} ({100*successful/total:.1f}%)")
    print(f"Feasible solutions: {feasible} ({100*feasible/total:.1f}%)")
    
    # Models tested
    models_tested = len(set(r.get("model") for r in all_results))
    print(f"Unique models tested: {models_tested}")
    
    # Quick status by model
    print(f"\n{'='*100}")
    print("Model Status Summary:")
    print(f"{'='*100}")
    print(f"{'Model':<15} {'Category':<18} {'Status':<30}")
    print("-" * 100)
    
    models = list(MODEL_SET.keys())
    for model in models:
        model_results = [r for r in all_results if r.get("model") == model]
        category = MODEL_SET[model]["category"]
        
        successful_count = sum(1 for r in model_results if r.get("success") and r.get("feasible"))
        total_count = len(model_results)
        
        if successful_count == total_count:
            status = f"OK - All {total_count} optimizers passed"
        elif successful_count > 0:
            status = f"PARTIAL - {successful_count}/{total_count} passed"
        else:
            status = f"FAILED - All optimizers failed"
        
        print(f"{model.upper():<15} {category.capitalize():<18} {status:<30}")
    
    # Optimizer performance overview
    print(f"\n{'='*100}")
    print("Optimizer Performance Overview:")
    print(f"{'='*100}")
    print(f"{'Optimizer':<25} {'Success Rate':<15} {'Avg Runtime(s)':<18}")
    print("-" * 100)
    
    for opt_name in ["Brute Force", "Rule-Based", "Simulated Annealing"]:
        opt_results = [r for r in all_results if r["optimizer"] == opt_name]
        if opt_results:
            opt_success = [r for r in opt_results if r.get("success") and r.get("feasible")]
            success_rate = 100.0 * len(opt_success) / len(opt_results)
            
            if opt_success:
                avg_runtime = sum(r["runtime_seconds"] for r in opt_success) / len(opt_success)
                print(f"{opt_name:<25} {success_rate:>6.1f}% ({len(opt_success)}/{len(opt_results)}){'':<3} {avg_runtime:>8.2f}s")
            else:
                print(f"{opt_name:<25} {success_rate:>6.1f}% ({len(opt_success)}/{len(opt_results)}){'':<3} {'N/A':>8}")
    
    print(f"\n{'='*100}")
    print("For detailed per-model analysis, run:")
    print("  python additions/results_to_table.py")
    print("  python additions/results_to_table.py --export-csv results.csv")
    print(f"{'='*100}\n")


def main():
    # Set up logging (like the CLI does)
    logging.basicConfig(filename='additions/benchmark.log', filemode='w', level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Benchmark SAMO optimizers on model set")
    parser.add_argument("--platform", default="platforms/u250_1slr.json",
                       help="Path to platform JSON (default: u250_1slr)")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size (default: 256)")
    parser.add_argument("--output", default="additions/benchmark_results.json",
                       help="Output JSON file")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--include-brute-force", action="store_true",
                       help="Include Brute Force optimizer (WARNING: very slow, can take hours/days)")
    args = parser.parse_args()
    
    # Load platform
    try:
        with open(args.platform, "r") as f:
            platform = json.load(f)
        print(f"Platform loaded: {platform.get('name', 'Unknown')}")
    except Exception as e:
        print(f"Error loading platform: {e}")
        sys.exit(1)
    
    # Run benchmarks for all models
    all_results = []
    
    for model_name, model_info in MODEL_SET.items():
        results = run_model_benchmark(model_name, model_info, platform, args.batch_size, args.seed, args.include_brute_force)
        if results:
            all_results.extend(results)
    
    # Print summary
    if all_results:
        print_summary_table(all_results)
        
        # Save results
        output_data = {
            "platform": args.platform,
            "batch_size": args.batch_size,
            "model_set": MODEL_SET,
            "results": all_results,
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {args.output}\n")
    else:
        print("\nNo results to report\n")


if __name__ == "__main__":
    main()
