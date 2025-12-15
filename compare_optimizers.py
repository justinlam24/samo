"""
Benchmark script to compare different SAMO optimizers.
Usage: python compare_optimizers.py --model models/tfc.onnx --platform platforms/zedboard.json
"""

import argparse
import json
import time
import sys
import copy
from pathlib import Path

# Import SAMO components
from samo.backend.fpgaconvnet import parser as fpgaconvnet_parser
from samo.optimiser.annealing import SimulatedAnnealing
from samo.optimiser.rule import RuleBased
from samo.optimiser.genetic import GeneticAlgorithm


def run_optimizer(optimizer_class, network, optimizer_name, **kwargs):
    """
    Run a single optimizer and return results.
    """
    print(f"\n{'='*60}")
    print(f"Running {optimizer_name} Optimizer")
    print(f"{'='*60}\n")
    
    # Create optimizer instance
    opt = optimizer_class(copy.deepcopy(network), **kwargs)
    
    # Time the optimization
    start_time = time.time()
    opt.start_time = start_time
    
    try:
        opt.optimise()
        elapsed_time = time.time() - start_time
        
        # Collect results
        results = {
            "optimizer": optimizer_name,
            "success": True,
            "time_seconds": elapsed_time,
            "latency_us": opt.network.eval_latency(),
            "throughput": opt.network.eval_throughput(),
            "feasible": opt.network.check_constraints(),
            "num_partitions": len(opt.network.partitions),
        }
        
        # Collect resource utilization per partition
        resources_per_partition = []
        for i, partition in enumerate(opt.network.partitions):
            rsc = partition.eval_resource()
            resources_per_partition.append({
                "partition_index": i,
                "DSP": rsc.get("DSP", 0),
                "BRAM": rsc.get("BRAM", 0),
                "LUT": rsc.get("LUT", 0),
                "FF": rsc.get("FF", 0),
                "latency_cycles": partition.eval_latency(),
            })
        
        results["resources"] = resources_per_partition
        
        print(f"\n✓ {optimizer_name} completed successfully!")
        print(f"  Time: {elapsed_time:.2f}s")
        print(f"  Latency: {results['latency_us']:.4f} μs")
        print(f"  Throughput: {results['throughput']:.6f} img/μs")
        print(f"  Feasible: {results['feasible']}")
        
    except Exception as e:
        print(f"\n✗ {optimizer_name} failed: {str(e)}")
        results = {
            "optimizer": optimizer_name,
            "success": False,
            "error": str(e),
            "time_seconds": time.time() - start_time,
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare SAMO optimizers")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--platform", required=True, help="Path to platform JSON")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--optimizers", nargs="+", 
                       choices=["rule", "genetic", "annealing"],
                       default=["rule", "genetic", "annealing"],
                       help="Optimizers to compare")
    parser.add_argument("--output", default="comparison_results.json",
                       help="Output JSON file")
    args = parser.parse_args()
    
    # Load platform
    with open(args.platform, "r") as f:
        platform = json.load(f)
    
    # Parse network
    print(f"Parsing model: {args.model}")
    network = fpgaconvnet_parser.parse(args.model, platform, args.batch_size)
    network.objective = "latency"
    network.enable_reconf = True
    
    # Initialize network
    for partition in network.partitions:
        partition.reset()
    
    print(f"\nNetwork loaded successfully!")
    print(f"  Platform: {platform.get('name', 'Unknown')}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Initial partitions: {len(network.partitions)}")
    
    # Define optimizers to test
    optimizer_configs = []
    
    if "rule" in args.optimizers:
        optimizer_configs.append({
            "class": RuleBased,
            "name": "Rule-Based",
            "kwargs": {}
        })
    
    if "genetic" in args.optimizers:
        optimizer_configs.append({
            "class": GeneticAlgorithm,
            "name": "Genetic Algorithm",
            "kwargs": {
                "population_size": 30,
                "generations": 50,
                "mutation_rate": 0.3,
                "crossover_rate": 0.7,
            }
        })
    
    if "annealing" in args.optimizers:
        optimizer_configs.append({
            "class": SimulatedAnnealing,
            "name": "Simulated Annealing",
            "kwargs": {
                "T": 10.0,
                "T_min": 0.1,  # Shorter run for comparison
                "cool": 0.99,
                "iterations": 50,
            }
        })
    
    # Run all optimizers
    all_results = []
    
    for config in optimizer_configs:
        result = run_optimizer(
            config["class"],
            network,
            config["name"],
            **config["kwargs"]
        )
        all_results.append(result)
    
    # Print comparison table
    print(f"\n\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")
    
    print(f"{'Optimizer':<25} {'Time (s)':<12} {'Latency (μs)':<15} {'Throughput':<15} {'Feasible':<10}")
    print(f"{'-'*80}")
    
    for result in all_results:
        if result["success"]:
            print(f"{result['optimizer']:<25} "
                  f"{result['time_seconds']:<12.2f} "
                  f"{result['latency_us']:<15.4f} "
                  f"{result['throughput']:<15.6f} "
                  f"{'✓' if result['feasible'] else '✗':<10}")
        else:
            print(f"{result['optimizer']:<25} {'FAILED':<12} {'-':<15} {'-':<15} {'-':<10}")
    
    # Find best results
    successful_results = [r for r in all_results if r["success"] and r["feasible"]]
    
    if successful_results:
        best_latency = min(successful_results, key=lambda x: x["latency_us"])
        fastest_time = min(successful_results, key=lambda x: x["time_seconds"])
        
        print(f"\n{'='*80}")
        print("BEST RESULTS")
        print(f"{'='*80}\n")
        print(f"🏆 Best Latency: {best_latency['optimizer']} ({best_latency['latency_us']:.4f} μs)")
        print(f"⚡ Fastest Optimization: {fastest_time['optimizer']} ({fastest_time['time_seconds']:.2f}s)")
        
        # Calculate improvement over baseline (rule-based)
        rule_based = next((r for r in successful_results if "Rule" in r["optimizer"]), None)
        if rule_based and best_latency["optimizer"] != rule_based["optimizer"]:
            improvement = ((rule_based["latency_us"] - best_latency["latency_us"]) / 
                          rule_based["latency_us"] * 100)
            print(f"\n📈 Improvement over Rule-Based: {improvement:.2f}%")
    
    # Save results to JSON
    output_data = {
        "model": args.model,
        "platform": args.platform,
        "batch_size": args.batch_size,
        "results": all_results,
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
