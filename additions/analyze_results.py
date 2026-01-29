#!/usr/bin/env python3
"""
Analyze benchmark results from CSV and generate Excel-ready summary tables.

Usage:
    python analyze_results.py <results_csv>
    
Example:
    python analyze_results.py additions/results/all_results.csv

Requirements:
    pip install pandas openpyxl
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Check for openpyxl (needed for Excel export)
try:
    import openpyxl
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False
    print("WARNING: openpyxl not installed. Excel export will be skipped.")
    print("Install with: pip install openpyxl")


def load_results(csv_path: str) -> pd.DataFrame:
    """Load and validate the results CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} experiments from {csv_path}")
    return df


def summary_by_optimizer(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics grouped by optimizer."""
    summary = df.groupby('optimizer').agg({
        'success': ['sum', 'count', lambda x: f"{(x.sum() / len(x) * 100):.1f}%"],
        'final_throughput': ['mean', 'median', 'max'],
        'runtime_seconds': ['mean', 'median'],
        'total_partitions': 'mean',
    }).round(6)
    
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary = summary.rename(columns={
        'success_sum': 'Successes',
        'success_count': 'Total_Runs',
        'success_<lambda>': 'Success_Rate',
        'final_throughput_mean': 'Avg_Throughput',
        'final_throughput_median': 'Median_Throughput',
        'final_throughput_max': 'Max_Throughput',
        'runtime_seconds_mean': 'Avg_Runtime_s',
        'runtime_seconds_median': 'Median_Runtime_s',
        'total_partitions_mean': 'Avg_Partitions',
    })
    
    return summary


def summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics grouped by model."""
    summary = df.groupby(['model_name', 'domain']).agg({
        'success': ['sum', 'count', lambda x: f"{(x.sum() / len(x) * 100):.1f}%"],
        'final_throughput': ['mean', 'max'],
        'total_partitions': 'mean',
    }).round(6)
    
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary = summary.rename(columns={
        'success_sum': 'Successes',
        'success_count': 'Total_Runs',
        'success_<lambda>': 'Success_Rate',
        'final_throughput_mean': 'Avg_Throughput',
        'final_throughput_max': 'Max_Throughput',
        'total_partitions_mean': 'Avg_Partitions',
    })
    
    return summary


def summary_by_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for SA grouped by temperature."""
    sa_df = df[df['optimizer'] == 'Simulated Annealing'].copy()
    
    if len(sa_df) == 0:
        print("No Simulated Annealing results found")
        return pd.DataFrame()
    
    summary = sa_df.groupby('temperature').agg({
        'success': ['sum', 'count', lambda x: f"{(x.sum() / len(x) * 100):.1f}%"],
        'final_throughput': ['mean', 'max'],
        'runtime_seconds': 'mean',
        'folding_moves_accepted': 'mean',
        'partition_moves_accepted': 'mean',
    }).round(6)
    
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary = summary.rename(columns={
        'success_sum': 'Successes',
        'success_count': 'Total_Runs',
        'success_<lambda>': 'Success_Rate',
        'final_throughput_mean': 'Avg_Throughput',
        'final_throughput_max': 'Max_Throughput',
        'runtime_seconds_mean': 'Avg_Runtime_s',
        'folding_moves_accepted_mean': 'Avg_Folding_Accepted',
        'partition_moves_accepted_mean': 'Avg_Partition_Accepted',
    })
    
    return summary


def optimizer_comparison_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Compare optimizer performance on each model."""
    comparison = df.groupby(['model_name', 'optimizer']).agg({
        'success': ['sum', lambda x: f"{(x.sum() / len(x) * 100):.1f}%"],
        'final_throughput': 'max',
        'runtime_seconds': 'mean',
    }).round(6)
    
    comparison.columns = ['_'.join(col).strip('_') for col in comparison.columns.values]
    comparison = comparison.rename(columns={
        'success_sum': 'Successes',
        'success_<lambda>': 'Success_Rate',
        'final_throughput_max': 'Best_Throughput',
        'runtime_seconds_mean': 'Avg_Runtime_s',
    })
    
    return comparison


def failure_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze all failures with error messages."""
    failures = df[df['success'] == False].copy()
    
    if len(failures) == 0:
        print("No failures found - all experiments succeeded!")
        return pd.DataFrame()
    
    failure_summary = failures[['model_name', 'optimizer', 'temperature', 'trial', 'error', 'runtime_seconds']].copy()
    failure_summary = failure_summary.sort_values(['model_name', 'optimizer', 'temperature'])
    
    return failure_summary


def resource_utilization_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze resource utilization for successful runs."""
    successful = df[df['success'] == True].copy()
    
    if len(successful) == 0:
        print("No successful runs found")
        return pd.DataFrame()
    
    resource_summary = successful.groupby('optimizer').agg({
        'final_DSP': ['mean', 'max'],
        'final_BRAM': ['mean', 'max'],
        'final_LUT': ['mean', 'max'],
        'final_FF': ['mean', 'max'],
        'max_partition_resource_util': ['mean', 'max'],
    }).round(2)
    
    resource_summary.columns = ['_'.join(col).strip('_') for col in resource_summary.columns.values]
    resource_summary = resource_summary.rename(columns={
        'final_DSP_mean': 'Avg_DSP',
        'final_DSP_max': 'Max_DSP',
        'final_BRAM_mean': 'Avg_BRAM',
        'final_BRAM_max': 'Max_BRAM',
        'final_LUT_mean': 'Avg_LUT',
        'final_LUT_max': 'Max_LUT',
        'final_FF_mean': 'Avg_FF',
        'final_FF_max': 'Max_FF',
        'max_partition_resource_util_mean': 'Avg_Max_Partition_Util',
        'max_partition_resource_util_max': 'Max_Partition_Util',
    })
    
    return resource_summary


def best_results_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Find the best throughput achieved for each model."""
    successful = df[df['success'] == True].copy()
    
    if len(successful) == 0:
        print("No successful runs found")
        return pd.DataFrame()
    
    # Find best run for each model
    best_runs = successful.loc[successful.groupby('model_name')['final_throughput'].idxmax()]
    
    best_summary = best_runs[['model_name', 'domain', 'optimizer', 'temperature', 'final_throughput', 
                               'total_partitions', 'final_DSP', 'final_BRAM', 'final_LUT', 'final_FF']].copy()
    best_summary = best_summary.sort_values('final_throughput', ascending=False)
    
    return best_summary


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_csv>")
        print("Example: python analyze_results.py additions/results/all_results.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Load results
    df = load_results(csv_path)
    
    # Create output directory
    output_dir = Path(csv_path).parent / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating analysis tables...\n")
    
    # Generate all summary tables
    tables = {
        "optimizer_summary": summary_by_optimizer(df),
        "model_summary": summary_by_model(df),
        "temperature_summary": summary_by_temperature(df),
        "optimizer_comparison": optimizer_comparison_by_model(df),
        "failure_analysis": failure_analysis(df),
        "resource_utilization": resource_utilization_summary(df),
        "best_results": best_results_per_model(df),
    }
    
    # Print tables to console
    print("=" * 80)
    print("OPTIMIZER SUMMARY")
    print("=" * 80)
    print(tables["optimizer_summary"].to_string())
    print()
    
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(tables["model_summary"].to_string())
    print()
    
    if not tables["temperature_summary"].empty:
        print("=" * 80)
        print("SIMULATED ANNEALING - TEMPERATURE COMPARISON")
        print("=" * 80)
        print(tables["temperature_summary"].to_string())
        print()
    
    print("=" * 80)
    print("OPTIMIZER COMPARISON BY MODEL")
    print("=" * 80)
    print(tables["optimizer_comparison"].to_string())
    print()
    
    if not tables["failure_analysis"].empty:
        print("=" * 80)
        print("FAILURE ANALYSIS")
        print("=" * 80)
        print(tables["failure_analysis"].to_string())
        print()
    
    print("=" * 80)
    print("RESOURCE UTILIZATION (Successful Runs Only)")
    print("=" * 80)
    print(tables["resource_utilization"].to_string())
    print()
    
    print("=" * 80)
    print("BEST RESULTS PER MODEL")
    print("=" * 80)
    print(tables["best_results"].to_string())
    print()
    
    # Save all tables to separate CSV files for Excel import
    for name, table in tables.items():
        if not table.empty:
            output_path = output_dir / f"{name}.csv"
            table.to_csv(output_path)
            print(f"Saved {name} to {output_path}")
    
    # Save a combined Excel file with multiple sheets (if openpyxl available)
    if HAS_EXCEL:
        excel_path = output_dir / "benchmark_analysis.xlsx"
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for name, table in tables.items():
                    if not table.empty:
                        table.to_excel(writer, sheet_name=name[:31])  # Excel sheet name limit is 31 chars
            print(f"\nSaved combined Excel workbook to {excel_path}")
        except Exception as e:
            print(f"\nWarning: Could not create Excel file: {e}")
            print("CSV files are still available for manual Excel import.")
    else:
        print(f"\nSkipped Excel export (openpyxl not installed)")
        print("CSV files saved - you can import them manually into Excel.")
    
    print(f"\nAnalysis complete! All tables saved to {output_dir}")


if __name__ == "__main__":
    main()
