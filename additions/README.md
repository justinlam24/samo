# SAMO Benchmark Suite

This directory contains scripts for comprehensive FPGA optimizer benchmarking.

## Files

### Core Scripts

**`test_model_set.py`** - Main benchmark orchestration
- Runs 12 experiments per model (3 Rule-Based + 9 Simulated Annealing)
- Tests 10 fpgaconvnet-compatible CNN models across 5 domains
- Outputs detailed CSV logs for analysis

**`instrumented_annealing.py`** - SA optimizer with detailed tracking
- Wrapper around SAMO's SimulatedAnnealing
- Logs per-iteration metrics, move statistics, resource violations
- Generates per-run CSV files for SA experiments

**`convert_models_to_onnx.py`** - Model conversion utility (experimental)
- Attempts to convert modern CNN models from PyTorch to ONNX format
- Note: Generated models may not be compatible with fpgaconvnet parser
- Not required for running the benchmark (which uses existing SAMO models)

## Usage

### Run Benchmark
```bash
python additions/test_model_set.py --platform platforms/u250_1slr.json
```

### Results Location
- **`additions/results/all_results.csv`** - Summary of all experiments
- **`additions/results/{model}_SA_T{temp}_{trial}_log/`** - Per-run detailed CSVs (SA only)
  - `cost_vs_iteration.csv` - Throughput/cost over iterations
  - `partition_and_folding.csv` - Move statistics
  - `resource_violation.csv` - Constraint violations

## Output Format

### all_results.csv columns:
- **Identification**: `model_name`, `domain`, `trial`, `temperature`, `optimizer`, `seed`
- **Performance**: `runtime_seconds`, `final_throughput`, `success`, `initial_feasible`, `feasible`
- **Partitioning**: `total_partitions`, `avg_partition_size`, `max_partition_resource_util`
- **Move Statistics** (SA only): `total_folding_moves`, `total_partition_moves`, `partition_moves_accepted`, `folding_moves_accepted`, `best_iteration_index`
- **Resources**: `final_DSP`, `final_BRAM`, `final_LUT`, `final_FF`
- **Error Info**: `error` (empty if successful)

**Key Feasibility Columns:**
- `initial_feasible` (True/False) - Was the network configuration feasible BEFORE optimization started?
- `feasible` (True/False) - Was the network configuration feasible AFTER optimization completed?

This allows analysis of optimizer recovery: networks that start infeasible but end feasible show the optimizer successfully found a valid solution.

**Note**: For Rule-Based optimizer, move statistics columns will contain "NA".

## Model Set

10 fpgaconvnet-compatible sequential CNN models across 5 domains:
1. **Image Classification**: DeepNarrow (VGG-16 style), CompactNet (edge deployment)
2. **Object Detection**: Detection Backbone, MobileNet-style (depthwise separable)
3. **Semantic Segmentation**: FCN Encoder, Dense Segmentation
4. **Super-Resolution**: SRCNN (classic), VDSR (deep)
5. **Medical Imaging**: Grayscale Encoder, Multi-scale (dilated)

**Note**: fpgaconvnet only supports sequential architectures without skip connections. Modern architectures like ResNet, U-Net, and EfficientNet use skip connections and are not supported.

## Experiment Configuration

- **Simulated Annealing**: 3 temperatures (10, 20, 50) × 3 trials = 9 runs
- **Rule-Based**: 3 trials = 3 runs
- **Total per model**: 12 runs
- **Total experiments**: 120 runs (10 models × 12 runs each)

## Progress Logging

The benchmark provides detailed real-time progress:

```
[MODEL 1/12] Processing resnet50...
================================================================================
Model: RESNET50 - Image Classification - ResNet-50
================================================================================

📊 Starting 12 runs for resnet50...
   └─ 3 Rule-Based trials + 9 SA trials (3 temps × 3 trials)

  [ 1/12] Rule-Based (trial 1, seed=43) - ✓ (5.2s, TP=0.000234)
  [ 2/12] Rule-Based (trial 2, seed=44) - ✓ (4.8s, TP=0.000229)
  [ 3/12] Rule-Based (trial 3, seed=45) - ✓ (5.1s, TP=0.000231)
  [ 4/12] SA T=10.0 (trial 1, seed=73) - simulated annealing iterations (T=0.012):  89%|████▉| 1247/1400
  [ 4/12] SA T=10.0 (trial 1, seed=73) - ✓ (92.3s, TP=0.000456)
  ...
```

**Shows:**
- Model progress (MODEL 1/12)
- Run progress within each model ([4/12])
- Optimizer type, temperature, trial, and seed
- Live SA iteration progress with tqdm bar
- Results: ✓ success with runtime & throughput, or ✗ failure

**Estimated Runtime**: 4-8 hours for all 144 experiments (varies by model complexity)
