# Genetic Algorithm Optimizer for SAMO

This directory contains the implementation of a Genetic Algorithm (GA) optimizer for SAMO, designed to outperform the existing Simulated Annealing optimizer while maintaining reasonable execution time.

## Quick Start

```bash
# Run genetic algorithm on TFC model
python -m samo --model models/tfc.onnx \
    --backend fpgaconvnet \
    --platform platforms/zedboard.json \
    --output-path outputs/ \
    --optimiser genetic \
    --objective latency

# Compare all optimizers
python compare_optimizers.py \
    --model models/tfc.onnx \
    --platform platforms/zedboard.json \
    --optimizers rule genetic annealing
```

## Why Genetic Algorithm?

### Problems with Simulated Annealing
1. **Serial search**: Explores only one solution at a time
2. **Slow convergence**: 108K iterations in 25 minutes for TFC model
3. **No diversity**: Single trajectory through search space
4. **Wasteful evaluations**: Applies 100 transformations before checking feasibility

### Genetic Algorithm Advantages
1. **Population-based**: Explores 50 solutions in parallel
2. **Fast convergence**: Typically 50-100 generations = 2.5K-5K evaluations (20-40× fewer)
3. **Crossover operator**: Combines good solutions to create better offspring
4. **Diversity maintenance**: Multiple search trajectories
5. **Elitism**: Never loses best solutions
6. **Adaptive mutation**: Increases exploration when stuck

## Algorithm Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  GENETIC ALGORITHM FLOW                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. INITIALIZATION (Generation 0)                          │
│     ├─ Create initial population (50 individuals)          │
│     ├─ Add baseline configuration                          │
│     ├─ Generate 49 random diverse configurations           │
│     └─ Evaluate fitness (check constraints + cost)         │
│                                                             │
│  2. EVOLUTION LOOP (Generations 1-100)                     │
│     │                                                       │
│     ├─ ELITISM                                             │
│     │  └─ Copy 5 best individuals unchanged                │
│     │                                                       │
│     ├─ SELECTION (Tournament)                              │
│     │  ├─ Pick 3 random individuals                        │
│     │  └─ Select best one as parent                        │
│     │                                                       │
│     ├─ CROSSOVER (70% probability)                         │
│     │  ├─ Uniform crossover between parents                │
│     │  └─ Each gene has 50% from each parent               │
│     │                                                       │
│     ├─ MUTATION (30% rate, adaptive)                       │
│     │  ├─ Random folding parameter changes                 │
│     │  └─ Partition split/merge operations                 │
│     │                                                       │
│     ├─ EVALUATION                                          │
│     │  ├─ Check resource constraints                       │
│     │  └─ Calculate fitness (latency/throughput)           │
│     │                                                       │
│     └─ ADAPTATION                                          │
│        ├─ Track stagnation (no improvement)                │
│        └─ Increase mutation if stagnant > 10 gens          │
│                                                             │
│  3. TERMINATION                                            │
│     └─ Stop if: max generations OR stagnant > 30 gens      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Architecture

### Class Structure

```python
@dataclass
class Individual:
    """Single solution in population"""
    network: Network
    fitness: float = inf
    
    def evaluate(self):
        """Check constraints and calculate cost"""

@dataclass
class GeneticAlgorithm:
    """Main optimizer class"""
    network: Network
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elitism_count: int = 5
    
    def initialize_population(self):
        """Create diverse initial population"""
    
    def tournament_selection(self) -> Individual:
        """Select parent via tournament"""
    
    def crossover(self, p1, p2) -> (Individual, Individual):
        """Uniform crossover operator"""
    
    def mutate(self, individual):
        """Apply random mutations"""
    
    def evolve_generation(self):
        """Execute one generation of evolution"""
    
    def optimise(self):
        """Main optimization loop"""
```

### Gene Encoding

Each individual's genotype is encoded as:

```python
genes = {
    "partition0_layer1": {
        "channel_in_folding": 4,
        "channel_out_folding": 8,
        "kernel_folding": 1
    },
    "partition0_layer2": { ... },
    # ... more layers
}
```

## Key Operators

### 1. Tournament Selection

Selects parents probabilistically based on fitness:

```python
def tournament_selection(population, tournament_size=3):
    # Pick 3 random individuals
    contestants = random.sample(population, 3)
    
    # Return the best (lowest fitness for minimization)
    return min(contestants, key=lambda x: x.fitness)
```

**Why tournament?**
- Simple and effective
- Adjustable selection pressure
- No need for fitness scaling
- Works with negative fitness values

### 2. Uniform Crossover

Combines two parents by randomly selecting genes:

```python
def crossover(parent1, parent2):
    offspring1_genes = {}
    offspring2_genes = {}
    
    for gene_name in parent1.genes:
        if random() < 0.5:
            # Swap genes between offspring
            offspring1_genes[gene_name] = parent2.genes[gene_name]
            offspring2_genes[gene_name] = parent1.genes[gene_name]
        else:
            # Keep original genes
            offspring1_genes[gene_name] = parent1.genes[gene_name]
            offspring2_genes[gene_name] = parent2.genes[gene_name]
    
    return decode(offspring1_genes), decode(offspring2_genes)
```

**Why uniform crossover?**
- No bias toward any position
- Good for independent parameters
- Maintains diversity
- Simple to implement

### 3. Adaptive Mutation

Mutation rate increases when population stagnates:

```python
def mutate(individual, base_rate=0.3, stagnation_count=0):
    # Increase mutation when stuck
    if stagnation_count > 10:
        rate = min(0.8, base_rate * 1.5)
    else:
        rate = base_rate
    
    # Number of mutations ~ Poisson distribution
    num_mutations = poisson(rate * 5)
    
    for _ in range(num_mutations):
        # Randomly change folding parameters or partitions
        apply_random_transformation(individual)
```

**Why adaptive?**
- Balances exploration vs exploitation
- Escapes local optima automatically
- No manual parameter tuning needed

## Hyperparameters

### Default Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `population_size` | 50 | Number of individuals in population |
| `generations` | 100 | Maximum number of generations |
| `mutation_rate` | 0.3 | Base probability of mutation |
| `crossover_rate` | 0.7 | Probability of crossover |
| `elitism_count` | 5 | Number of best individuals preserved |
| `tournament_size` | 3 | Number of contestants in tournament |
| `adaptive_mutation` | True | Enable adaptive mutation rate |

### Tuning Guidelines

**For small networks (< 5 layers):**
```python
GeneticAlgorithm(
    population_size=20,   # Smaller space
    generations=30,       # Converges faster
    mutation_rate=0.4,    # More exploration
)
```

**For large networks (> 10 layers):**
```python
GeneticAlgorithm(
    population_size=100,  # More diversity needed
    generations=200,      # More time to converge
    mutation_rate=0.2,    # Gentle mutations
    elitism_count=10,     # Preserve more solutions
)
```

**For tight resource constraints:**
```python
GeneticAlgorithm(
    population_size=30,
    generations=100,
    mutation_rate=0.5,    # Aggressive search
    adaptive_mutation=True,
)
```

## Expected Performance

### Convergence Comparison

| Metric | Simulated Annealing | Genetic Algorithm |
|--------|---------------------|-------------------|
| Evaluations | ~108,000 | ~2,500-5,000 |
| Time (TFC on Zedboard) | 25 minutes | 2-5 minutes |
| Solution Quality | 40.96 μs | 35-38 μs (expected) |
| Speedup | 1× (baseline) | 5-10× |
| Improvement | 0% (baseline) | 5-15% better |

### Typical Evolution Pattern

```
Generation    Best Fitness    Avg Fitness    Stagnation
---------------------------------------------------------
0             165.2 μs        245.3 μs       0
10            52.8 μs         89.4 μs        0
20            43.2 μs         61.7 μs        0
30            38.4 μs         52.1 μs        3
40            37.1 μs         48.3 μs        7
50            36.8 μs         45.6 μs        12  <- Adaptive mutation kicks in
60            35.9 μs         43.2 μs        0   <- Improvement found
70            35.7 μs         41.8 μs        2
80            35.6 μs         40.9 μs        5
90            35.5 μs         40.2 μs        8
100           35.4 μs         39.7 μs        12
```

## Running Experiments

### Basic Usage

```bash
# Run with default parameters
python -m samo --model models/lenet.onnx \
    --backend fpgaconvnet \
    --platform platforms/zedboard.json \
    --output-path outputs_genetic/ \
    --optimiser genetic \
    --objective latency
```

### Benchmark Against Other Optimizers

```bash
# Compare all three optimizers
python compare_optimizers.py \
    --model models/tfc.onnx \
    --platform platforms/zedboard.json \
    --optimizers rule genetic annealing \
    --output comparison_tfc.json

# View results
cat comparison_tfc.json
```

### Hyperparameter Sweep

```bash
# Test different population sizes
for pop in 20 50 100; do
    echo "Testing population_size=$pop"
    # Modify genetic.py temporarily or create config
    python -m samo --model models/cnv.onnx \
        --backend fpgaconvnet \
        --platform platforms/zedboard.json \
        --output-path "outputs_pop${pop}/" \
        --optimiser genetic
done
```

## Output Files

### Log File (`outputs/log.csv`)

Tracks optimization progress over time:

```csv
time_elapsed,best_fitness
0.0,165.2
0.5,165.2
1.2,52.8
1.8,52.8
2.3,43.2
...
```

Visualize with:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/log.csv', names=['time', 'fitness'])
plt.plot(df['time'], df['fitness'])
plt.xlabel('Time (s)')
plt.ylabel('Latency (μs)')
plt.title('Genetic Algorithm Convergence')
plt.show()
```

## Troubleshooting

### Issue: Population converges too quickly

**Symptoms:** All individuals have similar fitness after 10-20 generations

**Solution:**
```python
# Increase diversity
GeneticAlgorithm(
    population_size=100,     # More individuals
    mutation_rate=0.5,       # More mutations
    crossover_rate=0.9,      # More mixing
    tournament_size=2,       # Weaker selection pressure
)
```

### Issue: No improvement after many generations

**Symptoms:** Best fitness stuck for 30+ generations

**Solution:**
- Adaptive mutation should trigger automatically
- If not working, manually increase `mutation_rate`
- Try different random seeds: `--seed 42`
- Check if initial design is already optimal

### Issue: Out of memory errors

**Symptoms:** Python crashes during evolution

**Solution:**
```python
# Reduce population size
GeneticAlgorithm(
    population_size=20,  # Instead of 50
)
```

### Issue: All individuals infeasible

**Symptoms:** Fitness = inf for entire population

**Solution:**
- Check platform constraints are not too tight
- Increase initial population diversity
- Start with rule-based optimizer first
- Reduce `mutation_rate` to avoid breaking constraints

## Advanced Features

### Custom Fitness Function

Modify `Individual.evaluate()` to add custom objectives:

```python
def evaluate(self):
    if self.network.check_constraints():
        latency = self.network.eval_latency()
        resource = self.network.eval_resource()
        
        # Multi-objective: latency + resource utilization
        resource_penalty = sum(resource.values()) / 10000
        self.fitness = latency + resource_penalty
    else:
        self.fitness = float('inf')
    return self.fitness
```

### Island Model (Parallel GAs)

Run multiple populations in parallel:

```python
from multiprocessing import Pool

def run_island(seed):
    ga = GeneticAlgorithm(network, generations=50)
    random.seed(seed)
    ga.optimise()
    return ga.best_individual

# Run 4 islands
with Pool(4) as p:
    islands = p.map(run_island, [42, 123, 456, 789])

# Select best across all islands
best = min(islands, key=lambda x: x.fitness)
```

### Constraint Handling

Add repair operator for infeasible solutions:

```python
def repair(individual):
    """Fix constraint violations"""
    while not individual.network.check_constraints():
        # Reduce parallelism
        partition = random.choice(individual.network.partitions)
        layer = random.choice(list(partition.nodes()))
        node_hw = partition.nodes[layer]["hw"]
        
        # Reduce folding factors
        if node_hw.channel_in_folding > 1:
            node_hw.channel_in_folding //= 2
        # ... similar for other parameters
    
    individual.evaluate()
```

## Future Enhancements

### Planned Features
- [ ] Multi-objective optimization (Pareto front)
- [ ] Parallel fitness evaluation
- [ ] Coevolution of network topology
- [ ] Neural network-guided search
- [ ] Warm start from previous runs

### Research Directions
- Compare with other evolutionary algorithms (PSO, Differential Evolution)
- Investigate problem landscape (fitness landscape analysis)
- Learn good mutation strategies from data
- Hybrid with local search (memetic algorithm)

## References

1. **Original SAMO paper:**
   - Montgomerie et al., "Samo: Optimised mapping of convolutional neural networks to streaming architectures", FPT 2022

2. **Genetic algorithms:**
   - Goldberg, D.E., "Genetic Algorithms in Search, Optimization, and Machine Learning", 1989
   - Eiben & Smith, "Introduction to Evolutionary Computing", 2015

3. **FPGA optimization:**
   - Venieris & Bouganis, "fpgaConvNet: Mapping Regular and Irregular Convolutional Neural Networks on FPGAs", TNNLS 2019

## Contributing

To improve the genetic algorithm optimizer:

1. Fork the repository
2. Modify `samo/optimiser/genetic.py`
3. Test on multiple models:
   ```bash
   python compare_optimizers.py --model models/tfc.onnx
   python compare_optimizers.py --model models/cnv.onnx
   python compare_optimizers.py --model models/lenet.onnx
   ```
4. Submit pull request with benchmark results

## License

Same as SAMO framework (check main LICENSE file).

## Contact

For questions or issues with the genetic algorithm optimizer, please open an issue on GitHub.
