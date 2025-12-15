import copy
import csv
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from samo.model import Network


@dataclass
class Individual:
    """Represents a single solution in the population."""
    network: Network
    fitness: float = float('inf')  # Lower is better (for latency objective)
    
    def evaluate(self):
        """Evaluate fitness of this individual."""
        if self.network.check_constraints():
            self.fitness = self.network.eval_cost()
        else:
            # Penalize infeasible solutions heavily
            self.fitness = float('inf')
        return self.fitness


@dataclass
class GeneticAlgorithm:
    network: Network
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elitism_count: int = 5
    tournament_size: int = 3
    adaptive_mutation: bool = True
    valid_variables: list = field(default_factory=lambda: ["channel_in_folding", "channel_out_folding", "kernel_folding"])
    
    def __post_init__(self):
        """Initialize population and tracking variables."""
        self.population: List[Individual] = []
        self.best_individual: Individual = None
        self.generation_counter = 0
        self.stagnation_counter = 0
        self.best_fitness_history = []
        
    def encode_individual(self, network: Network) -> dict:
        """
        Encode a network configuration as a gene representation.
        
        Returns a dictionary mapping layer names to their folding parameters.
        """
        genes = {}
        for partition_idx, partition in enumerate(network.partitions):
            for layer in partition.nodes():
                node_hw = partition.nodes[layer]["hw"]
                genes[f"{partition_idx}_{layer}"] = {
                    "channel_in_folding": node_hw.channel_in_folding,
                    "channel_out_folding": node_hw.channel_out_folding,
                    "kernel_folding": node_hw.kernel_folding,
                }
        return genes
    
    def decode_individual(self, genes: dict, base_network: Network) -> Network:
        """
        Decode gene representation back into a network configuration.
        """
        network = copy.deepcopy(base_network)
        
        for partition_idx, partition in enumerate(network.partitions):
            for layer in partition.nodes():
                key = f"{partition_idx}_{layer}"
                if key in genes:
                    node_hw = partition.nodes[layer]["hw"]
                    node_hw.channel_in_folding = genes[key]["channel_in_folding"]
                    node_hw.channel_out_folding = genes[key]["channel_out_folding"]
                    node_hw.kernel_folding = genes[key]["kernel_folding"]
                    node_hw.update(hw_update=True)
        
        return network
    
    def initialize_population(self):
        """
        Create initial population with diverse configurations.
        Uses a mix of random and heuristic-based initialization.
        """
        self.population = []
        
        # Add the initial configuration
        initial_individual = Individual(network=copy.deepcopy(self.network))
        initial_individual.evaluate()
        self.population.append(initial_individual)
        
        # Generate diverse random individuals
        for _ in range(self.population_size - 1):
            network_copy = copy.deepcopy(self.network)
            
            # Apply random transformations
            num_transformations = random.randint(5, 20)
            for _ in range(num_transformations):
                self.apply_random_mutation(network_copy)
            
            # Update all hardware parameters
            self.update_network(network_copy)
            
            individual = Individual(network=network_copy)
            individual.evaluate()
            self.population.append(individual)
        
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness)
        self.best_individual = copy.deepcopy(self.population[0])
        
    def update_network(self, network: Network):
        """Update all hardware parameters in the network."""
        for partition in network.partitions:
            for layer in partition:
                partition.nodes[layer]["hw"].update(hw_update=True)
    
    def apply_random_mutation(self, network: Network):
        """
        Apply a single random mutation to the network.
        Similar to simulated annealing's random_transformation but more focused.
        """
        # Choose partition or variable mutation (80% variable, 20% partition)
        if random.random() < 0.8 and len(network.partitions) > 0:
            # Variable mutation
            partition = random.choice(network.partitions)
            if len(list(partition.nodes())) > 0:
                layer = random.choice(list(partition.nodes()))
                node_hw = partition.nodes[layer]["hw"]
                variable = random.choice(self.valid_variables)
                
                if variable == "channel_in_folding":
                    folding = random.choice(node_hw.valid_channel_in_folding)
                    node_hw.channel_in_folding = folding
                    partition.folding_match(layer, folding, "io")
                elif variable == "channel_out_folding":
                    folding = random.choice(node_hw.valid_channel_out_folding)
                    node_hw.channel_out_folding = folding
                    partition.folding_match(layer, folding, "io")
                elif variable == "kernel_folding":
                    node_hw.kernel_folding = random.choice(node_hw.valid_kernel_folding)
        else:
            # Partition mutation (split/merge)
            if len(network.partitions) > 0:
                partition_index = random.randint(0, len(network.partitions) - 1)
                
                if random.random() < 0.5:
                    # Try split
                    valid_splits = network.valid_splits(partition_index)
                    if valid_splits:
                        nodes = random.choice(valid_splits)
                        network.split(partition_index, nodes)
                else:
                    # Try merge
                    valid_merges = network.valid_merges()
                    if valid_merges:
                        merge = random.choice(valid_merges)
                        network.merge(merge)
    
    def tournament_selection(self) -> Individual:
        """
        Select an individual using tournament selection.
        """
        tournament = random.sample(self.population, self.tournament_size)
        tournament.sort(key=lambda x: x.fitness)
        return tournament[0]
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform uniform crossover between two parents.
        Each gene has 50% chance of coming from either parent.
        """
        # Encode parents
        genes1 = self.encode_individual(parent1.network)
        genes2 = self.encode_individual(parent2.network)
        
        # Create offspring genes
        offspring1_genes = {}
        offspring2_genes = {}
        
        # Uniform crossover
        for key in genes1.keys():
            if key in genes2 and random.random() < 0.5:
                offspring1_genes[key] = copy.deepcopy(genes2[key])
                offspring2_genes[key] = copy.deepcopy(genes1[key])
            else:
                offspring1_genes[key] = copy.deepcopy(genes1[key])
                if key in genes2:
                    offspring2_genes[key] = copy.deepcopy(genes2[key])
        
        # Decode offspring
        offspring1_network = self.decode_individual(offspring1_genes, parent1.network)
        offspring2_network = self.decode_individual(offspring2_genes, parent2.network)
        
        return (Individual(network=offspring1_network), 
                Individual(network=offspring2_network))
    
    def mutate(self, individual: Individual):
        """
        Apply mutation to an individual.
        Mutation rate adapts based on population diversity.
        """
        # Adaptive mutation rate
        if self.adaptive_mutation and self.stagnation_counter > 10:
            mutation_rate = min(0.8, self.mutation_rate * 1.5)
        else:
            mutation_rate = self.mutation_rate
        
        # Apply multiple mutations with probability
        num_mutations = np.random.poisson(mutation_rate * 5)  # Average 1-2 mutations
        
        for _ in range(num_mutations):
            self.apply_random_mutation(individual.network)
        
        # Update hardware
        self.update_network(individual.network)
    
    def evolve_generation(self):
        """
        Evolve the population for one generation using genetic operators.
        """
        new_population = []
        
        # Elitism: Keep best individuals
        self.population.sort(key=lambda x: x.fitness)
        for i in range(self.elitism_count):
            new_population.append(copy.deepcopy(self.population[i]))
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                offspring1, offspring2 = self.crossover(parent1, parent2)
            else:
                offspring1 = Individual(network=copy.deepcopy(parent1.network))
                offspring2 = Individual(network=copy.deepcopy(parent2.network))
            
            # Mutation
            self.mutate(offspring1)
            self.mutate(offspring2)
            
            # Evaluate
            offspring1.evaluate()
            offspring2.evaluate()
            
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        self.population = new_population
        
        # Update best individual
        self.population.sort(key=lambda x: x.fitness)
        if self.population[0].fitness < self.best_individual.fitness:
            self.best_individual = copy.deepcopy(self.population[0])
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        self.best_fitness_history.append(self.best_individual.fitness)
    
    def optimise(self):
        """
        Main optimization loop.
        """
        log = []
        
        print("Initializing population...")
        self.initialize_population()
        
        print(f"Initial best fitness: {self.best_individual.fitness:.4f}")
        log.append([time.time() - self.start_time, self.best_individual.fitness])
        
        # Evolution loop
        pbar = tqdm(range(self.generations), desc="Genetic Algorithm Generations")
        for generation in pbar:
            self.generation_counter = generation
            
            # Evolve population
            self.evolve_generation()
            
            # Update progress bar
            avg_fitness = np.mean([ind.fitness for ind in self.population if ind.fitness != float('inf')])
            pbar.set_description(
                f"Gen {generation+1}/{self.generations} | "
                f"Best: {self.best_individual.fitness:.4f} | "
                f"Avg: {avg_fitness:.4f} | "
                f"Stagnation: {self.stagnation_counter}"
            )
            
            # Log every generation
            log.append([time.time() - self.start_time, self.best_individual.fitness])
            
            # Early stopping if converged
            if self.stagnation_counter > 30:
                print(f"\nEarly stopping: No improvement for {self.stagnation_counter} generations")
                break
        
        # Update the main network with best solution
        self.network = copy.deepcopy(self.best_individual.network)
        
        # Write log to file
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
        with open("outputs/log.csv", "w", newline='') as f:
            writer = csv.writer(f)
            [writer.writerow(row) for row in log]
        
        print(f"\nOptimization complete!")
        print(f"Best fitness: {self.best_individual.fitness:.4f}")
        print(f"Total generations: {generation+1}")
