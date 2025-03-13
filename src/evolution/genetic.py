import logging
import random
import numpy as np
from typing import List, Callable, Optional, Dict, Any
from collections import defaultdict

from src.evolution.model_dna import ModelDNA


class GeneticSelector:
    """
    Implements selection strategies for the genetic algorithm.
    """
    @staticmethod
    def tournament_selection(population: List[ModelDNA], fitness_fn: Callable[[ModelDNA], float],
                             tournament_size: int = 3) -> ModelDNA:
        """
        Tournament selection: randomly selects tournament_size individuals from the population
        and returns the fittest one.
        """
        if not population:
            raise ValueError("Population cannot be empty")

        # Randomly select individuals for the tournament
        tournament = random.sample(population, min(tournament_size, len(population)))

        # Find the fittest individual in the tournament
        return max(tournament, key=fitness_fn)

    @staticmethod
    def roulette_wheel_selection(population: List[ModelDNA], fitness_fn: Callable[[ModelDNA], float]) -> ModelDNA:
        """
        Roulette wheel selection: selects individuals with probability proportional to their fitness.
        """
        if not population:
            raise ValueError("Population cannot be empty")

        # Calculate fitness values for all individuals
        fitness_values = [max(0.0001, fitness_fn(dna)) for dna in population]
        total_fitness = sum(fitness_values)

        # Normalize fitness values to get probabilities
        probabilities = [fitness / total_fitness for fitness in fitness_values]

        # Select an individual based on probabilities
        selected_index = np.random.choice(len(population), p=probabilities)
        return population[selected_index]

    @staticmethod
    def rank_selection(population: List[ModelDNA], fitness_fn: Callable[[ModelDNA], float]) -> ModelDNA:
        """
        Rank selection: sorts the population by fitness and selects individuals with
        probability proportional to their rank.
        """
        if not population:
            raise ValueError("Population cannot be empty")

        # Sort population by fitness (ascending)
        sorted_population = sorted(population, key=fitness_fn)

        # Calculate ranks (starting from 1)
        ranks = list(range(1, len(sorted_population) + 1))
        total_rank = sum(ranks)

        # Normalize ranks to get probabilities
        probabilities = [rank / total_rank for rank in ranks]

        # Select an individual based on probabilities
        selected_index = np.random.choice(len(sorted_population), p=probabilities)
        return sorted_population[selected_index]


class GeneticAlgorithm:
    """
    Implements a genetic algorithm for evolving AI models based on their DNA.
    """

    def __init__(self,
                 population_size: int = 20,
                 mutation_rate: float = 0.1,
                 mutation_strength: float = 0.2,
                 crossover_rate: float = 0.7,
                 selection_strategy: str = "tournament",
                 tournament_size: int = 3,
                 elitism_count: int = 2):
        """
        Initialize the genetic algorithm with the given parameters.

        Args:
            population_size: Number of individuals in the population
            mutation_rate: Probability of mutation for each gene
            mutation_strength: Strength of mutation (how much a gene can change)
            crossover_rate: Probability of crossover between two parents
            selection_strategy: Strategy for selecting parents ("tournament", "roulette", "rank")
            tournament_size: Size of tournament for tournament selection
            elitism_count: Number of top individuals to preserve in each generation
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.selection_strategy = selection_strategy
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.logger = logging.getLogger(__name__)

        # Initialize population tracking
        self.current_generation = 0
        self.current_population: List[ModelDNA] = []
        self.fitness_history: defaultdict[str, List[float]] = defaultdict(list)
        self.evolution_metrics: List[Dict[str, Any]] = []

    def initialize_population(self, initial_models: Optional[List[ModelDNA]] = None) -> List[ModelDNA]:
        """
        Initialize the population with either provided models or randomly generated ones.

        Args:
            initial_models: Optional list of initial models to seed the population

        Returns:
            List of ModelDNA instances representing the initial population
        """
        population = []

        # Add provided models if any
        if initial_models:
            population.extend(initial_models)

        # Generate random models to fill the population
        while len(population) < self.population_size:
            # Random values for basic genes
            gene_values = {
                "precision": random.random(),
                "recall": random.random()
            }
            population.append(ModelDNA(None, **gene_values))
        
        self.current_population = population
        return population
