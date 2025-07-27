"""
Base optimizer class for solar module parameter optimization
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple, List, Dict


class BaseOptimizer(ABC):
    """Abstract base class for optimization algorithms"""
    
    def __init__(self, 
                 objective_func: Callable,
                 bounds: Dict[str, List[float]],
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 seed: int = None):
        """
        Initialize the optimizer
        
        Args:
            objective_func: Objective function to minimize
            bounds: Dictionary with parameter bounds
            population_size: Number of individuals in population
            max_iterations: Maximum number of iterations
            seed: Random seed for reproducibility
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.dimension = len(bounds)
        
        # Extract bounds arrays
        self.lower_bounds = np.array([bounds[key][0] for key in ['a', 'Rs', 'Rp']])
        self.upper_bounds = np.array([bounds[key][1] for key in ['a', 'Rs', 'Rp']])
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize tracking variables
        self.best_solution = None
        self.best_fitness = np.inf
        self.convergence_history = []
        self.function_evaluations = 0
        
    def initialize_population(self) -> np.ndarray:
        """Initialize population within bounds"""
        population = np.random.rand(self.population_size, self.dimension)
        population = self.lower_bounds + population * (self.upper_bounds - self.lower_bounds)
        return population
    
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness for entire population"""
        fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitness[i] = self.objective_func(population[i])
            self.function_evaluations += 1
        return fitness
    
    def apply_bounds(self, solution: np.ndarray) -> np.ndarray:
        """Apply boundary constraints to solution"""
        return np.clip(solution, self.lower_bounds, self.upper_bounds)
    
    @abstractmethod
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run the optimization algorithm
        
        Returns:
            best_solution: Best solution found
            best_fitness: Best fitness value
        """
        pass
    
    def get_results(self) -> Dict:
        """Get optimization results"""
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'convergence_history': self.convergence_history,
            'function_evaluations': self.function_evaluations,
            'parameters': {
                'a': self.best_solution[0],
                'Rs': self.best_solution[1],
                'Rp': self.best_solution[2]
            }
        }