"""
JAYA Algorithm implementation for solar module parameter optimization
"""
import numpy as np
from typing import Tuple, Dict
from .base_optimizer import BaseOptimizer


class JAYAOptimizer(BaseOptimizer):
    """
    JAYA Algorithm - A simple and parameter-free optimization algorithm
    
    Reference: Rao, R. (2016). Jaya: A simple and new optimization algorithm 
    for solving constrained and unconstrained optimization problems. 
    International Journal of Industrial Engineering Computations, 7(1), 19-34.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run the JAYA optimization algorithm
        
        Returns:
            best_solution: Best solution found
            best_fitness: Best fitness value
        """
        # Initialize population
        population = self.initialize_population()
        fitness = self.evaluate_population(population)
        
        # Find initial best and worst solutions
        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        
        # Store initial convergence data
        self.convergence_history.append({
            'iteration': 0,
            'best_fitness': self.best_fitness,
            'avg_fitness': np.mean(fitness),
            'nfes': self.function_evaluations
        })
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Create new population
            new_population = np.zeros_like(population)
            
            for i in range(self.population_size):
                # JAYA update equation
                # X_new = X_old + rand * (X_best - |X_old|) - rand * (X_worst - |X_old|)
                r1 = np.random.rand(self.dimension)
                r2 = np.random.rand(self.dimension)
                
                new_solution = (population[i] + 
                               r1 * (population[best_idx] - np.abs(population[i])) - 
                               r2 * (population[worst_idx] - np.abs(population[i])))
                
                # Apply bounds
                new_population[i] = self.apply_bounds(new_solution)
            
            # Evaluate new population
            new_fitness = self.evaluate_population(new_population)
            
            # Greedy selection
            for i in range(self.population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
            
            # Update best and worst solutions
            best_idx = np.argmin(fitness)
            worst_idx = np.argmax(fitness)
            
            # Update global best if improved
            if fitness[best_idx] < self.best_fitness:
                self.best_solution = population[best_idx].copy()
                self.best_fitness = fitness[best_idx]
            
            # Store convergence data
            self.convergence_history.append({
                'iteration': iteration + 1,
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean(fitness),
                'nfes': self.function_evaluations
            })
            
            # Optional: Early stopping if fitness is very small
            if self.best_fitness < 1e-10:
                break
        
        return self.best_solution, self.best_fitness
    
    def get_algorithm_info(self) -> Dict:
        """Get algorithm information"""
        return {
            'name': 'JAYA',
            'parameters': 'Parameter-free algorithm',
            'reference': 'Rao (2016) - International Journal of Industrial Engineering Computations'
        }