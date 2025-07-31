"""
BMR (Best-Mean-Random) Algorithm implementation for solar module parameter optimization
"""
import numpy as np
from typing import Tuple, Dict
from .base_optimizer import BaseOptimizer


class BMROptimizer(BaseOptimizer):
    """
    BMR (Best-Mean-Random) Algorithm - A nature-inspired optimization algorithm
    
    The algorithm uses the best, mean, and randomly selected solutions to guide
    the search process with both exploitation and exploration capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Run the BMR optimization algorithm
        
        Returns:
            best_solution: Best solution found
            best_fitness: Best fitness value
        """
        # Initialize population
        population = self.initialize_population()
        fitness = self.evaluate_population(population)
        
        # Find initial best solution
        best_idx = np.argmin(fitness)
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
            
            # Find best solution and compute mean solution
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            mean_solution = np.mean(population, axis=0)
            
            for i in range(self.population_size):
                # Generate random numbers for the entire solution
                r4 = np.random.rand()
                
                if r4 > 0.5:
                    # BMR update equation (Eq. 1)
                    r1 = np.random.rand(self.dimension)
                    r2 = np.random.rand(self.dimension)
                    
                    # T factor randomly takes 1 or 2
                    T = np.random.choice([1, 2])
                    
                    # Select random solution
                    random_idx = np.random.randint(0, self.population_size)
                    random_solution = population[random_idx]
                    
                    new_solution = (population[i] + 
                                  r1 * (best_solution - T * mean_solution) +
                                  r2 * (best_solution - random_solution))
                else:
                    # Random reinitialization (Eq. 2)
                    # R = Uj - (Uj - Lj) * r3
                    r3 = np.random.rand(self.dimension)
                    new_solution = self.upper_bounds - (self.upper_bounds - self.lower_bounds) * r3
                
                # Apply bounds to the entire solution
                new_population[i] = self.apply_bounds(new_solution)
            
            # Evaluate new population
            new_fitness = self.evaluate_population(new_population)
            
            # Greedy selection - keep better solutions
            for i in range(self.population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
            
            # Update global best if improved
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_solution = population[current_best_idx].copy()
                self.best_fitness = fitness[current_best_idx]
            
            # Store convergence data
            self.convergence_history.append({
                'iteration': iteration + 1,
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean(fitness),
                'nfes': self.function_evaluations
            })
            
            # Optional: Early stopping if fitness is very small
            if self.best_fitness < 1e-14:
                break
        
        return self.best_solution, self.best_fitness
    
    def get_algorithm_info(self) -> Dict:
        """Get algorithm information"""
        return {
            'name': 'BMR (Best-Mean-Random)',
            'parameters': {
                'population_size': self.population_size,
                'max_iterations': self.max_iterations,
                'T_factor': 'Randomly 1 or 2',
                'random_threshold': 0.2
            },
            'description': 'Uses best, mean, and random solutions for population update',
            'update_mechanism': 'Exploitation via best-mean guidance, exploration via randomization'
        }