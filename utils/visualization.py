"""
Visualization utilities for solar module optimization results
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple


class Visualizer:
    """Visualization tools for optimization results"""
    
    @staticmethod
    def plot_convergence(convergence_history: List[Dict], title: str = "Convergence History"):
        """
        Plot convergence history
        
        Args:
            convergence_history: List of convergence data
            title: Plot title
        """
        iterations = [d['iteration'] for d in convergence_history]
        best_fitness = [d['best_fitness'] for d in convergence_history]
        avg_fitness = [d['avg_fitness'] for d in convergence_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(iterations, avg_fitness, 'r--', label='Average Fitness', linewidth=1)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.show()
    
    @staticmethod
    def plot_3d_solutions(results: List[Dict], algorithms: List[str] = None):
        """
        Plot 3D scatter of solutions from multiple algorithms
        
        Args:
            results: List of result dictionaries
            algorithms: List of algorithm names
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['r', 'b', 'k', 'c', 'm', 'g', 'y', 'orange']
        markers = ['*', 's', 'x', '+', 'd', 'o', '^', 'v']
        
        if algorithms is None:
            algorithms = [f'Algorithm {i+1}' for i in range(len(results))]
        
        for i, (result, alg) in enumerate(zip(results, algorithms)):
            if isinstance(result, list):
                # Multiple runs
                a_vals = [r['parameters']['a'] for r in result]
                Rs_vals = [r['parameters']['Rs'] for r in result]
                Rp_vals = [r['parameters']['Rp'] for r in result]
            else:
                # Single run
                a_vals = [result['parameters']['a']]
                Rs_vals = [result['parameters']['Rs']]
                Rp_vals = [result['parameters']['Rp']]
            
            ax.scatter(a_vals, Rs_vals, Rp_vals, 
                      c=colors[i % len(colors)],
                      marker=markers[i % len(markers)],
                      s=100, label=alg)
        
        ax.set_xlabel('a (Ideality Factor)')
        ax.set_ylabel('Rs (Series Resistance) [Ω]')
        ax.set_zlabel('Rp (Parallel Resistance) [Ω]')
        ax.legend()
        plt.title('3D Distribution of Optimized Parameters')
        plt.show()
    
    @staticmethod
    def plot_iv_curves(solar_model, solutions: List[np.ndarray], labels: List[str] = None):
        """
        Plot I-V curves for different solutions
        
        Args:
            solar_model: SolarModuleModel instance
            solutions: List of parameter arrays
            labels: List of labels for each solution
        """
        plt.figure(figsize=(10, 6))
        
        if labels is None:
            labels = [f'Solution {i+1}' for i in range(len(solutions))]
        
        for sol, label in zip(solutions, labels):
            V, I = solar_model.calculate_iv_curve(sol)
            plt.plot(V, I, label=label, linewidth=2)
        
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.title(f'I-V Characteristics - {solar_model.get_module_info()["type"]}')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_pv_curves(solar_model, solutions: List[np.ndarray], labels: List[str] = None):
        """
        Plot P-V curves for different solutions
        
        Args:
            solar_model: SolarModuleModel instance
            solutions: List of parameter arrays
            labels: List of labels for each solution
        """
        plt.figure(figsize=(10, 6))
        
        if labels is None:
            labels = [f'Solution {i+1}' for i in range(len(solutions))]
        
        for sol, label in zip(solutions, labels):
            V, P = solar_model.calculate_power_curve(sol)
            plt.plot(V, P, label=label, linewidth=2)
        
        plt.xlabel('Voltage (V)')
        plt.ylabel('Power (W)')
        plt.title(f'P-V Characteristics - {solar_model.get_module_info()["type"]}')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_statistics(all_results: List[List[Dict]], algorithm_names: List[str]):
        """
        Plot statistical analysis of multiple runs
        
        Args:
            all_results: List of lists containing results from multiple runs
            algorithm_names: Names of algorithms
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for alg_idx, (results, alg_name) in enumerate(zip(all_results, algorithm_names)):
            # Extract parameters
            a_vals = [r['parameters']['a'] for r in results]
            Rs_vals = [r['parameters']['Rs'] for r in results]
            Rp_vals = [r['parameters']['Rp'] for r in results]
            fitness_vals = [r['best_fitness'] for r in results]
            
            # Box plots for each parameter
            axes[0, 0].boxplot([a_vals], positions=[alg_idx], labels=[alg_name])
            axes[0, 1].boxplot([Rs_vals], positions=[alg_idx], labels=[alg_name])
            axes[1, 0].boxplot([Rp_vals], positions=[alg_idx], labels=[alg_name])
            axes[1, 1].boxplot([fitness_vals], positions=[alg_idx], labels=[alg_name])
        
        axes[0, 0].set_ylabel('a (Ideality Factor)')
        axes[0, 1].set_ylabel('Rs (Series Resistance) [Ω]')
        axes[1, 0].set_ylabel('Rp (Parallel Resistance) [Ω]')
        axes[1, 1].set_ylabel('Fitness Value')
        axes[1, 1].set_yscale('log')
        
        plt.suptitle('Statistical Analysis of Optimization Results')
        plt.tight_layout()
        plt.show()