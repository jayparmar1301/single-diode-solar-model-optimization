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
    def plot_multiple_convergence(convergence_dict: Dict[str, List[Dict]], 
                                title: str = "Convergence Comparison"):
        """
        Plot convergence comparison for multiple algorithms
        
        Args:
            convergence_dict: Dictionary with algorithm names as keys and convergence histories as values
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray']
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        
        for i, (algorithm, convergence_history) in enumerate(convergence_dict.items()):
            iterations = [d['iteration'] for d in convergence_history]
            best_fitness = [d['best_fitness'] for d in convergence_history]
            
            plt.plot(iterations, best_fitness, 
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    label=f'{algorithm}', 
                    linewidth=2.5)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Fitness Value', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Add some styling
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_algorithm_statistics(all_results: Dict[str, List[Dict]]):
        """
        Plot comprehensive statistical analysis of algorithm performance
        
        Args:
            all_results: Dictionary with algorithm names as keys and results lists as values
        """
        algorithms = list(all_results.keys())
        n_algorithms = len(algorithms)
        
        # Create subplot layout
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Fitness comparison (box plot)
        ax1 = plt.subplot(2, 3, 1)
        fitness_data = []
        for alg in algorithms:
            fitness_vals = [r['best_fitness'] for r in all_results[alg]]
            fitness_data.append(fitness_vals)
        
        bp1 = ax1.boxplot(fitness_data, labels=algorithms, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_ylabel('Best Fitness Value')
        ax1.set_title('Fitness Distribution Comparison')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 2. Parameter 'a' comparison
        ax2 = plt.subplot(2, 3, 2)
        a_data = []
        for alg in algorithms:
            a_vals = [r['parameters']['a'] for r in all_results[alg]]
            a_data.append(a_vals)
        
        bp2 = ax2.boxplot(a_data, labels=algorithms, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_ylabel('Ideality Factor (a)')
        ax2.set_title('Parameter a Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter Rs comparison
        ax3 = plt.subplot(2, 3, 3)
        Rs_data = []
        for alg in algorithms:
            Rs_vals = [r['parameters']['Rs'] for r in all_results[alg]]
            Rs_data.append(Rs_vals)
        
        bp3 = ax3.boxplot(Rs_data, labels=algorithms, patch_artist=True)
        for patch, color in zip(bp3['boxes'], colors):
            patch.set_facecolor(color)
        
        ax3.set_ylabel('Series Resistance Rs (Ω)')
        ax3.set_title('Parameter Rs Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Parameter Rp comparison
        ax4 = plt.subplot(2, 3, 4)
        Rp_data = []
        for alg in algorithms:
            Rp_vals = [r['parameters']['Rp'] for r in all_results[alg]]
            Rp_data.append(Rp_vals)
        
        bp4 = ax4.boxplot(Rp_data, labels=algorithms, patch_artist=True)
        for patch, color in zip(bp4['boxes'], colors):
            patch.set_facecolor(color)
        
        ax4.set_ylabel('Parallel Resistance Rp (Ω)')
        ax4.set_title('Parameter Rp Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 5. Execution time comparison
        ax5 = plt.subplot(2, 3, 5)
        time_data = []
        for alg in algorithms:
            time_vals = [r['execution_time'] for r in all_results[alg]]
            time_data.append(time_vals)
        
        bp5 = ax5.boxplot(time_data, labels=algorithms, patch_artist=True)
        for patch, color in zip(bp5['boxes'], colors):
            patch.set_facecolor(color)
        
        ax5.set_ylabel('Execution Time (s)')
        ax5.set_title('Execution Time Comparison')
        ax5.grid(True, alpha=0.3)
        
        # 6. Mean fitness bar chart with error bars
        ax6 = plt.subplot(2, 3, 6)
        means = []
        stds = []
        for alg in algorithms:
            fitness_vals = [r['best_fitness'] for r in all_results[alg]]
            means.append(np.mean(fitness_vals))
            stds.append(np.std(fitness_vals))
        
        bars = ax6.bar(algorithms, means, yerr=stds, capsize=5, 
                      color=colors[:len(algorithms)], alpha=0.7, 
                      edgecolor='black', linewidth=1)
        
        ax6.set_ylabel('Mean Best Fitness')
        ax6.set_title('Mean Performance with Std Dev')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean_val:.2e}', ha='center', va='bottom', 
                    fontsize=9, rotation=0)
        
        plt.suptitle('Comprehensive Algorithm Performance Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()
    
    @staticmethod
    def plot_3d_solutions(results: List[List[Dict]], algorithms: List[str] = None):
        """
        Plot 3D scatter of solutions from multiple algorithms
        
        Args:
            results: List of result lists for each algorithm
            algorithms: List of algorithm names
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['r', 'b', 'g', 'orange', 'm', 'c', 'y', 'purple']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        
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
                      s=60, label=alg, alpha=0.7, edgecolors='black')
        
        ax.set_xlabel('a (Ideality Factor)', fontsize=12)
        ax.set_ylabel('Rs (Series Resistance) [Ω]', fontsize=12)
        ax.set_zlabel('Rp (Parallel Resistance) [Ω]', fontsize=12)
        ax.legend(fontsize=11)
        ax.set_title('3D Distribution of Optimized Parameters', fontsize=14, fontweight='bold')
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
        
        colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
        
        for i, (sol, label) in enumerate(zip(solutions, labels)):
            V, I = solar_model.calculate_iv_curve(sol)
            plt.plot(V, I, label=label, linewidth=2.5, 
                    color=colors[i % len(colors)])
        
        plt.xlabel('Voltage (V)', fontsize=12)
        plt.ylabel('Current (A)', fontsize=12)
        plt.title(f'I-V Characteristics - {solar_model.get_module_info()["type"]}', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
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
        
        colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
        
        for i, (sol, label) in enumerate(zip(solutions, labels)):
            V, P = solar_model.calculate_power_curve(sol)
            plt.plot(V, P, label=label, linewidth=2.5, 
                    color=colors[i % len(colors)])
        
        plt.xlabel('Voltage (V)', fontsize=12)
        plt.ylabel('Power (W)', fontsize=12)
        plt.title(f'P-V Characteristics - {solar_model.get_module_info()["type"]}', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
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