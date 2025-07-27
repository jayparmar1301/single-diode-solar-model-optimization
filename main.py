"""
Main script for solar module parameter optimization using JAYA algorithm
"""
import numpy as np
import time
from tqdm import tqdm
from config.solar_modules import OPTIMIZATION_PARAMS
from models.solar_module import SolarModuleModel
from optimization.jaya import JAYAOptimizer
from utils.visualization import Visualizer
from utils.data_handler import DataHandler


def run_single_optimization(module_type='ST40', seed=None, verbose=False):
    """
    Run a single optimization
    
    Args:
        module_type: Type of solar module
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Optimization results
    """
    # Create solar module model
    solar_model = SolarModuleModel(
        module_type=module_type,
        temperature=OPTIMIZATION_PARAMS['temperature']
    )
    
    # Create optimizer
    optimizer = JAYAOptimizer(
        objective_func=solar_model.objective_function,
        bounds=OPTIMIZATION_PARAMS['bounds'],
        population_size=OPTIMIZATION_PARAMS['population_size'],
        max_iterations=OPTIMIZATION_PARAMS['max_iterations'],
        seed=seed
    )
    
    # Run optimization
    start_time = time.time()
    best_solution, best_fitness = optimizer.optimize()
    end_time = time.time()
    
    # Get results
    results = optimizer.get_results()
    results['execution_time'] = end_time - start_time
    results['module_type'] = module_type
    
    # Calculate Is and Iph
    Is, Iph = solar_model.calculate_parameters(best_solution)
    results['Is'] = Is
    results['Iph'] = Iph
    
    if verbose:
        print(f"\nOptimization completed in {results['execution_time']:.2f} seconds")
        print(f"Best fitness: {best_fitness:.6e}")
        print(f"Parameters: a={best_solution[0]:.4f}, Rs={best_solution[1]:.4f}, Rp={best_solution[2]:.4f}")
    
    return results


def run_multiple_optimizations(module_type='ST40', num_runs=30):
    """
    Run multiple optimizations with different random seeds
    
    Args:
        module_type: Type of solar module
        num_runs: Number of runs
        
    Returns:
        List of results from all runs
    """
    print(f"\nRunning {num_runs} optimizations for {module_type} module...")
    print("="*50)
    
    all_results = []
    
    for run_id in tqdm(range(num_runs), desc="Progress"):
        # Generate random seed based on current time (similar to MATLAB code)
        seed = int(np.sum(100 * np.array(time.localtime()[:6])))
        
        # Run optimization
        results = run_single_optimization(
            module_type=module_type,
            seed=seed,
            verbose=False
        )
        results['run_id'] = run_id + 1
        
        all_results.append(results)
        
        # Print progress every 5 runs
        if (run_id + 1) % 5 == 0:
            avg_fitness = np.mean([r['best_fitness'] for r in all_results])
            print(f"\nRun {run_id + 1}: Average fitness so far: {avg_fitness:.6e}")
    
    return all_results


def analyze_results(results, algorithm_name='JAYA'):
    """
    Analyze and display results
    
    Args:
        results: List of results from multiple runs
        algorithm_name: Name of the algorithm
    """
    print(f"\n{algorithm_name} Optimization Results Summary")
    print("="*50)
    
    # Extract statistics
    a_vals = [r['parameters']['a'] for r in results]
    Rs_vals = [r['parameters']['Rs'] for r in results]
    Rp_vals = [r['parameters']['Rp'] for r in results]
    fitness_vals = [r['best_fitness'] for r in results]
    
    print(f"Parameter a:  mean={np.mean(a_vals):.4f}, std={np.std(a_vals):.4f}")
    print(f"Parameter Rs: mean={np.mean(Rs_vals):.4f}, std={np.std(Rs_vals):.4f}")
    print(f"Parameter Rp: mean={np.mean(Rp_vals):.4f}, std={np.std(Rp_vals):.4f}")
    print(f"Fitness:      mean={np.mean(fitness_vals):.6e}, min={np.min(fitness_vals):.6e}")
    
    # Find best solution
    best_idx = np.argmin(fitness_vals)
    best_result = results[best_idx]
    print(f"\nBest solution (Run {best_result['run_id']}):")
    print(f"  a  = {best_result['parameters']['a']:.6f}")
    print(f"  Rs = {best_result['parameters']['Rs']:.6f} Ω")
    print(f"  Rp = {best_result['parameters']['Rp']:.6f} Ω")
    print(f"  Fitness = {best_result['best_fitness']:.10e}")
    print(f"  Is = {best_result['Is']:.6e} A")
    print(f"  Iph = {best_result['Iph']:.6f} A")


def main():
    """Main execution function"""
    # Configuration
    module_type = 'ST40'  # Options: 'KC200GT', 'SQ85', 'ST40'
    num_runs = 30
    
    # Print module information
    solar_model = SolarModuleModel(module_type=module_type)
    module_info = solar_model.get_module_info()
    print("\nSolar Module Information")
    print("="*50)
    for key, value in module_info.items():
        print(f"{key}: {value}")
    
    # Run optimizations
    start_total_time = time.time()
    results = run_multiple_optimizations(module_type=module_type, num_runs=num_runs)
    total_time = time.time() - start_total_time
    
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    # Analyze results
    analyze_results(results)
    
    # Save results
    data_handler = DataHandler()
    data_handler.save_results(results, f'JAYA_{module_type}')
    
    # Export for MATLAB compatibility
    data_handler.export_for_matlab(results, 'JAYA_data', f'JAYA_{module_type}_matlab')
    
    # Visualizations
    visualizer = Visualizer()
    
    # Plot convergence of best run
    best_idx = np.argmin([r['best_fitness'] for r in results])
    visualizer.plot_convergence(
        results[best_idx]['convergence_history'],
        title=f"JAYA Convergence - Best Run (Run {results[best_idx]['run_id']})"
    )
    
    # Plot 3D distribution of solutions
    visualizer.plot_3d_solutions([results], ['JAYA'])
    
    # Plot I-V and P-V curves for best solution
    best_solution = results[best_idx]['best_solution']
    visualizer.plot_iv_curves(solar_model, [best_solution], ['Best JAYA Solution'])
    visualizer.plot_pv_curves(solar_model, [best_solution], ['Best JAYA Solution'])
    
    # Optional: Compare with other algorithms if you have their results
    # Load the comparison data from your paste-2.txt file
    comparison_data = {
        'GA': np.array([
            [0.839165567, 0.430151986, 189.5119387],
            [1.269516896, 0.155615914, 112.1161261],
            # ... add all GA data
        ]),
        'PSO': np.array([
            [1.080175791, 0.242190756, 96.89379493],
            [1.193806739, 0.058949933, 63.71204357],
            # ... add all PSO data
        ]),
        # Add other algorithms...
    }
    
    # You can uncomment this to plot comparison if you add the data:
    # plot_algorithm_comparison(comparison_data, results)


if __name__ == "__main__":
    main()