"""
Main script for solar module parameter optimization using JAYA, BMR, and BWR algorithms
"""
import numpy as np
import time
from tqdm import tqdm
from config.solar_modules import OPTIMIZATION_PARAMS
from models.solar_module import SolarModuleModel
from optimization.jaya import JAYAOptimizer
from optimization.bmr import BMROptimizer
from optimization.bwr import BWROptimizer
from utils.visualization import Visualizer
from utils.data_handler import DataHandler


def get_optimizer(algorithm_name, objective_func, bounds, population_size, max_iterations, seed=None):
    """
    Factory function to create optimizer based on algorithm name
    
    Args:
        algorithm_name: Name of the algorithm ('JAYA', 'BMR', 'BWR')
        objective_func: Objective function to optimize
        bounds: Parameter bounds
        population_size: Population size for this algorithm
        max_iterations: Maximum iterations for this algorithm
        seed: Random seed
        
    Returns:
        Optimizer instance
    """
    optimizers = {
        'JAYA': JAYAOptimizer,
        'BMR': BMROptimizer,
        'BWR': BWROptimizer
    }
    
    if algorithm_name not in optimizers:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {list(optimizers.keys())}")
    
    return optimizers[algorithm_name](
        objective_func=objective_func,
        bounds=bounds,
        population_size=population_size,
        max_iterations=max_iterations,
        seed=seed
    )


def run_single_optimization(algorithm_name='JAYA', module_type='ST40', algorithm_params=None, seed=None, verbose=False):
    """
    Run a single optimization with specified algorithm
    
    Args:
        algorithm_name: Algorithm to use ('JAYA', 'BMR', 'BWR')
        module_type: Type of solar module
        algorithm_params: Dictionary with algorithm-specific parameters (population_size, max_iterations)
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Optimization results
    """
    # Use algorithm-specific parameters if provided, otherwise use defaults
    if algorithm_params is None:
        population_size = OPTIMIZATION_PARAMS['population_size']
        max_iterations = OPTIMIZATION_PARAMS['max_iterations']
    else:
        population_size = algorithm_params.get('population_size', OPTIMIZATION_PARAMS['population_size'])
        max_iterations = algorithm_params.get('max_iterations', OPTIMIZATION_PARAMS['max_iterations'])
    
    # Create solar module model
    solar_model = SolarModuleModel(
        module_type=module_type,
        temperature=OPTIMIZATION_PARAMS['temperature']
    )
    
    # Create optimizer
    optimizer = get_optimizer(
        algorithm_name=algorithm_name,
        objective_func=solar_model.objective_function,
        bounds=OPTIMIZATION_PARAMS['bounds'],
        population_size=population_size,
        max_iterations=max_iterations,
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
    results['algorithm'] = algorithm_name
    results['algorithm_params'] = {
        'population_size': population_size,
        'max_iterations': max_iterations
    }
    
    # Calculate Is and Iph
    Is, Iph = solar_model.calculate_parameters(best_solution)
    results['Is'] = Is
    results['Iph'] = Iph
    
    if verbose:
        print(f"\n{algorithm_name} Optimization completed in {results['execution_time']:.2f} seconds")
        print(f"Parameters: Pop={population_size}, Iter={max_iterations}")
        print(f"Best fitness: {best_fitness:.6e}")
        print(f"Parameters: a={best_solution[0]:.4f}, Rs={best_solution[1]:.4f}, Rp={best_solution[2]:.4f}")
    
    return results


def run_multiple_optimizations(algorithm_name='JAYA', module_type='ST40', algorithm_params=None, num_runs=30):
    """
    Run multiple optimizations with different random seeds for a specific algorithm
    
    Args:
        algorithm_name: Algorithm to use
        module_type: Type of solar module
        algorithm_params: Dictionary with algorithm-specific parameters
        num_runs: Number of runs
        
    Returns:
        List of results from all runs
    """
    # Get algorithm parameters
    if algorithm_params is None:
        pop_size = OPTIMIZATION_PARAMS['population_size']
        max_iter = OPTIMIZATION_PARAMS['max_iterations']
    else:
        pop_size = algorithm_params.get('population_size', OPTIMIZATION_PARAMS['population_size'])
        max_iter = algorithm_params.get('max_iterations', OPTIMIZATION_PARAMS['max_iterations'])
    
    print(f"\nRunning {num_runs} optimizations for {module_type} module using {algorithm_name}...")
    print(f"Parameters: Population={pop_size}, Iterations={max_iter}")
    print("="*60)
    
    all_results = []
    
    for run_id in tqdm(range(num_runs), desc=f"{algorithm_name} Progress"):
        # Generate random seed based on current time and run_id
        seed = int(np.sum(100 * np.array(time.localtime()[:6]))) + run_id
        
        # Run optimization
        results = run_single_optimization(
            algorithm_name=algorithm_name,
            module_type=module_type,
            algorithm_params=algorithm_params,
            seed=seed,
            verbose=False
        )
        results['run_id'] = run_id + 1
        
        all_results.append(results)
        
        # Print progress every 10 runs
        if (run_id + 1) % 10 == 0:
            avg_fitness = np.mean([r['best_fitness'] for r in all_results])
            print(f"\n{algorithm_name} Run {run_id + 1}: Average fitness so far: {avg_fitness:.6e}")
    
    return all_results


def run_all_algorithms(module_type='ST40', num_runs=30, algorithms=['JAYA', 'BMR', 'BWR'], algorithm_params_dict=None):
    """
    Run multiple algorithms for comparison
    
    Args:
        module_type: Type of solar module
        num_runs: Number of runs per algorithm
        algorithms: List of algorithms to run
        algorithm_params_dict: Dictionary with algorithm-specific parameters
        
    Returns:
        Dictionary with results for each algorithm
    """
    all_algorithm_results = {}
    total_start_time = time.time()
    
    print(f"\nRunning comparative study with {len(algorithms)} algorithms")
    print(f"Module: {module_type}, Runs per algorithm: {num_runs}")
    
    # Print algorithm parameters
    if algorithm_params_dict:
        print("\nAlgorithm-specific parameters:")
        for alg in algorithms:
            params = algorithm_params_dict.get(alg, {})
            pop = params.get('population_size', OPTIMIZATION_PARAMS['population_size'])
            iter_count = params.get('max_iterations', OPTIMIZATION_PARAMS['max_iterations'])
            print(f"  {alg}: Population={pop}, Iterations={iter_count}")
    
    print("="*70)
    
    for algorithm in algorithms:
        print(f"\n🚀 Starting {algorithm} Algorithm...")
        start_time = time.time()
        
        # Get algorithm-specific parameters
        algorithm_params = algorithm_params_dict.get(algorithm, None) if algorithm_params_dict else None
        
        results = run_multiple_optimizations(
            algorithm_name=algorithm,
            module_type=module_type,
            algorithm_params=algorithm_params,
            num_runs=num_runs
        )
        
        end_time = time.time()
        algorithm_time = end_time - start_time
        
        all_algorithm_results[algorithm] = results
        
        # Quick summary for this algorithm
        fitness_vals = [r['best_fitness'] for r in results]
        print(f"\n✅ {algorithm} Completed in {algorithm_time:.2f}s")
        print(f"   Best fitness: {np.min(fitness_vals):.6e}")
        print(f"   Average fitness: {np.mean(fitness_vals):.6e}")
        print(f"   Std deviation: {np.std(fitness_vals):.6e}")
    
    total_time = time.time() - total_start_time
    print(f"\n🏁 All algorithms completed in {total_time:.2f} seconds")
    
    return all_algorithm_results


def analyze_results(results, algorithm_name='Algorithm'):
    """
    Analyze and display results for a single algorithm
    
    Args:
        results: List of results from multiple runs
        algorithm_name: Name of the algorithm
    """
    print(f"\n{algorithm_name} Optimization Results Summary")
    print("="*60)
    
    # Show algorithm parameters if available
    if 'algorithm_params' in results[0]:
        params = results[0]['algorithm_params']
        print(f"Algorithm parameters: Population={params['population_size']}, Iterations={params['max_iterations']}")
    
    # Extract statistics
    a_vals = [r['parameters']['a'] for r in results]
    Rs_vals = [r['parameters']['Rs'] for r in results]
    Rp_vals = [r['parameters']['Rp'] for r in results]
    fitness_vals = [r['best_fitness'] for r in results]
    execution_times = [r['execution_time'] for r in results]
    
    print(f"Parameter a:  mean={np.mean(a_vals):.4f}, std={np.std(a_vals):.4f}")
    print(f"Parameter Rs: mean={np.mean(Rs_vals):.4f}, std={np.std(Rs_vals):.4f}")
    print(f"Parameter Rp: mean={np.mean(Rp_vals):.4f}, std={np.std(Rp_vals):.4f}")
    print(f"Fitness:      mean={np.mean(fitness_vals):.6e}, min={np.min(fitness_vals):.6e}")
    print(f"Fitness std:  {np.std(fitness_vals):.6e}")
    print(f"Avg time:     {np.mean(execution_times):.2f}s")
    
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
    print(f"  Time = {best_result['execution_time']:.2f}s")


def compare_algorithms(all_results):
    """
    Compare results from multiple algorithms
    
    Args:
        all_results: Dictionary with results for each algorithm
    """
    print(f"\n🔬 COMPARATIVE ANALYSIS")
    print("="*70)
    
    comparison_data = {}
    
    for algorithm, results in all_results.items():
        fitness_vals = [r['best_fitness'] for r in results]
        execution_times = [r['execution_time'] for r in results]
        
        comparison_data[algorithm] = {
            'best_fitness': np.min(fitness_vals),
            'mean_fitness': np.mean(fitness_vals),
            'std_fitness': np.std(fitness_vals),
            'mean_time': np.mean(execution_times),
            'success_rate': np.sum(np.array(fitness_vals) < 1e-4) / len(fitness_vals) * 100
        }
    
    # Print comparison table
    print(f"{'Algorithm':<10} {'Pop/Iter':<15} {'Best Fitness':<15} {'Mean Fitness':<15} {'Std Fitness':<15} {'Avg Time(s)':<12} {'Success%':<10}")
    print("-" * 105)
    
    for algorithm, data in comparison_data.items():
        # Get algorithm parameters from first result
        first_result = all_results[algorithm][0]
        if 'algorithm_params' in first_result:
            params = first_result['algorithm_params']
            pop_iter = f"{params['population_size']}/{params['max_iterations']}"
        else:
            pop_iter = "Default"
        
        print(f"{algorithm:<10} {pop_iter:<15} {data['best_fitness']:<15.6e} {data['mean_fitness']:<15.6e} "
              f"{data['std_fitness']:<15.6e} {data['mean_time']:<12.2f} {data['success_rate']:<10.1f}")
    
    # Find best performing algorithm
    best_algorithm = min(comparison_data.keys(), 
                        key=lambda x: comparison_data[x]['mean_fitness'])
    
    print(f"\n🏆 Best performing algorithm: {best_algorithm}")
    print(f"   Lowest mean fitness: {comparison_data[best_algorithm]['mean_fitness']:.6e}")
    
    return comparison_data


def create_visualizations(all_results, solar_model):
    """
    Create visualizations for all algorithms
    
    Args:
        all_results: Dictionary with results for each algorithm
        solar_model: Solar module model instance
    """
    visualizer = Visualizer()
    
    print(f"\n📊 Creating visualizations...")
    
    # 1. Convergence comparison
    best_convergences = {}
    best_solutions = {}
    
    for algorithm, results in all_results.items():
        # Find best run for each algorithm
        fitness_vals = [r['best_fitness'] for r in results]
        best_idx = np.argmin(fitness_vals)
        best_convergences[algorithm] = results[best_idx]['convergence_history']
        best_solutions[algorithm] = results[best_idx]['best_solution']
    
    # Plot convergence comparison
    visualizer.plot_multiple_convergence(
        best_convergences,
        title="Convergence Comparison - Best Runs"
    )
    
    # 2. 3D solutions distribution
    all_results_list = [results for results in all_results.values()]
    algorithm_names = list(all_results.keys())
    visualizer.plot_3d_solutions(all_results_list, algorithm_names)
    
    # 3. I-V and P-V curves comparison
    solutions_list = list(best_solutions.values())
    labels = [f"Best {alg}" for alg in algorithm_names]
    
    visualizer.plot_iv_curves(solar_model, solutions_list, labels)
    visualizer.plot_pv_curves(solar_model, solutions_list, labels)
    
    # 4. Statistical comparison plots
    visualizer.plot_algorithm_statistics(all_results)


def main():
    """Main execution function"""
    # Configuration
    module_type = 'SQ85'  # Options: 'KC200GT', 'SQ85', 'ST40'
    num_runs = 30
    algorithms = ['JAYA', 'BMR', 'BWR']  # Algorithms to compare
    
    # Algorithm-specific parameters
    algorithm_params_dict = {
        'JAYA': {
            'population_size': 20,
            'max_iterations': 1000
        },
        'BMR': {
            'population_size': 20,
            'max_iterations': 2000
        },
        'BWR': {
            'population_size': 40,
            'max_iterations': 1000
        }
    }
    
    # Print module information
    solar_model = SolarModuleModel(module_type=module_type)
    module_info = solar_model.get_module_info()
    print("\n☀️ Solar Module Information")
    print("="*50)
    for key, value in module_info.items():
        print(f"{key}: {value}")
    
    # Print optimization parameters
    print(f"\n⚙️ Optimization Parameters")
    print("="*50)
    print(f"Temperature: {OPTIMIZATION_PARAMS['temperature']}°C")
    print(f"Number of runs per algorithm: {num_runs}")
    print(f"Algorithms: {', '.join(algorithms)}")
    
    # Run all algorithms
    all_results = run_all_algorithms(
        module_type=module_type,
        num_runs=num_runs,
        algorithms=algorithms,
        algorithm_params_dict=algorithm_params_dict
    )
    
    # Analyze results for each algorithm
    print(f"\n📈 DETAILED ANALYSIS")
    print("="*70)
    
    for algorithm, results in all_results.items():
        analyze_results(results, algorithm)
    
    # Compare algorithms
    comparison_data = compare_algorithms(all_results)
    
    # Save results
    data_handler = DataHandler()
    
    for algorithm, results in all_results.items():
        # Save individual algorithm results
        data_handler.save_results(results, f'{algorithm}_{module_type}')
        
        # Export for MATLAB compatibility (simple format)
        data_handler.export_for_matlab(results, f'{algorithm}_data', f'{algorithm}_{module_type}_matlab')
        
        # Export comprehensive results (detailed format like your template)
        data_handler.export_comprehensive_results(results, algorithm, f'{algorithm}_{module_type}_comprehensive')
    
    # Save comparison data
    data_handler.save_comparison_results(comparison_data, f'comparison_{module_type}')
    
    # Create visualizations
    create_visualizations(all_results, solar_model)
    
    print(f"\n✅ Analysis completed! Results saved and visualizations created.")
    print(f"📁 Check the output directory for detailed results and plots.")


if __name__ == "__main__":
    main()