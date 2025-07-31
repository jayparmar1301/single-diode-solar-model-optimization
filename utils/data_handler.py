"""
Data handling utilities for saving and loading results
"""
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Union
from datetime import datetime


class DataHandler:
    """Handle saving and loading of optimization results"""
    
    @staticmethod
    def save_results(results: Union[Dict, List[Dict]], 
                    filename: str, 
                    directory: str = 'results'):
        """
        Save optimization results to file
        
        Args:
            results: Results dictionary or list of results
            filename: Output filename (without extension)
            directory: Output directory
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename}_{timestamp}"
        
        # Save as JSON
        json_path = os.path.join(directory, f"{base_filename}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        # If multiple runs, also save as CSV
        if isinstance(results, list) and len(results) > 0:
            # Extract data for CSV
            data = []
            for i, res in enumerate(results):
                row = {
                    'run': i + 1,
                    'a': res['parameters']['a'],
                    'Rs': res['parameters']['Rs'],
                    'Rp': res['parameters']['Rp'],
                    'fitness': res['best_fitness'],
                    'nfes': res['function_evaluations']
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            csv_path = os.path.join(directory, f"{base_filename}.csv")
            df.to_csv(csv_path, index=False)
            
            # Save summary statistics
            summary = df.describe()
            summary_path = os.path.join(directory, f"{base_filename}_summary.csv")
            summary.to_csv(summary_path)
            
            print(f"Results saved to:")
            print(f"  - {json_path}")
            print(f"  - {csv_path}")
            print(f"  - {summary_path}")
        else:
            print(f"Results saved to: {json_path}")
    
    @staticmethod
    def save_convergence_history(convergence_history: List[Dict], 
                               filename: str, 
                               directory: str = 'results'):
        """
        Save convergence history to CSV
        
        Args:
            convergence_history: List of convergence data
            filename: Output filename (without extension)
            directory: Output directory
        """
        os.makedirs(directory, exist_ok=True)
        
        df = pd.DataFrame(convergence_history)
        csv_path = os.path.join(directory, f"{filename}_convergence.csv")
        df.to_csv(csv_path, index=False)
        print(f"Convergence history saved to: {csv_path}")
    
    @staticmethod
    def save_comparison_results(comparison_data: Dict, 
                              filename: str, 
                              directory: str = 'results'):
        """
        Save algorithm comparison results to file
        
        Args:
            comparison_data: Dictionary with comparison statistics for each algorithm
            filename: Output filename (without extension)
            directory: Output directory
        """
        os.makedirs(directory, exist_ok=True)
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename}_{timestamp}"
        
        # Save as JSON
        json_path = os.path.join(directory, f"{base_filename}.json")
        with open(json_path, 'w') as f:
            json.dump(comparison_data, f, indent=4, default=str)
        
        # Save as CSV for easy analysis
        csv_data = []
        for algorithm, stats in comparison_data.items():
            row = {'algorithm': algorithm}
            row.update(stats)
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(directory, f"{base_filename}.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"Comparison results saved to:")
        print(f"  - {json_path}")
        print(f"  - {csv_path}")
    
    @staticmethod
    def load_results(filepath: str) -> Union[Dict, List[Dict]]:
        """
        Load results from file
        
        Args:
            filepath: Path to results file
            
        Returns:
            Loaded results
        """
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            return df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
    
    @staticmethod
    def compare_algorithms(results_dict: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Compare results from different algorithms
        
        Args:
            results_dict: Dictionary with algorithm names as keys and results lists as values
            
        Returns:
            DataFrame with comparison statistics
        """
        comparison_data = []
        
        for alg_name, results in results_dict.items():
            a_vals = [r['parameters']['a'] for r in results]
            Rs_vals = [r['parameters']['Rs'] for r in results]
            Rp_vals = [r['parameters']['Rp'] for r in results]
            fitness_vals = [r['best_fitness'] for r in results]
            
            stats = {
                'Algorithm': alg_name,
                'a_mean': np.mean(a_vals),
                'a_std': np.std(a_vals),
                'Rs_mean': np.mean(Rs_vals),
                'Rs_std': np.std(Rs_vals),
                'Rp_mean': np.mean(Rp_vals),
                'Rp_std': np.std(Rp_vals),
                'fitness_mean': np.mean(fitness_vals),
                'fitness_std': np.std(fitness_vals),
                'fitness_min': np.min(fitness_vals),
                'fitness_max': np.max(fitness_vals)
            }
            comparison_data.append(stats)
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def export_for_matlab(results: List[Dict], 
                         variable_name: str,
                         filename: str,
                         directory: str = 'results'):
        """
        Export results in MATLAB format
        
        Args:
            results: List of result dictionaries
            variable_name: MATLAB variable name
            filename: Output filename (without extension)
            directory: Output directory
        """
        os.makedirs(directory, exist_ok=True)
        
        # Extract data
        data = []
        for res in results:
            data.append([
                res['parameters']['a'],
                res['parameters']['Rs'],
                res['parameters']['Rp']
            ])
        
        data_array = np.array(data)
        
        # Write MATLAB-style text file
        matlab_path = os.path.join(directory, f"{filename}.txt")
        with open(matlab_path, 'w') as f:
            f.write(f"{variable_name} = [\n")
            for row in data_array:
                f.write(f"{row[0]:.10f}\t{row[1]:.10f}\t{row[2]:.10f}\n")
            f.write("];\n")
        
        print(f"MATLAB data exported to: {matlab_path}")
    
    @staticmethod
    def export_comprehensive_results(results: List[Dict], 
                                   algorithm_name: str,
                                   filename: str,
                                   directory: str = 'results'):
        """
        Export comprehensive results in detailed format similar to double diode template
        
        Args:
            results: List of result dictionaries
            algorithm_name: Name of the algorithm
            filename: Output filename (without extension)
            directory: Output directory
        """
        os.makedirs(directory, exist_ok=True)
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(directory, f"{filename}_{timestamp}.txt")
        
        # Get algorithm info from first result
        first_result = results[0]
        module_type = first_result.get('module_type', 'Unknown')
        
        # Get algorithm parameters
        if 'algorithm_params' in first_result:
            params = first_result['algorithm_params']
            population_size = params['population_size']
            max_iterations = params['max_iterations']
        else:
            population_size = "Unknown"
            max_iterations = "Unknown"
        
        # Calculate statistics
        fitness_vals = [r['best_fitness'] for r in results]
        execution_times = [r['execution_time'] for r in results]
        total_time = sum(execution_times)
        
        # Algorithm descriptions
        algorithm_info = {
            'JAYA': {
                'full_name': 'Jaya Algorithm',
                'type': 'Population-based metaheuristic',
                'parameters': 'Parameter-free',
                'characteristics': [
                    '- Moves towards best solution and away from worst solution',
                    '- No algorithm-specific parameters to tune',
                    '- Simple yet effective optimization strategy',
                    '- Good balance between exploration and exploitation'
                ],
                'update_mechanism': [
                    '- V\' = V + r1*(V_best - |V|) - r2*(V_worst - |V|)',
                    '- where r1, r2 are random numbers [0,1]',
                    '- |V| represents absolute value of current solution'
                ]
            },
            'BMR': {
                'full_name': 'Best-Mean-Random',
                'type': 'Population-based metaheuristic',
                'parameters': 'T factor (randomly 1 or 2)',
                'characteristics': [
                    '- Uses best solution, population mean, and random solution',
                    '- Combines global and local search strategies',
                    '- Random reinitialization for enhanced exploration',
                    '- T factor adds randomness to solution influence'
                ],
                'update_mechanism': [
                    '- Exploitation: V\' = V + r1*(V_best - T*V_mean) + r2*(V_best - V_random)',
                    '- Exploration: V\' = U - (U - L)*r3',
                    '- where T in {1,2}, r1,r2,r3 are random numbers [0,1]'
                ]
            },
            'BWR': {
                'full_name': 'Best-Worst-Random',
                'type': 'Population-based metaheuristic',
                'parameters': 'T factor (randomly 1 or 2)',
                'characteristics': [
                    '- Uses best solution, worst solution, and random solution',
                    '- Moves towards best and away from worst',
                    '- Random reinitialization for enhanced exploration',
                    '- T factor adds randomness to solution influence'
                ],
                'update_mechanism': [
                    '- Exploitation: V\' = V + r1*(V_best - T*V_random) - r2*(V_worst - V_random)',
                    '- Exploration: V\' = U - (U - L)*r3',
                    '- where T in {1,2}, r1,r2,r3 are random numbers [0,1]'
                ]
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"{algorithm_name} Single Diode Solar Module Optimization Results\n")
            f.write("=" * 68 + "\n")
            f.write(f"Optimizer: {algorithm_name}\n")
            f.write(f"Module Type: {module_type}\n")
            f.write(f"Model Type: SingleDiode\n")
            f.write(f"Number of Runs: {len(results)}\n")
            f.write(f"Population Size: {population_size}\n")
            f.write(f"Max Iterations: {max_iterations}\n")
            
            # Table header
            f.write("Run    a        Rs       Rp        Is         Iph      Fitness      Time\n")
            f.write("-" * 79 + "\n")
            
            # Data rows
            for i, result in enumerate(results, 1):
                a = result['parameters']['a']
                Rs = result['parameters']['Rs']
                Rp = result['parameters']['Rp']
                Is = result['Is']
                Iph = result['Iph']
                fitness = result['best_fitness']
                time_taken = result['execution_time']
                
                f.write(f"{i:3d}  {a:7.4f}  {Rs:7.4f}  {Rp:8.4f}  {Is:8.2e}  {Iph:7.4f}  {fitness:11.6e}  {time_taken:4.2f}\n")
            
            # Statistics
            f.write("=" * 68 + "\n")
            f.write("STATISTICS\n")
            f.write("=" * 68 + "\n")
            f.write(f"Best Fitness: {np.min(fitness_vals):.10e}\n")
            f.write(f"Mean Fitness: {np.mean(fitness_vals):.6e}\n")
            f.write(f"Std Fitness:  {np.std(fitness_vals):.6e}\n")
            f.write(f"Total Time:   {total_time:.2f} seconds\n")
            f.write(f"Avg Time:     {np.mean(execution_times):.2f} seconds/run\n")
            
            # Algorithm information
            f.write("=" * 68 + "\n")
            f.write("ALGORITHM INFORMATION\n")
            f.write("=" * 68 + "\n")
            
            if algorithm_name in algorithm_info:
                info = algorithm_info[algorithm_name]
                f.write(f"Algorithm: {algorithm_name}\n")
                f.write(f"Full Name: {info['full_name']}\n")
                f.write(f"Type: {info['type']}\n")
                f.write(f"Parameters: {info['parameters']}\n")
                f.write("Characteristics:\n")
                for char in info['characteristics']:
                    f.write(f"  {char}\n")
                f.write("Update Mechanism:\n")
                for mech in info['update_mechanism']:
                    f.write(f"  {mech}\n")
            else:
                f.write(f"Algorithm: {algorithm_name}\n")
                f.write("Full Name: Unknown Algorithm\n")
                f.write("Type: Optimization Algorithm\n")
        
        print(f"Comprehensive results exported to: {output_path}")