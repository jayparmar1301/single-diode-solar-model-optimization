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