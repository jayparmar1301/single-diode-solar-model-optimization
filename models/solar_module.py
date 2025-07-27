"""
Solar module model and objective function
"""
import numpy as np
from scipy.optimize import fsolve
from typing import Dict, Tuple, List
from config.solar_modules import Q, K, T0, SOLAR_MODULES


class SolarModuleModel:
    """Solar photovoltaic module model"""
    
    def __init__(self, module_type: str = 'ST40', temperature: float = 25):
        """
        Initialize solar module model
        
        Args:
            module_type: Type of solar module ('KC200GT', 'SQ85', 'ST40')
            temperature: Operating temperature in Celsius
        """
        if module_type not in SOLAR_MODULES:
            raise ValueError(f"Unknown module type: {module_type}")
        
        self.module_specs = SOLAR_MODULES[module_type]
        self.temperature = temperature
        self.T = temperature + T0  # Convert to Kelvin
        
        # Extract module parameters
        self.Voc = self.module_specs['Voc']
        self.Isc = self.module_specs['Isc']
        self.Vm = self.module_specs['Vm']
        self.Im = self.module_specs['Im']
        self.Nc = self.module_specs['Nc']
        
    def objective_function(self, x: np.ndarray) -> float:
        """
        Calculate the objective function (sum of squared errors)
        
        Args:
            x: Array of parameters [a, Rs, Rp]
            
        Returns:
            error: Sum of squared errors
        """
        a, Rs, Rp = x
        
        # Calculate saturation current Is
        exp_voc = np.exp(Q * self.Voc / (a * K * self.Nc * self.T))
        exp_rs_isc = np.exp(Q * Rs * self.Isc / (a * K * self.Nc * self.T))
        
        Is = (self.Isc + Rs * self.Isc / Rp - self.Voc / Rp) / (exp_voc - exp_rs_isc)
        
        # Calculate photocurrent Iph
        Iph = Is * (exp_voc - 1) + self.Voc / Rp
        
        # Calculate errors at three key points
        # Error 1: At open circuit (V = Voc, I = 0)
        eps1 = Is * (exp_voc - 1) + self.Voc / Rp - Iph
        
        # Error 2: At short circuit (V = 0, I = Isc)
        eps2 = self.Isc + Is * (exp_rs_isc - 1) + Rs * self.Isc / Rp - Iph
        
        # Error 3: At maximum power point (V = Vm, I = Im)
        exp_vm_im = np.exp(Q * (self.Vm + Rs * self.Im) / (a * K * self.Nc * self.T))
        eps3 = Iph - Is * (exp_vm_im - 1) - (self.Vm + Rs * self.Im) / Rp - self.Im
        
        # Total error (sum of squared errors)
        error = eps1**2 + eps2**2 + eps3**2
        
        return error
    
    def calculate_parameters(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Is and Iph for given parameters
        
        Args:
            x: Array of parameters [a, Rs, Rp]
            
        Returns:
            Is: Saturation current
            Iph: Photocurrent
        """
        a, Rs, Rp = x
        
        exp_voc = np.exp(Q * self.Voc / (a * K * self.Nc * self.T))
        exp_rs_isc = np.exp(Q * Rs * self.Isc / (a * K * self.Nc * self.T))
        
        Is = (self.Isc + Rs * self.Isc / Rp - self.Voc / Rp) / (exp_voc - exp_rs_isc)
        Iph = Is * (exp_voc - 1) + self.Voc / Rp
        
        return Is, Iph
    
    def calculate_iv_curve(self, x: np.ndarray, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate I-V curve for given parameters
        
        Args:
            x: Array of parameters [a, Rs, Rp]
            num_points: Number of points in the curve
            
        Returns:
            V: Voltage array
            I: Current array
        """
        a, Rs, Rp = x
        Is, Iph = self.calculate_parameters(x)
        
        # Generate voltage points
        V = np.linspace(0, self.Voc, num_points)
        I = np.zeros_like(V)
        
        # Define the implicit equation for current
        def current_equation(i, v):
            return Iph - Is * (np.exp(Q * (v + Rs * i) / (a * K * self.Nc * self.T)) - 1) - (v + Rs * i) / Rp - i
        
        # Solve for current at each voltage point
        for idx, v in enumerate(V):
            try:
                I[idx] = fsolve(current_equation, self.Isc/2, args=(v,))[0]
            except:
                I[idx] = 0
        
        return V, I
    
    def calculate_power_curve(self, x: np.ndarray, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate P-V curve for given parameters
        
        Args:
            x: Array of parameters [a, Rs, Rp]
            num_points: Number of points in the curve
            
        Returns:
            V: Voltage array
            P: Power array
        """
        V, I = self.calculate_iv_curve(x, num_points)
        P = V * I
        return V, P
    
    def get_module_info(self) -> Dict:
        """Get module information"""
        return {
            'type': self.module_specs['name'],
            'Voc': self.Voc,
            'Isc': self.Isc,
            'Vm': self.Vm,
            'Im': self.Im,
            'Pmax': self.Vm * self.Im,
            'Nc': self.Nc,
            'Temperature': self.temperature
        }