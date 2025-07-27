"""
Solar module configurations and constants
"""

# Physical constants
Q = 1.60217646e-19  # Charge of an electron (C)
K = 1.38064852e-23  # Boltzmann's constant (J/K)
T0 = 273.15         # Absolute temperature at 0°C (K)

# Solar module specifications
SOLAR_MODULES = {
    'KC200GT': {
        'name': 'Polycrystalline KC200GT',
        'Voc': 32.9,    # Open circuit voltage (V)
        'Isc': 8.21,    # Short circuit current (A)
        'Vm': 26.3,     # Voltage at max power (V)
        'Im': 7.61,     # Current at max power (A)
        'Nc': 54        # Number of cells
    },
    'SQ85': {
        'name': 'Monocrystalline Shell SQ85',
        'Voc': 22.2,
        'Isc': 5.45,
        'Vm': 17.2,
        'Im': 4.95,
        'Nc': 36
    },
    'ST40': {
        'name': 'Thin film ST40',
        'Voc': 23.3,
        'Isc': 2.68,
        'Vm': 16.6,
        'Im': 2.41,
        'Nc': 36
    }
}

# Optimization parameters
OPTIMIZATION_PARAMS = {
    'bounds': {
        'a': [0.5, 2.0],       # Ideality factor bounds
        'Rs': [0.001, 1.0],    # Series resistance bounds (Ohm)
        'Rp': [50, 200]        # Parallel resistance bounds (Ohm)
    },
    'population_size': 60,
    'max_iterations': 100000,    # max_nfes = 50000 / pop_size
    'num_runs': 1,
    'temperature': 25          # Operating temperature (°C)
}