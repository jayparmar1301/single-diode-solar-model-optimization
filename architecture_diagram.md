# Architecture Diagram

```mermaid
graph TD
    A[Main Script (main.py)] -->|Runs Optimization| B[Optimization (optimization/)]
    A -->|Loads Configurations| C[Configuration (config/)]
    B -->|Uses Objective Function| D[Models (models/)]
    D -->|Provides Data| E[Results (results/)]
    E -->|Visualized By| F[Utilities (utils/)]
    C -->|Provides Module Specs| D
    B -->|Optimizes Parameters| E
```

## Description
- **Main Script**: Entry point for running optimization trials.
- **Configuration**: Contains specifications for solar modules.
- **Models**: Implements the three-diode model and objective function.
- **Optimization**: Includes JAYA algorithm and base optimizer.
- **Results**: Stores outputs in JSON, CSV, and MATLAB formats.
- **Utilities**: Provides data handling and visualization tools.