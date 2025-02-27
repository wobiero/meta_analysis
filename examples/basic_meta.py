# examples/basic_meta.py

"""
Basic example demonstrating how to use MetaMeta for meta-analysis.
"""

import numpy as np
import matplotlib.pyplot as plt

# Try different import approaches to diagnose the issue
try:
    # Standard import if the package is properly installed
    from metameta import MetaAnalysis, calculate_variances
    print("Successfully imported from metameta package")
except ImportError:
    try:
        # Try an absolute import if the module structure is different
        from metameta.meta import MetaAnalysis
        from metameta.effect_sizes import calculate_variances
        print("Successfully imported using direct module imports")
    except ImportError:
        # If all else fails, try a relative import
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from metameta.meta import MetaAnalysis
        from metameta.effect_sizes import calculate_variances
        print("Successfully imported using path modification")

# Example data (correlations)
effect_sizes = [0.25, 0.32, 0.15, 0.42, 0.18, 0.30, 0.22]
sample_sizes = [120, 85, 200, 65, 150, 100, 180]
study_names = [
    "Smith et al. (2018)", 
    "Johnson & Lee (2019)", 
    "Williams (2017)", 
    "Garcia et al. (2020)", 
    "Brown & Davis (2018)", 
    "Taylor (2019)", 
    "Miller et al. (2021)"
]

# Calculate variances
variances = calculate_variances("r", effect_sizes, n_total=sample_sizes)

# Run meta-analysis with DerSimonian-Laird estimator (default)
meta = MetaAnalysis(
    effect_sizes, 
    variances=variances, 
    study_names=study_names,
    effect_measure="r"
)
meta.run()

# Print summary
meta.summary()

print("Example completed successfully!")