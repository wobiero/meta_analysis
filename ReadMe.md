
# README.md
# MetaMeta

A comprehensive meta-analysis library with support for multiple tau² estimators.

## Features

- Support for 8 different tau² estimators:
  - DerSimonian-Laird (DL)
  - Paule-Mandel (PM)
  - Restricted Maximum Likelihood (REML)
  - Maximum Likelihood (ML)
  - Hunter-Schmidt (HS)
  - Sidik-Jonkman (SJ)
  - Hedges (HE)
  - Empirical Bayes (EB)
- Multiple effect size types (correlation, standardized mean difference, odds ratio, risk ratio)
- Publication-quality forest plots
- Extensive documentation and examples
- Comprehensive test suite

## Installation

```bash
pip install metameta
```

## Quick Start

```python
import numpy as np
from metameta import MetaAnalysis, calculate_variances

# Example data (correlations)
effect_sizes = [0.25, 0.32, 0.15, 0.42, 0.18, 0.30, 0.22]
sample_sizes = [120, 85, 200, 65, 150, 100, 180]
study_names = ["Smith et al. (2018)", "Johnson & Lee (2019)", "Williams (2017)", 
              "Garcia et al. (2020)", "Brown & Davis (2018)", "Taylor (2019)", 
              "Miller et al. (2021)"]

# Calculate variances
variances = calculate_variances("r", effect_sizes, n_total=sample_sizes)

# Run meta-analysis with DerSimonian-Laird estimator (default)
meta = MetaAnalysis(effect_sizes, variances=variances)
meta.run()
meta.summary()

# Create forest plot
meta.forest_plot(study_names=study_names, save_path="forest_plot.png")
```

## License

MIT