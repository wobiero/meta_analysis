#!/usr/bin/env python
"""
Example of using MetaRegression in the metameta package.

This example demonstrates how to:
1. Load study data
2. Perform a meta-regression
3. Visualize the results
4. Export the results to different formats
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Try different import approaches to diagnose the issue
try:
    # Standard import if the package is properly installed
    from metameta import MetaRegression
    print("Successfully imported from metameta package")
except ImportError:
    try:
        # Try an absolute import if the module structure is different
        from metameta.metaregression import MetaRegression
        print("Successfully imported using direct module imports")
    except ImportError:
        # If all else fails, try a relative import
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from metameta.metaregression import MetaRegression
        print("Successfully imported using path modification")


# Create some example data
# In a real scenario, you would load this from a CSV or other data source
np.random.seed(42)  # For reproducibility

# Let's create a dataset with 20 studies
n_studies = 20

# Generate some random study characteristics
publication_year = np.random.randint(2000, 2024, size=n_studies)
sample_size = np.random.randint(30, 500, size=n_studies)
study_quality = np.random.uniform(1, 5, size=n_studies)  # Study quality score from 1-5

# Simulate effect of publication year on effect sizes (newer studies have smaller effect)
true_year_effect = -0.005
# Simulate effect of sample size on effect sizes (larger samples have smaller effect)
true_sample_effect = -0.0003
# Simulate effect of study quality on effect sizes (higher quality = smaller effect)
true_quality_effect = -0.05

# Create effect sizes with relationships to the covariates and some random noise
effect_sizes = (
    0.5  # Base effect
    + true_year_effect * (publication_year - 2000)  # Year effect 
    + true_sample_effect * sample_size  # Sample size effect
    + true_quality_effect * study_quality  # Study quality effect
    + np.random.normal(0, 0.1, size=n_studies)  # Random noise
)

# Create variances (typically related to sample size inversely)
variances = 0.1 / np.sqrt(sample_size) + 0.01

# Create a dataframe with covariates
covariates = pd.DataFrame({
    'publication_year': publication_year,
    'sample_size': sample_size,
    'study_quality': study_quality
})

print("Sample Study Data:")
study_data = pd.DataFrame({
    'study_id': [f"Study_{i+1}" for i in range(n_studies)],
    'publication_year': publication_year,
    'sample_size': sample_size,
    'study_quality': study_quality,
    'effect_size': effect_sizes,
    'variance': variances
})
print(study_data.head())
print("\n")

# Perform meta-regression
print("Performing Meta-Regression...")
meta_reg = MetaRegression(
    effect_sizes=effect_sizes,
    variances=variances, 
    covariates=covariates
)

# Fit the model
results = meta_reg.fit()
print(results.summary())

# Create visualizations
print("\nGenerating plots for each covariate...")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
meta_reg.plot('publication_year')
plt.title('Effect of Publication Year')

plt.subplot(1, 3, 2)
meta_reg.plot('sample_size')
plt.title('Effect of Sample Size')

plt.subplot(1, 3, 3)
meta_reg.plot('study_quality')
plt.title('Effect of Study Quality')

plt.tight_layout()
plt.savefig('metaregression_results.png')
print("Plots saved to 'metaregression_results.png'")

# Export results to different formats
print("\nExporting results to different formats...")
results_df = meta_reg.to_dataframe()
print("Results DataFrame:")
print(results_df)

meta_reg.to_csv('metaregression_results.csv')
print("Results saved to 'metaregression_results.csv'")

meta_reg.to_excel('metaregression_results.xlsx')
print("Results saved to 'metaregression_results.xlsx'")

# Compare actual vs. estimated effects
print("\nComparing true vs. estimated effects:")
print(f"Publication Year - True: {true_year_effect:.6f}, Estimated: {results.params['publication_year']:.6f}")
print(f"Sample Size - True: {true_sample_effect:.6f}, Estimated: {results.params['sample_size']:.6f}")
print(f"Study Quality - True: {true_quality_effect:.6f}, Estimated: {results.params['study_quality']:.6f}")

print("\nMeta-regression analysis complete!")