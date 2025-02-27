# examples/advanced_visualizations.py

"""
Example demonstrating advanced visualization options in MetaMeta.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

# Import the visualization functions directly
try:
    from metameta.visualization import forest_plot, funnel_plot, bubble_plot, labbe_plot
    print("Successfully imported visualization functions")
except ImportError as e:
    print(f"Error importing visualization functions: {e}")
    
import numpy as np
import matplotlib.pyplot as plt

# Set up example data
np.random.seed(42)

#--------------------------
# 1. Setup for various plots
#--------------------------

# Example data for continuous outcomes (e.g., standardized mean differences)
effect_sizes = [0.35, 0.42, 0.15, 0.58, -0.12, 0.29, 0.31, 0.08, 0.22, 0.47]
variances = [0.031, 0.028, 0.035, 0.042, 0.026, 0.029, 0.030, 0.027, 0.033, 0.038]
sample_sizes = [85, 92, 78, 65, 110, 88, 84, 95, 82, 70]
publication_years = [2010, 2012, 2013, 2015, 2016, 2017, 2018, 2019, 2020, 2022]
quality_scores = [7, 8, 5, 6, 9, 7, 6, 8, 7, 6]  # Study quality (1-10 scale)
study_names = [f"Study {i+1}" for i in range(len(effect_sizes))]

# Example data for binary outcomes (for L'Abbé plot)
treatment_events = [15, 12, 22, 8, 18, 14, 25, 10, 16, 20]
treatment_total = [50, 60, 70, 40, 80, 50, 90, 60, 70, 55]
control_events = [10, 15, 18, 12, 9, 11, 20, 14, 12, 14]
control_total = [50, 60, 70, 40, 80, 50, 90, 60, 70, 55]

#--------------------------
# 2. Basic meta-analysis
#--------------------------

# Run meta-analysis
meta = MetaAnalysis(
    effect_sizes,
    variances=variances,
    sample_sizes=sample_sizes,
    study_names=study_names,
    effect_measure="SMD"
)
meta.run()

# Print summary
print("Meta-Analysis Results:")
meta.summary()

# Add additional variables to results_df for bubble plot
# Modify this section of your advanced_visualizations.py file:

# Add additional variables to results_df for bubble plot
print(f"\nResults DataFrame shape: {meta.results_df.shape}")

# Check if pooled row is already included
has_pooled_row = "Pooled Effect" in meta.results_df['study'].values

if has_pooled_row:
    # If pooled row is already included
    meta.results_df.loc[meta.results_df['study'] != "Pooled Effect", 'year'] = publication_years
    meta.results_df.loc[meta.results_df['study'] != "Pooled Effect", 'quality'] = quality_scores
    meta.results_df.loc[meta.results_df['study'] != "Pooled Effect", 'sample_size'] = sample_sizes
    # Set values for pooled row
    meta.results_df.loc[meta.results_df['study'] == "Pooled Effect", 'year'] = None
    meta.results_df.loc[meta.results_df['study'] == "Pooled Effect", 'quality'] = None
    meta.results_df.loc[meta.results_df['study'] == "Pooled Effect", 'sample_size'] = sum(sample_sizes)
else:
    # If no pooled row yet
    if len(meta.results_df) == len(effect_sizes):
        meta.results_df['year'] = publication_years
        meta.results_df['quality'] = quality_scores
        meta.results_df['sample_size'] = sample_sizes
    else:
        print("Warning: DataFrame has unexpected shape, skipping additional columns")
#--------------------------
# 3. Create various plots
#--------------------------

# Create and save all plots
plt.figure(figsize=(10, 8))

# Bubble plot with year as x variable and quality as color
print("\nCreating bubble plot...")
bubble_fig = meta.funnel_plot()
plt.savefig("funnel_plot.png", dpi=300, bbox_inches="tight")
plt.close()

# Create bubble plot
print("Creating bubble plot...")
try:
    bubble_fig = bubble_plot(
        meta,
        x_var=publication_years,
        y_var=effect_sizes,
        size_var=sample_sizes,
        color_var=quality_scores,
        title="Effect Size by Publication Year"
    )
    plt.savefig("bubble_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating bubble plot: {str(e)}")

# Create L'Abbé plot for binary outcomes
print("Creating L'Abbé plot...")
try:
    labbe_fig = labbe_plot(
        meta,
        treatment_events=treatment_events,
        treatment_total=treatment_total,
        control_events=control_events,
        control_total=control_total,
        title="L'Abbé Plot of Treatment vs Control Event Rates"
    )
    plt.savefig("labbe_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating L'Abbé plot: {str(e)}")

print("\nAll plots have been saved to disk.")

# Create a bubble plot with sample size as x-variable
print("Creating another bubble plot with sample size...")
# bubble_size_fig = meta.bubble_plot(
#     x_var="sample_size",
#     y_var="effect_size",
#     color_var="year",
#     #title="Effect Size by Sample Size (colored by year)",
#     save_path="bubble_sample_plot.png"
# )
# plt.close()

# Add study quality as variable
import pandas as pd
correlation = np.corrcoef(publication_years, effect_sizes)[0, 1]
print(f"\nCorrelation between publication year and effect size: {correlation:.3f}")

print("\nRegression of effect size on publication year:")
import statsmodels.api as sm
X = sm.add_constant(publication_years)
model = sm.OLS(effect_sizes, X).fit()
print(model.summary())

print("\nAll plots have been saved to disk.")
