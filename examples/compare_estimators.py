# examples/compare_estimators.py

"""
Example demonstrating how to compare different tau² estimators in MetaMeta.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metameta import MetaAnalysis, calculate_variances

# Create simulated data with known heterogeneity
def simulate_meta_data(k=20, mean_effect=0.3, tau2=0.04, min_n=50, max_n=500, seed=42):
    """
    Simulate data for meta-analysis with known parameters.
    
    Parameters:
    -----------
    k : int, default=20
        Number of studies
    mean_effect : float, default=0.3
        True mean effect size
    tau2 : float, default=0.04
        True between-study variance
    min_n, max_n : int
        Minimum and maximum sample sizes
    seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        Effect sizes, variances, sample sizes
    """
    np.random.seed(seed)
    
    # Generate sample sizes
    sample_sizes = np.random.randint(min_n, max_n, size=k)
    
    # Generate true effects from normal distribution with mean and tau²
    true_effects = np.random.normal(mean_effect, np.sqrt(tau2), size=k)
    
    # Generate within-study variances based on sample size
    within_variances = (1 - mean_effect**2)**2 / (sample_sizes - 1)
    
    # Generate observed effects with sampling error
    effect_sizes = np.random.normal(true_effects, np.sqrt(within_variances))
    
    # Return the data
    return effect_sizes, within_variances, sample_sizes


# Simulate data for three scenarios
scenarios = {
    "Low heterogeneity": {"tau2": 0.01, "k": 10},
    "Moderate heterogeneity": {"tau2": 0.05, "k": 20},
    "High heterogeneity": {"tau2": 0.15, "k": 30}
}

methods = ["DL", "PM", "REML", "ML", "HS", "SJ", "HE", "EB"]
results = []

for name, params in scenarios.items():
    print(f"\nSimulating {name} scenario...")
    
    # Simulate data
    effect_sizes, variances, sample_sizes = simulate_meta_data(
        k=params["k"], 
        tau2=params["tau2"]
    )
    
    # Run meta-analysis with different estimators
    for method in methods:
        print(f"  Running {method} estimator...")
        
        if method == "HS":
            # Hunter-Schmidt method needs sample sizes
            meta = MetaAnalysis(
                effect_sizes, 
                sample_sizes=sample_sizes, 
                method=method
            )
        else:
            # Other methods use variances
            meta = MetaAnalysis(
                effect_sizes, 
                variances=variances, 
                method=method
            )
        
        meta.run()
        
        # Calculate bias and store results
        bias = meta.tau2 - params["tau2"]
        rel_bias = bias / params["tau2"] * 100 if params["tau2"] > 0 else np.nan
        
        results.append({
            "Scenario": name,
            "Method": method,
            "True tau²": params["tau2"],
            "Estimated tau²": meta.tau2,
            "Absolute Bias": bias,
            "Relative Bias (%)": rel_bias,
            "Mean Effect": meta.mean_effect,
            "Lower CI": meta.ci_lower,
            "Upper CI": meta.ci_upper,
            "I²": meta.i2,
            "k": params["k"]
        })

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results)

# Print summary table
print("\nSummary of tau² estimation accuracy:")
summary = results_df.groupby(["Scenario", "Method"])[
    ["Estimated tau²", "Absolute Bias", "Relative Bias (%)"]
].mean().reset_index()

for scenario in scenarios.keys():
    print(f"\n{scenario}:")
    scenario_data = summary[summary["Scenario"] == scenario]
    print(scenario_data.sort_values("Absolute Bias").to_string(index=False))

# Create visualization
plt.figure(figsize=(12, 8))

# One subplot for each scenario
for i, scenario in enumerate(scenarios.keys(), 1):
    plt.subplot(1, 3, i)
    
    scenario_data = results_df[results_df["Scenario"] == scenario]
    true_tau2 = scenarios[scenario]["tau2"]
    
    # Bar chart of estimated tau²
    methods_order = scenario_data.groupby("Method")["Absolute Bias"].mean().sort_values().index
    scenario_data_sorted = scenario_data.set_index("Method").loc[methods_order].reset_index()
    
    plt.bar(scenario_data_sorted["Method"], scenario_data_sorted["Estimated tau²"])
    plt.axhline(y=true_tau2, color='r', linestyle='-', label=f'True tau² = {true_tau2}')
    
    plt.title(scenario)
    plt.ylabel("Estimated tau²")
    plt.xticks(rotation=45)
    
    if i == 1:
        plt.legend()

plt.tight_layout()
plt.savefig("estimator_comparison.png", dpi=300, bbox_inches="tight")
print("\nComparison plot saved as 'estimator_comparison.png'")

# Create a table for manuscript
table_df = results_df.pivot_table(
    index="Method",
    columns="Scenario",
    values=["Estimated tau²", "Relative Bias (%)"],
    aggfunc="mean"
)

table_df.to_csv("estimator_comparison_table.csv")
print("Comparison table saved as 'estimator_comparison_table.csv'")