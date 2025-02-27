# metameta/meta.py

"""
Core meta-analysis class.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .estimators import (
    dersimonian_laird, paule_mandel, reml, maximum_likelihood,
    hunter_schmidt, sidik_jonkman, hedges, empirical_bayes
)


class MetaAnalysis:
    """
    A flexible meta-analysis class supporting multiple tau² estimators.
    
    Supported estimators:
    - "DL": DerSimonian-Laird (default)
    - "PM": Paule-Mandel
    - "REML": Restricted Maximum Likelihood
    - "ML": Maximum Likelihood
    - "HS": Hunter-Schmidt
    - "SJ": Sidik-Jonkman
    - "HE": Hedges
    - "EB": Empirical Bayes
    """
    
    def __init__(self, effect_sizes, variances=None, sample_sizes=None, 
                 method="DL", reliabilities=None, range_restrictions=None, 
                 study_names=None, effect_measure="g"):
        """
        Initialize meta-analysis with study data.
        
        Parameters:
        -----------
        effect_sizes : array-like
            Effect sizes from each study (correlations or standardized mean differences)
        variances : array-like, optional
            Sampling variances of effect sizes (required for most methods except HS)
        sample_sizes : array-like, optional
            Sample sizes from each study (required for Hunter-Schmidt method)
        method : str, default="DL"
            Method for estimating tau²: "DL", "PM", "REML", "ML", "HS", "SJ", "HE", or "EB"
        reliabilities : array-like, optional
            Reliability coefficients (for Hunter-Schmidt corrections)
        range_restrictions : array-like, optional
            Range restriction values (for Hunter-Schmidt corrections)
        study_names : array-like, optional
            Names or identifiers for studies
        effect_measure : str, default="g"
            Type of effect size used (for display purposes)
        """
        self.effect_sizes = np.array(effect_sizes)
        self.k = len(effect_sizes)  # Number of studies
        self.method = method
        self.effect_measure = effect_measure
        
        if variances is not None:
            self.variances = np.array(variances)
            self.weights = 1 / self.variances  # Fixed-effect weights
        else:
            self.variances = None
            self.weights = None
            
        if sample_sizes is not None:
            self.sample_sizes = np.array(sample_sizes)
            self.total_n = np.sum(sample_sizes)
        else:
            self.sample_sizes = None
            self.total_n = None
            
        self.reliabilities = reliabilities
        self.range_restrictions = range_restrictions
        
        # Study names
        if study_names is None:
            self.study_names = [f"Study {i+1}" for i in range(self.k)]
        else:
            self.study_names = study_names
            
        # Validate inputs based on method
        self._validate_inputs()
        
        # Results to be populated
        self.tau2 = None
        self.tau = None
        self.mean_effect = None
        self.se = None
        self.ci_lower = None
        self.ci_upper = None
        self.q = None
        self.p_value = None
        self.i2 = None
        self.h2 = None
        self.corrected_effect = None
        self.prediction_interval = None
        
    def _validate_inputs(self):
        """Validate that the required inputs for the chosen method are available."""
        if self.method == "HS" and self.sample_sizes is None:
            raise ValueError("Sample sizes are required for Hunter-Schmidt method")
            
        if self.method != "HS" and self.variances is None:
            raise ValueError(f"{self.method} method requires variances")
        
        arrays = [
            ("effect_sizes", self.effect_sizes),
            ("variances", self.variances),
            ("sample_sizes", self.sample_sizes),
            ("study_names", self.study_names)
        ]

        arrays = [(name, arr) for name, arr in arrays if arr is not None]

        for name, arr in arrays[1:]:
            if len(arr) != len(self.effect_sizes):
                raise ValueError(f"Length of {name} ({len(arr)}) does not match length of effect sizes ({len(self.effect_sizes)})")
    
    def run(self, alpha=0.05):
        """
        Run the meta-analysis with the specified method.
        
        Parameters:
        -----------
        alpha : float, default=0.05
            Significance level for confidence intervals
            
        Returns:
        --------
        self : MetaAnalysis
            Returns self for method chaining
        """
        # Calculate critical values for confidence intervals
        z_crit = abs(norm.ppf(alpha / 2))
        
        # Calculate Q statistic (same for all methods)
        if self.method != "HS":
            # Fixed-effect weighted mean for Q calculation
            fe_mean = np.sum(self.effect_sizes * self.weights) / np.sum(self.weights)
            self.q = np.sum(self.weights * (self.effect_sizes - fe_mean)**2)
            df = self.k - 1
            self.p_value = 1 - chi2.cdf(self.q, df) if self.k > 1 else 1.0
            
            # Estimate tau² based on selected method
            if self.method == "DL":
                self.tau2 = dersimonian_laird(self.effect_sizes, self.variances)
            elif self.method == "PM":
                self.tau2 = paule_mandel(self.effect_sizes, self.variances)
            elif self.method == "REML":
                self.tau2 = reml(self.effect_sizes, self.variances)
            elif self.method == "ML":
                self.tau2 = maximum_likelihood(self.effect_sizes, self.variances)
            elif self.method == "SJ":
                self.tau2 = sidik_jonkman(self.effect_sizes, self.variances)
            elif self.method == "HE":
                self.tau2 = hedges(self.effect_sizes, self.variances)
            elif self.method == "EB":
                self.tau2 = empirical_bayes(self.effect_sizes, self.variances)
            
            # Calculate random-effects weights
            self.random_weights = 1 / (self.variances + self.tau2)
            
            # Calculate random-effects mean
            self.mean_effect = np.sum(self.effect_sizes * self.random_weights) / np.sum(self.random_weights)
            
            # Standard error and confidence interval
            self.se = np.sqrt(1 / np.sum(self.random_weights))
            self.ci_lower = self.mean_effect - z_crit * self.se
            self.ci_upper = self.mean_effect + z_crit * self.se
            
            # I² statistic - percentage of variance due to heterogeneity
            if self.q > df:
                self.i2 = 100 * (self.q - df) / self.q
            else:
                self.i2 = 0
                
            # H² statistic - ratio of total variance to sampling variance
            self.h2 = self.q / df
            
            # Prediction interval
            self.tau = np.sqrt(max(0, self.tau2))
            pi_se = np.sqrt(self.se**2 + self.tau2)
            t_crit = abs(t.ppf(alpha / 2, self.k - 1))
            self.prediction_lower = self.mean_effect - t_crit * pi_se
            self.prediction_upper = self.mean_effect + t_crit * pi_se
            self.prediction_interval = (self.prediction_lower, self.prediction_upper)
            
        else:  # Hunter-Schmidt method
            hs_results = hunter_schmidt(
                self.effect_sizes, 
                self.sample_sizes, 
                self.reliabilities, 
                self.range_restrictions
            )
            
            self.tau2 = hs_results["tau2"]
            self.mean_effect = hs_results["mean_effect"]
            self.corrected_effect = hs_results["corrected_effect"]
            
            # Calculate observed variance and sampling error variance
            observed_variance = hs_results["observed_variance"]
            sampling_error_variance = hs_results["sampling_error_variance"]
            
            # Calculate standard error and confidence interval
            self.se = np.sqrt(observed_variance / self.k)
            self.ci_lower = self.mean_effect - z_crit * self.se
            self.ci_upper = self.mean_effect + z_crit * self.se
            
            # Calculate Q and I²
            self.q = np.sum(self.sample_sizes * (self.effect_sizes - self.mean_effect)**2)
            df = self.k - 1
            self.p_value = 1 - chi2.cdf(self.q, df) if self.k > 1 else 1.0
            
            if self.q > df:
                self.i2 = 100 * (self.q - df) / self.q
            else:
                self.i2 = 0
                
            # H² statistic
            self.h2 = self.q / df if df > 0 else 1.0
            
            # Prediction interval
            self.tau = np.sqrt(max(0, self.tau2))
            pi_se = np.sqrt(self.se**2 + self.tau2)
            t_crit = abs(t.ppf(alpha / 2, self.k - 1))
            self.prediction_lower = self.mean_effect - t_crit * pi_se
            self.prediction_upper = self.mean_effect + t_crit * pi_se
            self.prediction_interval = (self.prediction_lower, self.prediction_upper)
        
        # Create results dataframe
        self._create_results_df()
        
        return self
    
    def _create_results_df(self):
        """Create a dataframe with study-level results."""
        data = {
            'study': self.study_names,
            'effect_size': self.effect_sizes,
        }
        
        if self.variances is not None:
            data['variance'] = self.variances
            data['se'] = np.sqrt(self.variances)
            data['ci_lower'] = self.effect_sizes - 1.96 * np.sqrt(self.variances)
            data['ci_upper'] = self.effect_sizes + 1.96 * np.sqrt(self.variances)
            
        if self.sample_sizes is not None:
            data['sample_size'] = self.sample_sizes
            
        if hasattr(self, 'random_weights'):
            data['weight_pct'] = 100 * self.random_weights / np.sum(self.random_weights)
            
        self.results_df = pd.DataFrame(data)
        
        # Add pooled result as an additional row
        pooled = {
            'study': 'Pooled Effect',
            'effect_size': self.mean_effect,
        }
        
        if self.se is not None:
            pooled['se'] = self.se
            pooled['ci_lower'] = self.ci_lower
            pooled['ci_upper'] = self.ci_upper
            
        if self.variances is not None:
            pooled['variance'] = self.se**2
            
        if 'weight_pct' in self.results_df.columns:
            pooled['weight_pct'] = 100
            
        self.pooled_result = pd.DataFrame([pooled])
        
    def summary(self, print_summary=True):
        """
        Generate a summary of meta-analysis results.
        
        Parameters:
        -----------
        print_summary : bool, default=True
            Whether to print the summary to the console
            
        Returns:
        --------
        dict
            Dictionary containing the meta-analysis results
        """
        if self.tau2 is None:
            raise ValueError("Meta-analysis has not been run yet. Call run() first.")
        
        results = {
            "method": self.method,
            "k": self.k,
            "mean_effect": self.mean_effect,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "tau2": self.tau2,
            "tau": self.tau,
            "q": self.q,
            "p_value": self.p_value,
            "i2": self.i2,
            "h2": self.h2,
            "prediction_interval": self.prediction_interval
        }
        
        if self.method == "HS":
            results["total_n"] = self.total_n
            if self.corrected_effect is not None:
                results["corrected_effect"] = self.corrected_effect
        
        if print_summary:
            print("\nMeta-Analysis Results")
            print("=====================")
            print(f"Method for tau² estimation: {self.method}")
            print(f"Number of studies: {self.k}")
            
            if self.method == "HS":
                print(f"Total sample size: {self.total_n}")
                
            print(f"\nPooled effect size ({self.effect_measure}): {self.mean_effect:.4f}")
            print(f"95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]")
            
            if self.method == "HS" and self.corrected_effect is not None:
                print(f"Corrected effect size: {self.corrected_effect:.4f}")
                
            print(f"\nHeterogeneity:")
            print(f"tau²: {self.tau2:.4f}")
            print(f"tau: {self.tau:.4f}")
            print(f"Q statistic: {self.q:.2f}, df = {self.k - 1}, p = {self.p_value:.4f}")
            print(f"I²: {self.i2:.1f}%")
            print(f"H²: {self.h2:.2f}")
            
            print(f"\n95% Prediction interval: [{self.prediction_lower:.4f}, {self.prediction_upper:.4f}]")
        
        return results
    
    def to_dataframe(self):
        """
        Convert meta-analysis results to a pandas DataFrame.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing study-level results
        """
        if not hasattr(self, 'results_df'):
            self._create_results_df()
            
        return pd.concat([self.results_df, self.pooled_result])
    
    from .visualization import forest_plot, funnel_plot, galbraith_plot
    
    def forest_plot(self, figsize=(10, None), annotate_stats=True, save_path=None):
        """Wrapper for the forest_plot function."""
        from .visualization import forest_plot
        return forest_plot(
            self,
            figsize=figsize,
            annotate_stats=annotate_stats,
            save_path=save_path
        )
    
    def funnel_plot(self, figsize=(8, 8), pseudo_ci=True, save_path=None,
                    contour_enhanced=False, title=None, effect_label=None):
        """Wrapper for the funnel_plot function."""
        from .visualization import funnel_plot as viz_funnel_plot
        return viz_funnel_plot(
            self,
            figsize=figsize,
            pseudo_ci=pseudo_ci,
            contour_enhanced=contour_enhanced,
            save_path=save_path,
            title=title,
            effect_label=effect_label
        )
    
    def galbraith_plot(self, figsize=(10,8), save_path=None, title=None,
                       effect_label=None, show_outliers=True, z_threshold=2,
                       display_weights=True, arc_radius=.9,  **kwargs):
        """Wrapper for galbraith_plot function"""
        from .visualization import galbraith_plot as viz_galbraith_plot
        return viz_galbraith_plot(
            self,
            figsize=figsize,
            save_path=save_path,
            title=title,
            effect_label=effect_label,
            show_outliers=show_outliers,
            z_threshold=z_threshold,
            arc_radius=arc_radius,
            display_weights=display_weights,
            **kwargs
        )
    
    def labbe_plot(self, treatment_events=None, treatment_total=None, 
              control_events=None, control_total=None, logscale=True,
              figsize=(8, 8), save_path=None, title=None, show_labels=True, **kwargs):
        """Wrapper for labbe_plot function"""
        from .visualization import labbe_plot as viz_labbe_plot
        return viz_labbe_plot(
            self,
            treatment_events=treatment_events,
            treatment_total=treatment_total,
            control_events=control_events,
            control_total=control_total,
            logscale=logscale,
            figsize=figsize,
            save_path=save_path,
            title=title,
            show_labels=show_labels,
            **kwargs
        )
    def __repr__(self):
        """String representation of the meta-analysis object."""
        if self.tau2 is None:
            return f"MetaAnalysis(k={self.k}, method='{self.method}', not run)"
        
        return (f"MetaAnalysis(k={self.k}, method='{self.method}', "
                f"effect={self.mean_effect:.3f} [{self.ci_lower:.3f}, {self.ci_upper:.3f}], "
                f"tau²={self.tau2:.3f}, I²={self.i2:.1f}%)")


from scipy.stats import norm, t