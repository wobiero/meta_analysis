# metameta/utils.py

"""
Utility functions for meta-analysis.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t, chi2


def z_to_p(z, two_tailed=True):
    """
    Convert Z-scores to p-values.
    
    Parameters:
    -----------
    z : float or array-like
        Z-score(s)
    two_tailed : bool, default=True
        If True, compute two-tailed p-value, else one-tailed
        
    Returns:
    --------
    float or numpy.ndarray
        p-value(s)
    """
    z = np.abs(np.array(z))
    if two_tailed:
        return 2 * (1 - norm.cdf(z))
    else:
        return 1 - norm.cdf(z)


def p_to_z(p, two_tailed=True):
    """
    Convert p-values to Z-scores.
    
    Parameters:
    -----------
    p : float or array-like
        p-value(s)
    two_tailed : bool, default=True
        If True, assume p-values are two-tailed
        
    Returns:
    --------
    float or numpy.ndarray
        Z-score(s)
    """
    p = np.array(p)
    if two_tailed:
        p = p / 2
    return -norm.ppf(p)


def cochrans_q_test(effect_sizes, variances):
    """
    Perform Cochran's Q test for heterogeneity.
    
    Parameters:
    -----------
    effect_sizes : array-like
        Effect sizes from each study
    variances : array-like
        Sampling variances of the effect sizes
        
    Returns:
    --------
    tuple
        Q statistic, degrees of freedom, p-value
    """
    effect_sizes = np.array(effect_sizes)
    variances = np.array(variances)
    weights = 1 / variances
    
    k = len(effect_sizes)
    df = k - 1
    
    # Fixed-effect weighted mean
    fe_mean = np.sum(weights * effect_sizes) / np.sum(weights)
    
    # Q statistic
    q = np.sum(weights * (effect_sizes - fe_mean)**2)
    
    # p-value
    p_value = 1 - chi2.cdf(q, df) if k > 1 else 1.0
    
    return q, df, p_value


def i_squared(q, df):
    """
    Calculate I² statistic from Q and degrees of freedom.
    
    Parameters:
    -----------
    q : float
        Q statistic
    df : int
        Degrees of freedom
        
    Returns:
    --------
    float
        I² statistic (percentage)
    """
    if q <= df:
        return 0.0
    return 100 * (q - df) / q


def h_squared(q, df):
    """
    Calculate H² statistic from Q and degrees of freedom.
    
    Parameters:
    -----------
    q : float
        Q statistic
    df : int
        Degrees of freedom
        
    Returns:
    --------
    float
        H² statistic
    """
    if df == 0:
        return np.nan
    return q / df


def egger_test(effect_sizes, standard_errors):
    """
    Perform Egger's regression test for funnel plot asymmetry.
    
    Parameters:
    -----------
    effect_sizes : array-like
        Effect sizes from each study
    standard_errors : array-like
        Standard errors of the effect sizes
        
    Returns:
    --------
    tuple
        Intercept, standard error of intercept, t-value, p-value
    """
    effect_sizes = np.array(effect_sizes)
    standard_errors = np.array(standard_errors)
    
    # The precision is the inverse of the standard error
    precision = 1 / standard_errors
    
    # Create a design matrix with precision as the predictor
    X = np.column_stack((np.ones_like(precision), precision))
    
    # The outcome is the effect size divided by its standard error
    y = effect_sizes / standard_errors
    
    # Perform weighted least squares regression
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    
    # Extract the intercept and its standard error
    intercept = beta[0]
    
    # Calculate the residuals
    residuals = y - X @ beta
    
    # Calculate the residual standard error
    n = len(effect_sizes)
    p = 2  # Number of parameters (intercept and slope)
    
    if n <= p:
        # Not enough data points for inference
        return intercept, np.nan, np.nan, np.nan
    
    resid_var = np.sum(residuals**2) / (n - p)
    
    # Calculate the variance-covariance matrix of the coefficients
    cov_matrix = resid_var * np.linalg.pinv(X.T @ X)
    se_intercept = np.sqrt(cov_matrix[0, 0])
    
    # Calculate t-value and p-value for the intercept
    t_value = intercept / se_intercept
    p_value = 2 * (1 - t.cdf(abs(t_value), n - p))
    
    return intercept, se_intercept, t_value, p_value


def rank_correlation_test(effect_sizes, variances):
    """
    Perform the rank correlation test for funnel plot asymmetry.
    
    Parameters:
    -----------
    effect_sizes : array-like
        Effect sizes from each study
    variances : array-like
        Sampling variances of the effect sizes
        
    Returns:
    --------
    tuple
        Kendall's tau, p-value
    """
    from scipy.stats import kendalltau
    
    effect_sizes = np.array(effect_sizes)
    variances = np.array(variances)
    
    # Calculate the standardized effect sizes
    standard_errors = np.sqrt(variances)
    
    # Rank the effect sizes by their absolute values
    abs_effects = np.abs(effect_sizes)
    ranks_effects = np.argsort(np.argsort(abs_effects))
    
    # Rank the variances
    ranks_variances = np.argsort(np.argsort(variances))
    
    # Calculate Kendall's tau correlation
    tau, p_value = kendalltau(ranks_effects, ranks_variances)
    
    return tau, p_value


def trim_and_fill(effect_sizes, variances, side='right', estimator='DL', iterations=50, alpha=0.05):
    """
    Perform the trim-and-fill method to adjust for publication bias.
    
    Parameters:
    -----------
    effect_sizes : array-like
        Effect sizes from each study
    variances : array-like
        Sampling variances of the effect sizes
    side : str, default='right'
        Side to trim: 'left' or 'right'
    estimator : str, default='DL'
        Estimator for tau²: 'DL', 'PM', 'REML', or 'ML'
    iterations : int, default=50
        Maximum number of iterations
    alpha : float, default=0.05
        Significance level for confidence intervals
        
    Returns:
    --------
    dict
        Dictionary with adjusted meta-analysis results
    """
    from .meta import MetaAnalysis
    
    effect_sizes = np.array(effect_sizes)
    variances = np.array(variances)
    k = len(effect_sizes)
    
    # Initialize output
    result = {
        'original_k': k,
        'filled_k': 0,
        'original_effect': None,
        'adjusted_effect': None,
        'missing_studies': 0,
        'filled_effects': [],
        'filled_variances': []
    }
    
    # Run original meta-analysis
    original_meta = MetaAnalysis(effect_sizes, variances=variances, method=estimator)
    original_meta.run(alpha=alpha)
    result['original_effect'] = original_meta.mean_effect
    result['original_ci'] = (original_meta.ci_lower, original_meta.ci_upper)
    result['original_tau2'] = original_meta.tau2
    
    # Calculate effect size signs relative to the side parameter
    if side == 'right':
        centered_effects = effect_sizes - original_meta.mean_effect
    else:  # side == 'left'
        centered_effects = original_meta.mean_effect - effect_sizes
    
    # Rank the studies by their effect size
    ranks = np.argsort(centered_effects)
    ranked_effects = effect_sizes[ranks]
    ranked_variances = variances[ranks]
    
    # Iterative trim and fill procedure
    iter_count = 0
    missing = 0
    
    while iter_count < iterations:
        iter_count += 1
        
        # Calculate the number of extreme studies to trim
        r0 = np.sum(np.arange(k) + 1) - np.sum(ranks[centered_effects > 0] + 1)
        missing = int(np.ceil(r0))
        
        if missing == 0 or missing == result['missing_studies']:
            # No missing studies or number stabilized
            break
            
        # Update the estimate
        result['missing_studies'] = missing
        
        # If no missing studies, exit
        if missing == 0:
            break
    
    # Generate the "missing" studies by mirroring the most extreme studies
    if missing > 0:
        # Identify most extreme studies
        if side == 'right':
            extreme_idx = np.argsort(effect_sizes)[:missing]
            mirror_point = 2 * original_meta.mean_effect
            filled_effects = mirror_point - effect_sizes[extreme_idx]
        else:  # side == 'left'
            extreme_idx = np.argsort(-effect_sizes)[:missing]
            mirror_point = 2 * original_meta.mean_effect
            filled_effects = mirror_point - effect_sizes[extreme_idx]
        
        # Use the same variances for the mirrored studies
        filled_variances = variances[extreme_idx]
        
        # Store the filled studies
        result['filled_effects'] = filled_effects.tolist()
        result['filled_variances'] = filled_variances.tolist()
        
        # Combine original and filled studies
        combined_effects = np.concatenate([effect_sizes, filled_effects])
        combined_variances = np.concatenate([variances, filled_variances])
        
        # Run adjusted meta-analysis
        adjusted_meta = MetaAnalysis(
            combined_effects, 
            variances=combined_variances, 
            method=estimator
        )
        adjusted_meta.run(alpha=alpha)
        
        result['filled_k'] = missing
        result['adjusted_effect'] = adjusted_meta.mean_effect
        result['adjusted_ci'] = (adjusted_meta.ci_lower, adjusted_meta.ci_upper)
        result['adjusted_tau2'] = adjusted_meta.tau2
    else:
        # No adjustment needed
        result['adjusted_effect'] = result['original_effect']
        result['adjusted_ci'] = result['original_ci']
        result['adjusted_tau2'] = result['original_tau2']
    
    return result


def failsafe_n(effect_sizes, variances, alpha=0.05, target=0):
    """
    Calculate the failsafe N (file drawer number).
    
    Parameters:
    -----------
    effect_sizes : array-like
        Effect sizes from each study
    variances : array-like
        Sampling variances of the effect sizes
    alpha : float, default=0.05
        Significance level
    target : float, default=0
        Target effect size (usually 0 for no effect)
        
    Returns:
    --------
    int
        Failsafe N
    """
    from .meta import MetaAnalysis
    
    effect_sizes = np.array(effect_sizes)
    variances = np.array(variances)
    k = len(effect_sizes)
    
    # Run original meta-analysis (fixed-effect for failsafe N)
    weights = 1 / variances
    mean_effect = np.sum(weights * effect_sizes) / np.sum(weights)
    z_score = (mean_effect - target) / np.sqrt(1 / np.sum(weights))
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    
    # If not significant, return 0
    if p_value >= alpha:
        return 0
    
    # Calculate Rosenthal's failsafe N
    z_alpha = norm.ppf(1 - alpha / 2)
    failsafe = int(np.ceil(k * (abs(z_score) / z_alpha - 1)**2))
    
    return max(0, failsafe)


def combine_p_values(p_values, method='fisher'):
    """
    Combine p-values from multiple studies.
    
    Parameters:
    -----------
    p_values : array-like
        p-values from each study
    method : str, default='fisher'
        Method to combine p-values: 'fisher', 'stouffer', or 'binomial'
        
    Returns:
    --------
    tuple
        Combined statistic, p-value
    """
    p_values = np.array(p_values)
    k = len(p_values)
    
    if method == 'fisher':
        # Fisher's method (-2 * sum(ln(p)))
        statistic = -2 * np.sum(np.log(p_values))
        combined_p = 1 - chi2.cdf(statistic, 2 * k)
        return statistic, combined_p
    
    elif method == 'stouffer':
        # Stouffer's Z-score method
        z_scores = norm.ppf(1 - p_values)
        z_combined = np.sum(z_scores) / np.sqrt(k)
        combined_p = 1 - norm.cdf(z_combined)
        return z_combined, combined_p
    
    elif method == 'binomial':
        # Binomial test
        sig_count = np.sum(p_values < 0.05)
        statistic = sig_count
        combined_p = 1 - binom.cdf(sig_count - 1, k, 0.05)
        return statistic, combined_p
    
    else:
        raise ValueError(f"Unknown method: {method}")


from scipy.stats import binom