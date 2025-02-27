# metameta/estimators.py

"""
Implementation of various tau² estimators for random-effects meta-analysis.
"""

import numpy as np
from scipy import optimize


def dersimonian_laird(effect_sizes, variances, weights=None):
    """
    DerSimonian-Laird estimator for tau².
    
    Parameters:
    -----------
    effect_sizes : array-like
        Effect sizes from each study
    variances : array-like
        Sampling variances of the effect sizes
    weights : array-like, optional
        Fixed-effect weights (1/variance). If not provided, will be calculated.
        
    Returns:
    --------
    float
        Estimated tau²
    """
    effect_sizes = np.array(effect_sizes)
    variances = np.array(variances)
    
    if weights is None:
        weights = 1 / variances
        
    k = len(effect_sizes)
    
    # Fixed-effect weighted mean
    mean = np.sum(weights * effect_sizes) / np.sum(weights)
    
    # Q statistic
    q = np.sum(weights * (effect_sizes - mean)**2)
    df = k - 1
    
    # DL estimator
    c = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
    tau2 = max(0, (q - df) / c)
    
    return tau2


def paule_mandel(effect_sizes, variances):
    """
    Paule-Mandel estimator for tau².
    
    Parameters:
    -----------
    effect_sizes : array-like
        Effect sizes from each study
    variances : array-like
        Sampling variances of the effect sizes
        
    Returns:
    --------
    float
        Estimated tau²
    """
    effect_sizes = np.array(effect_sizes)
    variances = np.array(variances)
    k = len(effect_sizes)
    
    def pm_criterion(tau2_val):
        weights = 1 / (variances + tau2_val)
        mean = np.sum(weights * effect_sizes) / np.sum(weights)
        q = np.sum(weights * (effect_sizes - mean)**2)
        return q - (k - 1)
    
    # Find tau² where Q = df using iterative search
    # Start with DL estimate for better initialization
    try:
        dl_est = dersimonian_laird(effect_sizes, variances)
        upper_bound = max(100, dl_est * 10)  # Ensuring upper bound is large enough
        
        # Check if lower or upper bound already satisfy the criterion
        if pm_criterion(0) <= 0:
            return 0
        elif pm_criterion(upper_bound) >= 0:
            # If criterion still positive at upper bound, try larger bound or return upper bound
            return upper_bound
            
        tau2 = optimize.brentq(pm_criterion, 0, upper_bound, xtol=1e-6, rtol=1e-6)
        return max(0, tau2)
    except:
        # Fallback to DL if optimization fails
        return dersimonian_laird(effect_sizes, variances)


def reml(effect_sizes, variances):
    """
    Restricted Maximum Likelihood estimator for tau².
    
    Parameters:
    -----------
    effect_sizes : array-like
        Effect sizes from each study
    variances : array-like
        Sampling variances of the effect sizes
        
    Returns:
    --------
    float
        Estimated tau²
    """
    effect_sizes = np.array(effect_sizes)
    variances = np.array(variances)
    
    def neg_log_reml(tau2_val):
        weights = 1 / (variances + tau2_val)
        mean = np.sum(weights * effect_sizes) / np.sum(weights)
        
        # REML log-likelihood
        loglike = -0.5 * np.sum(np.log(variances + tau2_val)) - \
                  0.5 * np.sum(weights * (effect_sizes - mean)**2) - \
                  0.5 * np.log(np.sum(weights))
        
        return -loglike
    
    # Minimize negative log-likelihood
    # Start with DL estimate for better initialization
    dl_est = dersimonian_laird(effect_sizes, variances)
    upper_bound = max(100, dl_est * 10)
    
    try:
        result = optimize.minimize_scalar(
            neg_log_reml, 
            bounds=(0, upper_bound), 
            method='bounded'
        )
        return max(0, result.x)
    except:
        # Fallback to DL if optimization fails
        return dl_est


def maximum_likelihood(effect_sizes, variances):
    """
    Maximum Likelihood estimator for tau².
    
    Parameters:
    -----------
    effect_sizes : array-like
        Effect sizes from each study
    variances : array-like
        Sampling variances of the effect sizes
        
    Returns:
    --------
    float
        Estimated tau²
    """
    effect_sizes = np.array(effect_sizes)
    variances = np.array(variances)
    
    def neg_log_ml(tau2_val):
        weights = 1 / (variances + tau2_val)
        mean = np.sum(weights * effect_sizes) / np.sum(weights)
        
        # ML log-likelihood
        loglike = -0.5 * np.sum(np.log(variances + tau2_val)) - \
                  0.5 * np.sum(weights * (effect_sizes - mean)**2)
        
        return -loglike
    
    # Minimize negative log-likelihood
    # Start with DL estimate for better initialization
    dl_est = dersimonian_laird(effect_sizes, variances)
    upper_bound = max(100, dl_est * 10)
    
    try:
        result = optimize.minimize_scalar(
            neg_log_ml, 
            bounds=(0, upper_bound), 
            method='bounded'
        )
        return max(0, result.x)
    except:
        # Fallback to DL if optimization fails
        return dl_est


def hunter_schmidt(effect_sizes, sample_sizes, reliabilities=None, range_restrictions=None):
    """
    Hunter-Schmidt estimator for tau².
    
    Parameters:
    -----------
    effect_sizes : array-like
        Effect sizes from each study
    sample_sizes : array-like
        Sample sizes for each study
    reliabilities : array-like, optional
        Reliability coefficients for each study
    range_restrictions : array-like, optional
        Range restriction values for each study
        
    Returns:
    --------
    dict
        Dictionary containing tau², mean effect, and corrected effect
    """
    effect_sizes = np.array(effect_sizes)
    sample_sizes = np.array(sample_sizes)
    
    k = len(effect_sizes)
    total_n = np.sum(sample_sizes)
    
    # Calculate weighted mean
    mean_effect = np.sum(effect_sizes * sample_sizes) / total_n
    
    # Calculate observed variance
    weighted_squared_diff = sample_sizes * (effect_sizes - mean_effect)**2
    observed_variance = np.sum(weighted_squared_diff) / total_n
    
    # Calculate sampling error variance
    sampling_error_variance = (1 - mean_effect**2)**2 * (k / total_n)
    
    # Calculate tau² (corrected variance)
    tau2 = max(0, observed_variance - sampling_error_variance)
    
    # Apply artifact corrections if provided
    corrected_effect = mean_effect
    
    # Correct for measurement reliability
    if reliabilities is not None:
        reliabilities = np.array(reliabilities)
        avg_reliability = np.mean(reliabilities)
        corrected_effect = corrected_effect / np.sqrt(avg_reliability)
    
    # Correct for range restriction
    if range_restrictions is not None:
        range_restrictions = np.array(range_restrictions)
        avg_restriction = np.mean(range_restrictions)
        corrected_effect = corrected_effect / avg_restriction
    
    return {
        "tau2": tau2,
        "mean_effect": mean_effect,
        "corrected_effect": corrected_effect,
        "observed_variance": observed_variance,
        "sampling_error_variance": sampling_error_variance
    }


def sidik_jonkman(effect_sizes, variances):
    """
    Sidik-Jonkman estimator for tau².
    
    Parameters:
    -----------
    effect_sizes : array-like
        Effect sizes from each study
    variances : array-like
        Sampling variances of the effect sizes
        
    Returns:
    --------
    float
        Estimated tau²
    """
    effect_sizes = np.array(effect_sizes)
    variances = np.array(variances)
    
    # Initial estimate using method of moments
    fixed_weights = 1 / variances
    mu_hat = np.sum(fixed_weights * effect_sizes) / np.sum(fixed_weights)
    r0 = np.mean((effect_sizes - mu_hat)**2)
    
    # Ensure r0 is positive
    r0 = max(r0, 0.0001)
    
    # Iterative estimator
    weights = 1 / (variances + r0)
    mu_hat = np.sum(weights * effect_sizes) / np.sum(weights)
    tau2 = np.sum(weights**2 * (effect_sizes - mu_hat)**2) / np.sum(weights**2)
    
    return max(0, tau2)


def hedges(effect_sizes, variances):
    """
    Hedges estimator for tau².
    
    Parameters:
    -----------
    effect_sizes : array-like
        Effect sizes from each study
    variances : array-like
        Sampling variances of the effect sizes
        
    Returns:
    --------
    float
        Estimated tau²
    """
    effect_sizes = np.array(effect_sizes)
    variances = np.array(variances)
    weights = 1 / variances
    
    k = len(effect_sizes)
    
    # Fixed-effect weighted mean
    mean = np.sum(weights * effect_sizes) / np.sum(weights)
    
    # Q statistic
    q = np.sum(weights * (effect_sizes - mean)**2)
    
    # Hedges' estimator
    numerator = q - (k - 1)
    s1 = np.sum(weights)
    s2 = np.sum(weights**2)
    
    # Avoid division by zero
    denominator = s1 - (s2 / s1)
    if denominator <= 0:
        return 0
    
    tau2 = numerator / denominator
    
    return max(0, tau2)


def empirical_bayes(effect_sizes, variances):
    """
    Empirical Bayes (Morris) estimator for tau².
    This is often similar to REML but conceptually different.
    
    Parameters:
    -----------
    effect_sizes : array-like
        Effect sizes from each study
    variances : array-like
        Sampling variances of the effect sizes
        
    Returns:
    --------
    float
        Estimated tau²
    """
    # For now, uses same implementation as REML
    # In a full implementation, this could be modified with EB-specific optimizations
    return reml(effect_sizes, variances)