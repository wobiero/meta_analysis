# metameta/effect_sizes.py

"""
Functions for calculating and converting effect sizes.
"""

import numpy as np
from scipy.stats import norm


def calculate_variances(effect_type, effect_sizes, n1=None, n2=None, n_total=None):
    """
    Calculate sampling variances for effect sizes.
    
    Parameters:
    -----------
    effect_type : str
        Type of effect size: "r" (correlation), "d" (Cohen's d), "g" (Hedges' g),
        "OR" (odds ratio), "RR" (risk ratio), "z" (Fisher's z)
    effect_sizes : array-like
        Effect sizes
    n1, n2 : array-like, optional
        Sample sizes for each group (for d, g, OR, RR)
    n_total : array-like, optional
        Total sample size (for correlations and Fisher's z)
        
    Returns:
    --------
    numpy.ndarray
        Sampling variances for the effect sizes
    """
    effect_sizes = np.array(effect_sizes)
    
    if effect_type == "r":
        if n_total is None:
            raise ValueError("n_total is required for correlation effect sizes")
        n_total = np.array(n_total)
        # Variance for correlation coefficient
        return (1 - effect_sizes**2)**2 / (n_total - 1)
    
    elif effect_type == "z":
        if n_total is None:
            raise ValueError("n_total is required for Fisher's z effect sizes")
        n_total = np.array(n_total)
        # Variance for Fisher's z transformation
        return 1 / (n_total - 3)
    
    elif effect_type in ["d", "g"]:
        if n1 is None or n2 is None:
            raise ValueError("n1 and n2 are required for d or g effect sizes")
        n1 = np.array(n1)
        n2 = np.array(n2)
        # Variance for standardized mean difference
        return (n1 + n2) / (n1 * n2) + effect_sizes**2 / (2 * (n1 + n2))
    
    elif effect_type == "OR":
        # For log odds ratio
        if all(param is not None for param in [n1, n2]):
            # If we have sample sizes and effect sizes are log odds ratios
            # This assumes effect_sizes are log(OR) values
            n1 = np.array(n1)
            n2 = np.array(n2)
            # This is an approximation - in practice, need a 2x2 table
            # and should use more specific formula
            return 1/n1 + 1/n2
        else:
            raise ValueError("For OR, need more specific data (2x2 table) or SE values")
    
    elif effect_type == "RR":
        # For log risk ratio
        if all(param is not None for param in [n1, n2]):
            # This assumes effect_sizes are log(RR) values
            n1 = np.array(n1)
            n2 = np.array(n2)
            # This is an approximation - in practice, need a 2x2 table
            # and should use more specific formula
            return 1/n1 + 1/n2
        else:
            raise ValueError("For RR, need more specific data (2x2 table) or SE values")
    
    else:
        raise ValueError(f"Unsupported effect type: {effect_type}")


def convert_effect_size(effect_size, from_type, to_type, **kwargs):
    """
    Convert between different effect size metrics.
    
    Parameters:
    -----------
    effect_size : float or array-like
        Effect size value(s) to convert
    from_type : str
        Original effect size type: "r", "d", "g", "OR", "RR", "z"
    to_type : str
        Target effect size type: "r", "d", "g", "OR", "RR", "z"
    **kwargs : dict
        Additional parameters required for specific conversions
        
    Returns:
    --------
    float or numpy.ndarray
        Converted effect size(s)
    """
    effect_size = np.array(effect_size)
    
    # r to other metrics
    if from_type == "r" and to_type == "z":
        # Fisher's z transformation
        return 0.5 * np.log((1 + effect_size) / (1 - effect_size))
    
    elif from_type == "r" and to_type == "d":
        # Correlation to Cohen's d
        return 2 * effect_size / np.sqrt(1 - effect_size**2)
    
    # Fisher's z to other metrics
    elif from_type == "z" and to_type == "r":
        # Inverse Fisher's z transformation
        exp_2z = np.exp(2 * effect_size)
        return (exp_2z - 1) / (exp_2z + 1)
    
    # d/g conversions
    elif from_type == "d" and to_type == "g":
        # Need degrees of freedom
        if 'df' not in kwargs:
            raise ValueError("Need 'df' parameter for d to g conversion")
        df = kwargs['df']
        j = 1 - (3 / (4 * df - 1))  # Correction factor
        return effect_size * j
    
    elif from_type == "g" and to_type == "d":
        if 'df' not in kwargs:
            raise ValueError("Need 'df' parameter for g to d conversion")
        df = kwargs['df']
        j = 1 - (3 / (4 * df - 1))  # Correction factor
        return effect_size / j
    
    elif from_type == "d" and to_type == "r":
        # Cohen's d to correlation
        return effect_size / np.sqrt(effect_size**2 + 4)
    
    # Log odds ratio conversions
    elif from_type == "OR" and to_type == "d":
        # Assumes effect_size is log(OR)
        return effect_size * np.sqrt(3) / np.pi
    
    elif from_type == "d" and to_type == "OR":
        # d to log odds ratio
        return effect_size * np.pi / np.sqrt(3)
    
    # Same type - no conversion needed
    elif from_type == to_type:
        return effect_size
    
    else:
        raise ValueError(f"Conversion from {from_type} to {to_type} not implemented")


def calc_effect_from_means(mean1, mean2, sd1, sd2, n1, n2, type="d"):
    """
    Calculate standardized mean difference from group means and standard deviations.
    
    Parameters:
    -----------
    mean1, mean2 : float or array-like
        Means of the two groups
    sd1, sd2 : float or array-like
        Standard deviations of the two groups
    n1, n2 : float or array-like
        Sample sizes of the two groups
    type : str, default="d"
        Type of effect size to calculate: "d" (Cohen's d) or "g" (Hedges' g)
        
    Returns:
    --------
    float or numpy.ndarray
        Standardized mean difference (d or g)
    """
    mean1, mean2 = np.array(mean1), np.array(mean2)
    sd1, sd2 = np.array(sd1), np.array(sd2)
    n1, n2 = np.array(n1), np.array(n2)
    
    # Pooled standard deviation
    sd_pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (mean1 - mean2) / sd_pooled
    
    if type == "d":
        return d
    elif type == "g":
        # Hedges' g (corrected d)
        df = n1 + n2 - 2
        j = 1 - (3 / (4 * df - 1))  # Correction factor
        return d * j
    else:
        raise ValueError("Type must be 'd' or 'g'")


def calc_effect_from_t(t_value, n1, n2, type="d"):
    """
    Calculate standardized mean difference from t-value and sample sizes.
    
    Parameters:
    -----------
    t_value : float or array-like
        t-value from independent samples t-test
    n1, n2 : float or array-like
        Sample sizes of the two groups
    type : str, default="d"
        Type of effect size to calculate: "d" (Cohen's d) or "g" (Hedges' g)
        
    Returns:
    --------
    float or numpy.ndarray
        Standardized mean difference (d or g)
    """
    t_value = np.array(t_value)
    n1, n2 = np.array(n1), np.array(n2)
    
    # Cohen's d from t-value
    d = t_value * np.sqrt((n1 + n2) / (n1 * n2))
    
    if type == "d":
        return d
    elif type == "g":
        # Hedges' g (corrected d)
        df = n1 + n2 - 2
        j = 1 - (3 / (4 * df - 1))  # Correction factor
        return d * j
    else:
        raise ValueError("Type must be 'd' or 'g'")


def calc_effect_from_f(f_value, df1, df2, n1, n2, type="d"):
    """
    Calculate standardized mean difference from F-value, degrees of freedom, and sample sizes.
    
    Parameters:
    -----------
    f_value : float or array-like
        F-value from one-way ANOVA or similar test
    df1, df2 : float or array-like
        Degrees of freedom for the F-test
    n1, n2 : float or array-like
        Sample sizes of the two groups
    type : str, default="d"
        Type of effect size to calculate: "d" (Cohen's d) or "g" (Hedges' g)
        
    Returns:
    --------
    float or numpy.ndarray
        Standardized mean difference (d or g)
    """
    f_value = np.array(f_value)
    
    # Cohen's d from F-value
    # This assumes a one-way ANOVA with two groups
    d = np.sqrt(f_value * (n1 + n2) / (n1 * n2))
    
    if type == "d":
        return d
    elif type == "g":
        # Hedges' g (corrected d)
        j = 1 - (3 / (4 * (n1 + n2 - 2) - 1))  # Correction factor
        return d * j
    else:
        raise ValueError("Type must be 'd' or 'g'")


def calc_effect_from_2x2(a, b, c, d, type="OR"):
    """
    Calculate effect size from a 2x2 contingency table.
    
    Parameters:
    -----------
    a, b, c, d : float or array-like
        Cell counts in a 2x2 table:
        | a | b |
        | c | d |
    type : str, default="OR"
        Type of effect size to calculate: "OR" (odds ratio) or "RR" (risk ratio)
        
    Returns:
    --------
    float or numpy.ndarray
        Log odds ratio or log risk ratio
    """
    a, b, c, d = np.array(a), np.array(b), np.array(c), np.array(d)
    
    # Add continuity correction for zero cells
    has_zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    if np.any(has_zero):
        a = np.where(has_zero, a + 0.5, a)
        b = np.where(has_zero, b + 0.5, b)
        c = np.where(has_zero, c + 0.5, c)
        d = np.where(has_zero, d + 0.5, d)
    
    if type == "OR":
        # Log odds ratio
        log_or = np.log((a * d) / (b * c))
        # Variance of log odds ratio
        var_log_or = 1/a + 1/b + 1/c + 1/d
        return log_or, var_log_or
    
    elif type == "RR":
        # Log risk ratio
        log_rr = np.log((a / (a + b)) / (c / (c + d)))
        # Variance of log risk ratio
        var_log_rr = (b / (a * (a + b))) + (d / (c * (c + d)))
        return log_rr, var_log_rr
    
    else:
        raise ValueError("Type must be 'OR' or 'RR'")


def calc_correlation_from_p(p_value, n, sign=1):
    """
    Convert p-value to correlation coefficient.
    
    Parameters:
    -----------
    p_value : float or array-like
        p-value from statistical test
    n : float or array-like
        Sample size
    sign : float or array-like, default=1
        Direction of the effect (1 for positive, -1 for negative)
        
    Returns:
    --------
    float or numpy.ndarray
        Correlation coefficient
    """
    p_value = np.array(p_value)
    n = np.array(n)
    sign = np.array(sign)
    
    # Convert p-value to two-tailed z-score
    z = norm.ppf(1 - p_value / 2)
    
    # Convert z to correlation
    r = sign * np.sqrt(z**2 / (z**2 + n))
    
    return r