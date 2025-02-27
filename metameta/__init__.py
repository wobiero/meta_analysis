#metameta/__init__.py
"""
MetaMeta: A comprehensive meta-analysis package
"""
from .meta import MetaAnalysis
from .effect_sizes import (calculate_variances, convert_effect_size, 
                           calc_effect_from_means, calc_effect_from_t, 
                           calc_effect_from_f, calc_effect_from_2x2, 
                           calc_correlation_from_p)
from .estimators import (dersimonian_laird, paule_mandel, reml, 
                         maximum_likelihood, hunter_schmidt, 
                         sidik_jonkman, hedges, empirical_bayes)
from .visualization import (forest_plot, funnel_plot, baujat_plot, 
                            leave_one_out_plot, bubble_plot, labbe_plot, galbraith_plot)

from .metaregression import MetaRegression

__version__ = "0.1.0"
__all__ = [
    "MetaAnalysis",
    "MetaRegression",
    "calculate_variances",
    "convert_effect_size",
    "calc_effect_from_means",
    "calc_effect_from_t",
    "calc_effect_from_f",
    "calc_effect_from_2x2",
    "calc_correlation_from_p",
    "dersimonian_laird",
    "paule_mandel",
    "reml",
    "maximum_likelihood",
    "hunter_schmidt",
    "sidik_jonkman",
    "hedges",
    "empirical_bayes",
    "forest_plot",
    "funnel_plot",
    "labbe_plot",
    "bubble_plot",
    "leave_one_out_plot",
    "baujat_plot",
    "galbraith_plot"
]
