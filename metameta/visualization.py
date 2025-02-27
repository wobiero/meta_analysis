# metameta/visualization.py

"""
Visualization functions for meta-analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def forest_plot(meta, figsize=(10, None), annotate_stats=True, 
               save_path=None, show_weights=True, title=None,
               effect_label=None, show_prediction=True, 
               show_zero_line=True, show_mean_line=True, **kwargs):
    """
    Create a forest plot of meta-analysis results.
    
    Parameters:
    -----------
    meta : MetaAnalysis
        MetaAnalysis object with results
    figsize : tuple, optional
        Figure size (width, height). If None for height, it's auto-calculated
    annotate_stats : bool, default=True
        Whether to annotate heterogeneity statistics on the plot
    save_path : str, optional
        Path to save the figure to disk
    show_weights : bool, default=True
        Whether to show study weights in the plot
    title : str, optional
        Title for the plot. If None, a default is used
    effect_label : str, optional
        Label for effect size. If None, uses meta.effect_measure
    show_prediction : bool, default=True
        Whether to show the prediction interval
    **kwargs : dict
        Additional keyword arguments passed to plt.figure()
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    if meta.tau2 is None:
        raise ValueError("Meta-analysis has not been run yet. Call run() first.")
    
    # Get study-level data
    study_names = meta.study_names
    effect_sizes = meta.effect_sizes
    
    # Calculate standard errors and confidence intervals for individual studies
    if meta.method == "HS":
        # For HS, approximate SE from sample size
        if meta.variances is None:
            study_se = np.sqrt((1 - effect_sizes**2)**2 / (meta.sample_sizes - 1))
        else:
            study_se = np.sqrt(meta.variances)
    else:
        study_se = np.sqrt(meta.variances)
        
    study_ci_lower = effect_sizes - 1.96 * study_se
    study_ci_upper = effect_sizes + 1.96 * study_se
    
    # Default effect size label
    if effect_label is None:
        effect_label = meta.effect_measure
    
    # Auto-calculate figure height if not provided
    if figsize[1] is None:
        figsize = (figsize[0], max(4, meta.k * 0.5 + 2.5))
    
    # Calculate study weights as percentages
    if hasattr(meta, 'random_weights'):
        weight_pct = 100 * meta.random_weights / np.sum(meta.random_weights)
    else:
        weight_pct = None
    
    # Create the plot with clean styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
    
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    
    # Default title
    if title is None:
        title = f"Forest Plot ({meta.method} method)"
    
    # Calculate reasonable x-axis limits with padding
    all_limits = np.concatenate([
        study_ci_lower, 
        study_ci_upper, 
        [meta.ci_lower, meta.ci_upper]
    ])
    if show_prediction and hasattr(meta, 'prediction_lower'):
        all_limits = np.concatenate([
            all_limits, 
            [meta.prediction_lower, meta.prediction_upper]
        ])
    
    x_min = min(min(all_limits) - abs(min(all_limits) * 0.2), -.2)
    x_max = max(max(all_limits) + abs(max(all_limits) * 0.2), .2)
    
    # Study row positions
    y_positions = np.arange(meta.k, 0, -1)
    pooled_y = -1  # Position pooled result below studies with gap
    
    # Determine marker sizes based on precision
    if meta.method == "HS" and hasattr(meta, 'sample_sizes'):
        # Scale by sample sizes for Hunter-Schmidt
        norm_sizes = meta.sample_sizes / np.max(meta.sample_sizes)
        marker_sizes = 40 + norm_sizes * 60  # Range from 40 to 100
    elif hasattr(meta, 'random_weights'):
        # Scale by weights for other methods
        norm_sizes = meta.random_weights / np.max(meta.random_weights)
        marker_sizes = 40 + norm_sizes * 60  # Range from 40 to 100
    else:
        marker_sizes = np.ones(meta.k) * 60
    
        
    # Add grid for readability
    #ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # Plot individual studies
    for i, (name, es, lower, upper, y, size) in enumerate(
        zip(study_names, effect_sizes, study_ci_lower, study_ci_upper, 
            y_positions, marker_sizes)):
        
        # Confidence interval line
        ax.plot([lower, upper], [y, y], 'k-', linewidth=1.5, zorder=2)
        
        # Study point
        ax.scatter(es, y, s=size, color='#3366CC', marker="s",
                   edgecolor='white', linewidth=0.5, zorder=3)
        
        # Study name
        ax.text(-0.02, y, name, ha='right', va='center', 
                fontsize=10, transform=ax.get_yaxis_transform(), 
                zorder=4)
        
        # Effect size and CI
        es_text = f"{es:.2f} [{lower:.2f}, {upper:.2f}]"
        ax.text(1.02, y, es_text, ha='left', va='center', 
                fontsize=9, transform=ax.get_yaxis_transform(), 
                zorder=4)
        
        # Weight percentage
        if show_weights and weight_pct is not None:
            ax.text(1.25, y, f"{weight_pct[i]:.1f}%", ha='right', va='center', 
                    fontsize=9, transform=ax.get_yaxis_transform(), 
                    zorder=4)
    
    # Horizontal separator line between studies and pooled effect
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, zorder=1)
    # Plot vertical reference line at zero or no effect
    if show_zero_line and x_min <=0 <= x_max:
        ax.axvline(0, color='blue', linestyle=':', alpha=0.7, zorder=1, label="No Effect")

    if show_mean_line:
        ax.axvline(meta.mean_effect, color='red', linestyle=':', alpha=0.7, zorder=1, 
                  linewidth=1.5, label='Pooled mean')
        
    if (show_zero_line and x_min <=0 <= x_max) or show_mean_line:
        ax.legend(loc="lower right", bbox_to_anchor=(1.2, -.1), frameon=True, framealpha=.7, fontsize=9)
    # Plot pooled effect
    # Diamond for pooled effect (more precisely drawn)
    diamond_height = 0.6
    diamond_half_width = (meta.ci_upper - meta.ci_lower) / 2
    diamond_x = [
        meta.mean_effect - diamond_half_width, 
        meta.mean_effect, 
        meta.mean_effect + diamond_half_width, 
        meta.mean_effect
    ]
    diamond_y = [
        pooled_y, 
        pooled_y + diamond_height/2, 
        pooled_y, 
        pooled_y - diamond_height/2
    ]
    ax.fill(diamond_x, diamond_y, color='#E41A1C', alpha=0.8, zorder=3)
    
    # Add prediction interval if requested
    if show_prediction and hasattr(meta, 'prediction_lower'):
        ax.plot(
            [meta.prediction_lower, meta.prediction_upper], 
            [pooled_y, pooled_y], 
            'k-', alpha=0.5, linewidth=2, zorder=2
        )
    
    # Add pooled effect text and stats
    ax.text(-0.02, pooled_y, "Pooled Effect", ha='right', va='center', 
            fontweight='bold', fontsize=10, transform=ax.get_yaxis_transform(), 
            zorder=4)
    
    pooled_text = f"{meta.mean_effect:.2f} [{meta.ci_lower:.2f}, {meta.ci_upper:.2f}]"
    ax.text(1.02, pooled_y, pooled_text, ha='left', va='center', 
            fontweight='bold', fontsize=9, transform=ax.get_yaxis_transform(), 
            zorder=4)
    
    if show_weights:
        ax.text(1.25, pooled_y, "100.0%", ha='right', va='center', 
                fontweight='bold', fontsize=9, transform=ax.get_yaxis_transform(), 
                zorder=4)
    
    # Add column headers
    ax.text(-0.02, meta.k + 0.5, "Study", ha='right', va='bottom', 
            fontweight='bold', fontsize=11, transform=ax.get_yaxis_transform())
    
    ax.text(1.02, meta.k + 0.5, f"{effect_label} [95% CI]", ha='left', va='bottom', 
            fontweight='bold', fontsize=11, transform=ax.get_yaxis_transform())
    
    if show_weights:
        ax.text(1.25, meta.k + 0.5, "Weight", ha='right', va='bottom', 
                fontweight='bold', fontsize=11, transform=ax.get_yaxis_transform())
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(pooled_y - 1, meta.k + 1)
    
    # Format axes
    ax.set_xlabel(f"Effect Size ({effect_label})", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    
    # Set title with proper spacing
    ax.set_title(title, fontsize=14, pad=20)
    
    # Annotate heterogeneity statistics if requested
    if annotate_stats:
        stats_text = (
            f"Heterogeneity: τ² = {meta.tau2:.3f}, I² = {meta.i2:.1f}%, "
            f"Q = {meta.q:.2f} (p = {meta.p_value:.3f})"
        )
        
        # Add text at the bottom of the figure
        fig.text(0.5, 0.01, stats_text, ha='center', va='bottom', 
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8, 
                                         edgecolor='none', pad=5))
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

"""
Extended funnel_plot function with option for contour enhancement
"""

def funnel_plot(meta, figsize=(8, 8), pseudo_ci=True, 
               contour_enhanced=False, save_path=None, title=None, 
               effect_label=None, **kwargs):
    """
    Create a funnel plot to assess publication bias.
    
    Parameters:
    -----------
    meta : MetaAnalysis
        MetaAnalysis object with results
    figsize : tuple, optional
        Figure size (width, height)
    pseudo_ci : bool, default=True
        Whether to show pseudo-confidence intervals
    contour_enhanced : bool, default=False
        Whether to show a contour-enhanced funnel plot with significance contours
    save_path : str, optional
        Path to save the figure to disk
    title : str, optional
        Title for the plot. If None, a default is used
    effect_label : str, optional
        Label for effect size. If None, uses meta.effect_measure
    **kwargs : dict
        Additional keyword arguments passed to plt.figure()
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import norm
    
    if meta.tau2 is None:
        raise ValueError("Meta-analysis has not been run yet. Call run() first.")
    
    # Get study-level data
    effect_sizes = meta.effect_sizes
    
    # Calculate standard errors and precisions
    if meta.variances is not None:
        se = np.sqrt(meta.variances)
        precisions = 1 / se
    elif meta.sample_sizes is not None:
        precisions = np.sqrt(meta.sample_sizes)
        se = 1 / precisions
    else:
        raise ValueError("Need either variances or sample sizes for funnel plot")
    
    # Default title and effect label
    if title is None:
        title = "Contour-Enhanced Funnel Plot" if contour_enhanced else "Funnel Plot"
    if effect_label is None:
        effect_label = meta.effect_measure
    
    # Set up figure with clean styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
    
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    
    # Determine marker sizes based on study sizes/weights
    if hasattr(meta, 'sample_sizes'):
        sizes = meta.sample_sizes / np.max(meta.sample_sizes) * 100 + 30
    elif hasattr(meta, 'random_weights'):
        sizes = meta.random_weights / np.max(meta.random_weights) * 100 + 30
    else:
        sizes = np.ones(meta.k) * 60
    
    # Determine reasonable axis limits
    effect_range = max(effect_sizes) - min(effect_sizes)
    # Make sure we have at least some range
    if effect_range < 0.1:
        effect_range = 0.1
        
    x_min = min(effect_sizes) - effect_range * 0.3
    x_max = max(effect_sizes) + effect_range * 0.3
    
    # Ensure x_min and x_max are symmetric around the pooled effect if close
    if abs((x_max + x_min) / 2 - meta.mean_effect) < effect_range * 0.1:
        max_dist = max(abs(meta.mean_effect - x_min), abs(x_max - meta.mean_effect))
        x_min = meta.mean_effect - max_dist
        x_max = meta.mean_effect + max_dist
    
    # Calculate y-axis limits (standard error)
    y_min = 0
    y_max = max(se) * 1.3

    if contour_enhanced == False:
        y_min = min(se)
    
    # For contour-enhanced funnel plot
    if contour_enhanced:
        # We work with standard errors on y-axis for contour-enhanced plot
        y_plot = se
        
        # Create the contour shading to indicate significance zones
        # Area outside of 1% significance (p > 0.01, z = 2.58)
        z_1pct = 2.58
        z_5pct = 1.96
        z_10pct = 1.65
        
        # Determine coordinates for each significance region
        # Note: in a funnel plot, y-axis is standard error, and the lines
        # show where the effect size would be significant at a given p-value
        
        # Create significance zone for p < 0.01
        x_range = np.linspace(x_min, x_max, 1000)
        y_range = np.linspace(0, y_max, 1000)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Calculate z-statistics for each point
        Y_safe = np.where(Y == 0, np.finfo(float).eps, Y)
        Z = np.abs((X - meta.mean_effect) / Y_safe)
        
        # Fill the contours by significance level
        # p < 0.01
        ax.contourf(X, Y, Z, levels=[z_1pct, 100], colors=['#f2f2f2'], alpha=0.6)
        # 0.01 < p < 0.05
        ax.contourf(X, Y, Z, levels=[z_5pct, z_1pct], colors=['#cccccc'], alpha=0.6)
        # 0.05 < p < 0.10
        ax.contourf(X, Y, Z, levels=[z_10pct, z_5pct], colors=['#999999'], alpha=0.6)
        # p > 0.10 - leave as white
        
        # Create a legend for the significance contours
        # Create custom legend elements
        handles = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#f2f2f2', 
                   markersize=15, label='p > 0.10'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#999999', 
                   markersize=15, label='0.05 < p < 0.10'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#cccccc', 
                   markersize=15, label='0.01 < p < 0.05'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='white', 
                   markersize=15, label='p < 0.01')
        ]
        
        # Add legend
        legend1 = ax.legend(handles=handles, loc='lower left', title='Significance', 
                          frameon=True, framealpha=0.9)
        ax.add_artist(legend1)
        
        # Set y-axis label for standard error
        y_label = "Standard Error"
        
    else:
        # Regular funnel plot with precision (1/SE) on y-axis
        y_plot = precisions
        
        # Set y-axis label for precision
        y_label = "Precision (1/SE)"
        y_min = 0
        y_max = max(precisions) * 1.1
    
    # Add light grid for readability
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # Plot the studies as points with better styling
    ax.scatter(
        effect_sizes, y_plot, 
        s=sizes, 
        c='#3366CC', 
        alpha=0.7, 
        edgecolor='white',
        linewidth=0.5,
        zorder=3
    )
    
    # Add study names as tooltips or annotations if not too many
    if meta.k <= 15:  # Only label if not too crowded
        for i, name in enumerate(meta.study_names):
            ax.annotate(
                name,
                (effect_sizes[i], y_plot[i]),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7,
                zorder=4
            )
    
    # Plot vertical line for pooled effect
    ax.axvline(meta.mean_effect, color='#E41A1C', linestyle='-', linewidth=1.5, 
               label=f"Pooled Effect ({meta.mean_effect:.3f})", zorder=2)
    
    # Add confidence interval shading for pooled effect
    ax.axvspan(meta.ci_lower, meta.ci_upper, alpha=0.1, color='#E41A1C', zorder=1)
    
    # Add pseudo confidence intervals if requested and not in contour-enhanced mode
    if pseudo_ci and not contour_enhanced:
        # Generate points along the y-axis
        y_vals = np.linspace(y_min + 0.01, y_max, 100)  # Avoid division by zero
        
        # Generate contours for different significance levels
        for level, style in zip([0.05, 0.01], [('k--', 0.5), ('k:', 0.3)]):
            line_style, alpha = style
            z = abs(norm.ppf(level / 2))
            
            # Plot the contour lines
            if not contour_enhanced:
                # For regular funnel plot (precision on y-axis)
                upper_bounds = []
                lower_bounds = []
                
                for precision in y_vals:
                    se_val = 1 / precision
                    upper_bound = meta.mean_effect + z * se_val
                    lower_bound = meta.mean_effect - z * se_val
                    upper_bounds.append(upper_bound)
                    lower_bounds.append(lower_bound)
                
                # Plot the contour lines
                ax.plot(upper_bounds, y_vals, line_style, alpha=alpha, zorder=1, 
                        label=f"{int((1-level)*100)}% CI" if level == 0.05 else None)
                ax.plot(lower_bounds, y_vals, line_style, alpha=alpha, zorder=1)
                
    # Set labels with proper font sizes
    ax.set_xlabel(f"Effect Size ({effect_label})", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add vertical reference line at zero if it's in range
    if x_min <= 0 <= x_max:
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)
    
    # For contour-enhanced funnel plot, invert y-axis (standard error should increase downward)
    if contour_enhanced:
        ax.invert_yaxis()
    
    # Add legend for pooled effect
    if not contour_enhanced:
        ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    
    # Add annotation about heterogeneity
    if meta.i2 is not None:
        stats_text = f"I² = {meta.i2:.1f}%, τ² = {meta.tau2:.3f}"
        
        # Add text box in bottom left or right
        position = 'right' if contour_enhanced else 'left'
        x_pos = 0.98 if position == 'right' else 0.02
        ax.text(
            x_pos, 0.03, stats_text,
            transform=ax.transAxes,
            ha=position, va='bottom',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5),
            zorder=5
        )
    
    # Clean up the plot style
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=10)
    
    # Add a note about interpretation if few studies
    if meta.k < 10:
        ax.text(
            0.98, 0.97,
            f"Note: Funnel plot has limited\ninterpretability with only {meta.k} studies",
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=8, style='italic',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5),
            zorder=5
        )
    
    # Adjust layout
    fig.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def leave_one_out_plot(meta, figsize=(10, 6), save_path=None, 
                      title=None, effect_label=None, **kwargs):
    """
    Create a leave-one-out analysis plot to assess influence of individual studies.
    
    Parameters:
    -----------
    meta : MetaAnalysis
        MetaAnalysis object with results
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure to disk
    title : str, optional
        Title for the plot. If None, a default is used
    effect_label : str, optional
        Label for effect size. If None, uses meta.effect_measure
    **kwargs : dict
        Additional keyword arguments passed to plt.figure()
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    dict
        Leave-one-out results
    """
    if meta.tau2 is None:
        raise ValueError("Meta-analysis has not been run yet. Call run() first.")
    
    # Default title and effect label
    if title is None:
        title = "Leave-one-out Analysis"
    if effect_label is None:
        effect_label = meta.effect_measure
    
    # Run leave-one-out analyses
    k = meta.k
    loo_results = []
    
    for i in range(k):
        # Exclude one study
        idx = np.ones(k, dtype=bool)
        idx[i] = False
        
        # Create new data excluding the study
        es_subset = meta.effect_sizes[idx]
        
        # Create a new meta-analysis with the subset
        if meta.method == "HS":
            ss_subset = meta.sample_sizes[idx]
            
            # Create new meta-analysis with HS method
            loo_meta = type(meta)(
                es_subset, 
                sample_sizes=ss_subset,
                method=meta.method
            )
        else:
            var_subset = meta.variances[idx]
            
            # Create new meta-analysis with the same method
            loo_meta = type(meta)(
                es_subset, 
                variances=var_subset,
                method=meta.method
            )
        
        # Run the meta-analysis
        loo_meta.run()
        
        # Save the results
        loo_results.append({
            'excluded_study': meta.study_names[i],
            'effect': loo_meta.mean_effect,
            'ci_lower': loo_meta.ci_lower,
            'ci_upper': loo_meta.ci_upper,
            'tau2': loo_meta.tau2,
            'i2': loo_meta.i2
        })
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    
    # Plot positions
    y_positions = np.arange(k + 1)
    
    # Add the original pooled result
    ax.plot(
        [meta.ci_lower, meta.ci_upper], 
        [0, 0], 
        'r-', 
        linewidth=2
    )
    ax.plot(
        meta.mean_effect, 
        0, 
        'ro', 
        ms=8
    )
    
    # Plot individual leave-one-out results
    for i, res in enumerate(loo_results):
        # i+1 because 0 is the original pooled result
        ax.plot(
            [res['ci_lower'], res['ci_upper']], 
            [i+1, i+1], 
            'b-', 
            linewidth=1.5
        )
        ax.plot(
            res['effect'], 
            i+1, 
            'bo', 
            ms=6
        )
    
    # Set y-tick labels
    y_labels = ['Pooled Effect'] + [f"Excl. {res['excluded_study']}" for res in loo_results]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    
    # Set labels and title
    ax.set_xlabel(f"Effect Size ({effect_label})")
    ax.set_title(title)
    
    # Add vertical line at pooled effect
    ax.axvline(meta.mean_effect, linestyle='--', color='gray', alpha=0.7)
    
    # Add vertical line at zero/no effect
    ax.axvline(0, linestyle='-', color='gray', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, loo_results


def baujat_plot(meta, figsize=(8, 8), save_path=None, title=None, **kwargs):
    """
    Create a Baujat plot to identify studies that contribute to heterogeneity.
    
    Parameters:
    -----------
    meta : MetaAnalysis
        MetaAnalysis object with results
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure to disk
    title : str, optional
        Title for the plot. If None, a default is used
    **kwargs : dict
        Additional keyword arguments passed to plt.figure()
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    if meta.tau2 is None or meta.method == "HS":
        raise ValueError("Baujat plot requires a non-HS method meta-analysis")
    
    if not hasattr(meta, 'weights') or not hasattr(meta, 'random_weights'):
        raise ValueError("Meta-analysis lacks required weight attributes")
    
    # Default title
    if title is None:
        title = "Baujat Plot"
    
    # Calculate Baujat plot coordinates
    # X-axis: contribution to Q
    # Y-axis: influence on the pooled effect size
    
    # Fixed effect mean
    fe_mean = np.sum(meta.effect_sizes * meta.weights) / np.sum(meta.weights)
    
    # Contribution to heterogeneity (Q statistic)
    q_contrib = meta.weights * (meta.effect_sizes - fe_mean)**2
    
    # Influence on pooled effect
    influence = meta.random_weights**2 * (meta.effect_sizes - meta.mean_effect)**2
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    
    # Plot the studies as points
    sc = ax.scatter(
        q_contrib, 
        influence, 
        s=50, 
        alpha=0.7, 
        c=meta.random_weights/np.sum(meta.random_weights)*100,
        cmap='viridis'
    )
    
    # Add colorbar for weights
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Weight (%)')
    
    # Add study labels
    for i, name in enumerate(meta.study_names):
        ax.annotate(
            name, 
            (q_contrib[i], influence[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    # Set labels and title
    ax.set_xlabel('Contribution to heterogeneity (Q)')
    ax.set_ylabel('Influence on pooled effect')
    ax.set_title(title)
    
    # Add grid
    #ax.grid(False, linestyle='--', alpha=0.3)
    
    # Make axes start at 0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def bubble_plot(meta, x_var, y_var=None, size_var='weight', color_var=None,
               figsize=(10, 8), save_path=None, title=None, legend=True,
               regressions=True, **kwargs):
    """
    Create a bubble plot for meta-regression or exploring relationships between variables.
    
    Parameters:
    -----------
    meta : MetaAnalysis
        MetaAnalysis object with results
    x_var : str or array-like
        Variable for x-axis. Either column name in meta.results_df or array of values
    y_var : str or array-like, optional
        Variable for y-axis. Either column name in meta.results_df or array of values.
        If None, uses effect sizes.
    size_var : str or array-like, default='weight'
        Variable for bubble size. Either 'weight', column name, or array of values.
    color_var : str or array-like, optional
        Variable for bubble color. Either column name or array of values.
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure to disk
    title : str, optional
        Title for the plot. If None, a default is used
    legend : bool, default=True
        Whether to display a legend for bubble sizes
    regressions : bool, default=True
        Whether to show regression lines (weighted and unweighted)
    **kwargs : dict
        Additional keyword arguments passed to plt.figure()
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    if meta.tau2 is None:
        raise ValueError("Meta-analysis has not been run yet. Call run() first.")
    
    # Ensure results DataFrame is available
    if not hasattr(meta, 'results_df'):
        meta._create_results_df()
    
    # Get data for x-axis
    if isinstance(x_var, str):
        if x_var in meta.results_df.columns:
            x_data = meta.results_df[x_var].iloc[:-1].values  # Exclude pooled row
        else:
            raise ValueError(f"Column '{x_var}' not found in results DataFrame")
    else:
        x_data = np.array(x_var)
        if len(x_data) != meta.k:
            raise ValueError(f"Length of x_var ({len(x_data)}) must match number of studies ({meta.k})")
    
    # Get data for y-axis
    if y_var is None:
        y_data = meta.effect_sizes
        y_label = f"Effect Size ({meta.effect_measure})"
    elif isinstance(y_var, str):
        if y_var in meta.results_df.columns:
            y_data = meta.results_df[y_var].iloc[:-1].values  # Exclude pooled row
            y_label = y_var
        else:
            raise ValueError(f"Column '{y_var}' not found in results DataFrame")
    else:
        y_data = np.array(y_var)
        if len(y_data) != meta.k:
            raise ValueError(f"Length of y_var ({len(y_data)}) must match number of studies ({meta.k})")
        y_label = "y_var"
    
    # Get data for bubble size
    if size_var == 'weight':
        if hasattr(meta, 'random_weights'):
            # Use random-effects weights
            size_data = meta.random_weights / np.max(meta.random_weights) * 500
        else:
            # Use fixed-effect weights if random weights not available
            size_data = meta.weights / np.max(meta.weights) * 500
    elif isinstance(size_var, str):
        if size_var in meta.results_df.columns:
            size_values = meta.results_df[size_var].iloc[:-1].values  # Exclude pooled row
            size_data = size_values / np.max(size_values) * 500
        else:
            raise ValueError(f"Column '{size_var}' not found in results DataFrame")
    else:
        size_values = np.array(size_var)
        if len(size_values) != meta.k:
            raise ValueError(f"Length of size_var ({len(size_values)}) must match number of studies ({meta.k})")
        size_data = size_values / np.max(size_values) * 500
    
    # Get data for bubble color
    if color_var is None:
        color_data = None
        color_label = None
    elif isinstance(color_var, str):
        if color_var in meta.results_df.columns:
            color_data = meta.results_df[color_var].iloc[:-1].values  # Exclude pooled row
            color_label = color_var
        else:
            raise ValueError(f"Column '{color_var}' not found in results DataFrame")
    else:
        color_data = np.array(color_var)
        if len(color_data) != meta.k:
            raise ValueError(f"Length of color_var ({len(color_data)}) must match number of studies ({meta.k})")
        color_label = "color_var"
    
    # Default title
    if title is None:
        title = "Bubble Plot"
        if isinstance(x_var, str):
            title += f" ({x_var} vs. {y_label})"
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    
    # Plot the bubbles
    if color_data is not None:
        sc = ax.scatter(x_data, y_data, s=size_data, c=color_data, alpha=0.6, cmap='viridis')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(color_label)
    else:
        sc = ax.scatter(x_data, y_data, s=size_data, alpha=0.6)
    
    # Add study labels
    for i, name in enumerate(meta.study_names):
        ax.annotate(
            name,
            (x_data[i], y_data[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    # Add regression lines if requested
    if regressions and len(x_data) > 2:
        # Simple unweighted regression
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
            
            x_range = np.linspace(np.min(x_data), np.max(x_data), 100)
            y_pred = intercept + slope * x_range
            
            ax.plot(x_range, y_pred, 'b--', alpha=0.5, 
                   label=f'Unweighted (r={r_value:.2f}, p={p_value:.3f})')
            
            # Weighted regression
            if hasattr(meta, 'random_weights'):
                # Use numpy's polyfit with weights
                weights = meta.random_weights / np.sum(meta.random_weights)
                w_slope, w_intercept = np.polyfit(x_data, y_data, 1, w=weights)
                
                w_y_pred = w_intercept + w_slope * x_range
                ax.plot(x_range, w_y_pred, 'r-', alpha=0.5, 
                       label=f'Weighted (slope={w_slope:.3f})')
            
            if legend:
                ax.legend(loc='best')
        except:
            # Skip regression lines if they can't be calculated
            pass
    
    # Add horizontal line at pooled effect
    ax.axhline(meta.mean_effect, color='r', linestyle=':', alpha=0.6,
              label=f'Pooled Effect ({meta.mean_effect:.3f})')
    
    # Add confidence interval band for pooled effect
    ax.axhspan(meta.ci_lower, meta.ci_upper, color='r', alpha=0.1)
    
    # Set labels and title
    ax.set_xlabel(x_var if isinstance(x_var, str) else "x_var")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend for bubble sizes if requested
    if legend and size_var == 'weight':
        # Create dummy scatter points for the legend
        sizes = [0.25, 0.5, 1.0]
        labels = []
        
        if hasattr(meta, 'random_weights'):
            max_weight = np.max(meta.random_weights)
            for size in sizes:
                weight_pct = size * 100
                labels.append(f"{weight_pct:.0f}% of max weight")
        else:
            for size in sizes:
                labels.append(f"{size:.2f} × max size")
        
        legend_sizes = [s * 500 for s in sizes]
        legend_handles = []
        
        for i, (size, label) in enumerate(zip(legend_sizes, labels)):
            legend_handles.append(ax.scatter([], [], s=size, alpha=0.6, 
                                           color='gray', label=label))
        
        # Add a second legend specifically for bubble sizes
        size_legend = ax.legend(handles=legend_handles, loc='upper left', 
                              title="Bubble Size", bbox_to_anchor=(1.05, 1))
        
        # Add the first legend back
        ax.add_artist(size_legend)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def labbe_plot(meta, treatment_events=None, treatment_total=None, 
              control_events=None, control_total=None, logscale=True,
              figsize=(8, 8), save_path=None, title=None, show_labels=True,
              **kwargs):
    """
    Create a L'Abbé plot for binary outcome meta-analyses.
    
    A L'Abbé plot displays the event rates in the treatment group against 
    the event rates in the control group, with the size of each point 
    proportional to the study size.
    
    Parameters:
    -----------
    meta : MetaAnalysis
        MetaAnalysis object with results
    treatment_events : array-like, optional
        Number of events in treatment group for each study
    treatment_total : array-like, optional
        Total number of participants in treatment group for each study
    control_events : array-like, optional
        Number of events in control group for each study
    control_total : array-like, optional
        Total number of participants in control group for each study
    logscale : bool, default=True
        Whether to use logarithmic scales for the axes
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure to disk
    title : str, optional
        Title for the plot. If None, a default is used
    show_labels : bool, default=True
        Whether to show study names as labels on the plot
    **kwargs : dict
        Additional keyword arguments passed to plt.figure()
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    if meta.tau2 is None:
        raise ValueError("Meta-analysis has not been run yet. Call run() first.")
        
    # Check if binary data was provided
    if any(data is None for data in [treatment_events, treatment_total, 
                                    control_events, control_total]):
        raise ValueError("L'Abbé plot requires binary outcome data (treatment_events, "
                        "treatment_total, control_events, control_total)")
    
    # Convert inputs to arrays
    treatment_events = np.array(treatment_events)
    treatment_total = np.array(treatment_total)
    control_events = np.array(control_events)
    control_total = np.array(control_total)
    
    # Check dimensions
    if len(treatment_events) != meta.k:
        raise ValueError(f"Length of treatment_events ({len(treatment_events)}) must match "
                        f"number of studies ({meta.k})")
    
    # Calculate event rates
    treatment_rate = treatment_events / treatment_total
    control_rate = control_events / control_total
    
    # Calculate relative risks for each study
    relative_risks = treatment_rate / control_rate
    
    # Calculate study sizes for marker scaling
    study_sizes = treatment_total + control_total
    marker_sizes = study_sizes / np.max(study_sizes) * 500
    
    # Default title
    if title is None:
        title = "L'Abbé Plot"
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    
    # Plot diagonal line indicating no difference
    if logscale:
        min_rate = np.min([np.min(treatment_rate), np.min(control_rate)])
        max_rate = np.max([np.max(treatment_rate), np.max(control_rate)])
        
        # Add a small offset to avoid log(0)
        if min_rate <= 0:
            min_rate = 0.5 / np.max([np.max(treatment_total), np.max(control_total)])
        
        # Expand range slightly
        min_rate = max(min_rate * 0.5, 1e-3)
        max_rate = min(max_rate * 1.5, 1.0)
        
        # Set logarithmic scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Plot diagonal line
        diag_line = np.logspace(np.log10(min_rate), np.log10(max_rate), 100)
        ax.plot(diag_line, diag_line, 'k--', alpha=0.5)
        
        # Set axis limits
        ax.set_xlim(min_rate, max_rate)
        ax.set_ylim(min_rate, max_rate)
    else:
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # Plot the studies
    sc = ax.scatter(control_rate, treatment_rate, s=marker_sizes, 
                   alpha=0.7, c=relative_risks, cmap='RdYlGn')
    
    # Add colorbar for relative risks
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Relative Risk')
    
    # Add study labels if requested
    if show_labels:
        for i, name in enumerate(meta.study_names):
            ax.annotate(
                name,
                (control_rate[i], treatment_rate[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
    
    # Plot curves for different relative risks
    if logscale:
        for rr in [0.1, 0.2, 0.5, 2, 5, 10]:
            x_range = np.logspace(np.log10(min_rate), np.log10(max_rate), 100)
            y_range = x_range * rr
            
            # Only plot where y is in range
            valid_idx = (y_range >= min_rate) & (y_range <= max_rate)
            if any(valid_idx):
                ax.plot(x_range[valid_idx], y_range[valid_idx], 'k-', alpha=0.2, linewidth=0.5)
                
                # Label the line at the middle point
                middle_idx = len(valid_idx) // 2
                if valid_idx[middle_idx]:
                    ax.annotate(
                        f"RR={rr}",
                        (x_range[middle_idx], y_range[middle_idx]),
                        fontsize=8,
                        ha='center',
                        va='bottom',
                        alpha=0.7
                    )
    else:
        for rr in [0.25, 0.5, 2, 4]:
            # For non-log scale, only draw where in range
            x_range = np.linspace(0, 1, 100)
            y_range = x_range * rr
            
            # Only plot where y is in range
            valid_idx = (y_range >= 0) & (y_range <= 1)
            if any(valid_idx):
                ax.plot(x_range[valid_idx], y_range[valid_idx], 'k-', alpha=0.2, linewidth=0.5)
                
                # Label the line at a suitable point
                middle_idx = len(valid_idx) // 2
                if valid_idx[middle_idx]:
                    ax.annotate(
                        f"RR={rr}",
                        (x_range[middle_idx], y_range[middle_idx]),
                        fontsize=8,
                        ha='center',
                        va='bottom',
                        alpha=0.7
                    )
    
    # Set labels and title
    ax.set_xlabel('Event Rate in Control Group')
    ax.set_ylabel('Event Rate in Treatment Group')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Create legend for bubble sizes
    sizes = [0.2, 0.5, 1.0]
    legend_sizes = [s * 500 for s in sizes]
    legend_handles = []
    
    max_size = np.max(study_sizes)
    for i, size_fraction in enumerate(sizes):
        actual_size = max_size * size_fraction
        legend_handles.append(ax.scatter([], [], s=legend_sizes[i], 
                                         color='gray', alpha=0.7,
                                         label=f"{actual_size:.0f} participants"))
    
    # Add legend
    ax.legend(handles=legend_handles, loc='upper left', 
              title="Study Size", bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def galbraith_plot(meta, figsize=(10, 10), save_path=None, title=None, 
                effect_label=None, show_outliers=True, z_threshold=2,
                arc_radius=0.9, display_weights=True, **kwargs):
    """
    Create a Galbraith plot (radial plot) for meta-analysis results.
    
    This plot displays the standardized effect sizes against precision, with a radial
    axis indicating the effect size magnitude. It's particularly useful for identifying
    outliers and examining heterogeneity in meta-analyses.
    
    Parameters:
    -----------
    meta : MetaAnalysis
        MetaAnalysis object with results
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure to disk
    title : str, optional
        Title for the plot. If None, a default is used
    effect_label : str, optional
        Label for effect size. If None, uses meta.effect_measure
    show_outliers : bool, default=True
        Whether to highlight potential outlier studies
    z_threshold : float, default=2
        Z-score threshold for identifying outliers (in standardized units)
    arc_radius : float, default=0.9
        Radius of the radial axis arc (in axis coordinates)
    display_weights : bool, default=True
        Whether to scale point sizes by study weights
    **kwargs : dict
        Additional keyword arguments passed to plt.figure()
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Polygon
    from matplotlib import collections as mc
    from matplotlib.ticker import MaxNLocator
    
    if meta.tau2 is None:
        raise ValueError("Meta-analysis has not been run yet. Call run() first.")
    
    if meta.variances is None:
        raise ValueError("Galbraith plot requires study variances")
    
    # Default title and effect label
    if title is None:
        title = "Galbraith Plot (Radial Plot)"
    if effect_label is None:
        effect_label = meta.effect_measure
    
    # Create figure with square aspect ratio
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    
    # Extract data for plotting
    effect_sizes = meta.effect_sizes
    variances = meta.variances
    se = np.sqrt(variances)
    precision = 1 / se
    
    # Standardized effect sizes (z-scores)
    z_scores = (effect_sizes - meta.mean_effect) / se
    
    # Calculate x, y coordinates for the plot
    x = precision
    y = z_scores
    
    # Determine marker sizes based on study weights
    if display_weights and hasattr(meta, 'random_weights'):
        weights = meta.random_weights
        norm_sizes = weights / np.max(weights)
        marker_sizes = 40 + norm_sizes * 100  # Range from 40 to 140
    else:
        marker_sizes = np.ones(meta.k) * 60
    
    # Identify potential outliers
    outliers = np.abs(z_scores) > z_threshold
    
    # Plot the studies with appropriate styling
    if show_outliers and np.any(~outliers):
        # Non-outlier studies
        ax.scatter(
            x[~outliers], 
            y[~outliers], 
            s=marker_sizes[~outliers] if display_weights else 60, 
            c='#3366CC', 
            alpha=0.8, 
            edgecolor='white',
            linewidth=0.5,
            zorder=3,
            label='Studies'
        )
    
    if show_outliers and np.any(outliers):
        # Outlier studies
        ax.scatter(
            x[outliers], 
            y[outliers], 
            s=marker_sizes[outliers] if display_weights else 60, 
            c='#E41A1C', 
            alpha=0.8, 
            edgecolor='white',
            linewidth=0.5,
            zorder=4,
            label='Potential outliers'
        )
    
    if not show_outliers:
        # Plot all studies the same if outlier highlighting is disabled
        ax.scatter(
            x, y, 
            s=marker_sizes if display_weights else 60, 
            c='#3366CC', 
            alpha=0.8, 
            edgecolor='white',
            linewidth=0.5,
            zorder=3
        )
    
    # Add study labels
    for i, name in enumerate(meta.study_names):
        color = '#E41A1C' if show_outliers and outliers[i] else 'black'
        fontweight = 'bold' if show_outliers and outliers[i] else 'normal'
        
        ax.annotate(
            name,
            (x[i], y[i]),
            xytext=(30, 7),
            textcoords='offset points',
            ha="right",
            fontsize=8,
            color=color,
            fontweight=fontweight,
            alpha=0.8,
            zorder=5
        )
    
    # Add horizontal reference lines
    ax.axhline(0, color='k', linestyle='-', alpha=0.3, zorder=1)
    ax.axhline(1.96, color='gray', linestyle='--', alpha=0.6, zorder=1, label='z = ±1.96')
    ax.axhline(-1.96, color='gray', linestyle='--', alpha=0.6, zorder=1)
    
    # Add shaded 95% confidence region
    polygon_y = np.concatenate([np.ones(50) * -1.96, np.ones(50) * 1.96])
    max_x_value = np.max(x) * 1.1
    polygon_x = np.concatenate([np.linspace(0, max_x_value, 50), np.linspace(max_x_value, 0, 50)])
    ax.fill(polygon_x, polygon_y, color='silver', alpha=0.2, zorder=0)
    
    # Set appropriate axis limits with padding
    ax.set_xlim(0, max_x_value)
    y_limit = max(3, np.max(np.abs(y)) * 1.2)
    ax.set_ylim(-y_limit, y_limit)
    
    # Add axis labels
    ax.set_xlabel('Precision (1/SE)', fontsize=12)
    ax.set_ylabel('Standardized Estimate (z)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # Add legend if showing outliers
    if show_outliers and np.any(outliers):
        ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    
    # Create radial axis for effect sizes
    # Calculate the angles based on effect sizes
    def get_angle(effect_size, se_value):
        # Calculate angle based on effect size and SE
        # This is the core of the radial plot transformation
        return np.arctan2(effect_size, 1/se_value)
    
    # Determine range for radial axis
    effect_range = max(effect_sizes) - min(effect_sizes)
    effect_min = min(effect_sizes) - 0.2 * effect_range
    effect_max = max(effect_sizes) + 0.2 * effect_range
    
    # Create tick values for the radial axis
    loc = MaxNLocator(nbins=7)
    radial_ticks = loc.tick_values(effect_min, effect_max)
    
    # Calculate angles for the radial ticks
    angles = []
    for tick in radial_ticks:
        angles.append(np.rad2deg(np.arctan2(tick - meta.mean_effect, 1)))
    
    # Add the radial axis arc
    arc = Arc(
        (0, 0), 
        2 * arc_radius * max_x_value, 
        2 * arc_radius * max_x_value,
        theta1=min(angles),
        theta2=max(angles),
        linewidth=1,
        color='k',
        zorder=2
    )
    ax.add_patch(arc)
    
    # Add tick marks and labels for the radial axis
    for i, (tick, angle) in enumerate(zip(radial_ticks, angles)):
        # Calculate endpoint of the tick mark
        x_start = arc_radius * max_x_value * np.cos(np.deg2rad(angle))
        y_start = arc_radius * max_x_value * np.sin(np.deg2rad(angle))
        x_end = 1.02 * arc_radius * max_x_value * np.cos(np.deg2rad(angle))
        y_end = 1.02 * arc_radius * max_x_value * np.sin(np.deg2rad(angle))
        
        # Draw the tick mark
        ax.plot([x_start, x_end], [y_start, y_end], 'k-', linewidth=1, zorder=2)
        
        # Add the label
        x_label = 1.05 * arc_radius * max_x_value * np.cos(np.deg2rad(angle))
        y_label = 1.05 * arc_radius * max_x_value * np.sin(np.deg2rad(angle))
        ax.text(
            x_label, y_label, 
            f"{tick:.2f}", 
            ha='center', va='center', 
            fontsize=9,
            color='k',
            zorder=2
        )
    
    # Add radial axis label
    ax.text(
        0, 1.15 * arc_radius * max_x_value,
        effect_label,
        ha='center', va='center',
        fontsize=12,
        color='k',
        zorder=2
    )
    
    # Draw lines from origin to each study for better visualization
    for i in range(meta.k):
        angle = np.arctan2(effect_sizes[i] - meta.mean_effect, precision[i])
        # Calculate line endpoint based on point's radial distance
        line_length = np.sqrt(precision[i]**2 + ((effect_sizes[i] - meta.mean_effect) * precision[i])**2)
        end_x = line_length * np.cos(angle)
        end_y = line_length * np.sin(angle)
        
        # Plot the line with very light opacity
        ax.plot([0, end_x], [0, end_y], 'k-', alpha=0.1, zorder=1)
    
    # Add annotation for heterogeneity statistics
    stats_text = (
        f"Heterogeneity: I² = {meta.i2:.1f}%, "
        f"τ² = {meta.tau2:.3f}, "
        f"Q = {meta.q:.2f} (p = {meta.p_value:.3f})"
    )
    
    # Add text box in bottom right
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5),
        zorder=5
    )
    
    # Add explanation of the plot
    explanation = (
        "Radial distance: Effect size\n"
        "Angular position: Standardized effect\n"
        "Points outside ±1.96 may contribute to heterogeneity"
    )
    
    ax.text(
        0.02, 0.98, explanation,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=8, style='italic',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5),
        zorder=5
    )
    
    # Add marker size legend if using weights
    if display_weights and hasattr(meta, 'random_weights'):
        # Create dummy scatter points for the legend
        sizes = [0.25, 0.5, 1.0]
        labels = []
        
        for size in sizes:
            weight_pct = size * 100
            labels.append(f"{weight_pct:.0f}% of max weight")
        
        legend_sizes = [40 + s * 100 for s in sizes]
        legend_handles = []
        
        for i, (size, label) in enumerate(zip(legend_sizes, labels)):
            legend_handles.append(ax.scatter([], [], s=size, alpha=0.7, 
                                          color='gray', label=label))
        
        # Add the legend
        weight_legend = ax.legend(
            handles=legend_handles, 
            loc='upper center', 
            title="Study Weight",
            bbox_to_anchor=(1.0, 0.9),
            frameon=True,
            framealpha=0.9
        )
        
        # If outlier legend exists, add it back
        if show_outliers and np.any(outliers):
            first_legend = ax.legend(loc='upper center')
            ax.add_artist(weight_legend)
            ax.add_artist(first_legend)
    
    # Adjust layout to ensure everything fits
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


