#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities for consensus benchmarking results.

This module provides functions to create visualizations for comparing
single model performance against consensus approaches and exporting
the results in various formats.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec


def plot_comparison(df, metric, title=None, figsize=(10, 6)):
    """
    Create a clean comparative visualization between single models and consensus approach.
    
    This function creates a bar chart comparing the performance of different models
    and the consensus approach on a specific metric.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the benchmark results with model names as index
        and metrics as columns.
    metric : str
        Name of the metric to plot (should be a column in df).
    title : str, optional
        Title for the plot. If None, uses the metric name.
    figsize : tuple, optional
        Size of the figure (width, height) in inches.
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
        
    Examples
    --------
    >>> results_df = pd.DataFrame({
    ...     'accuracy': [0.78, 0.82, 0.85, 0.91],
    ...     'latency': [1.2, 0.9, 1.5, 1.1]
    ... }, index=['Model A', 'Model B', 'Model C', 'Consensus'])
    >>> fig = plot_comparison(results_df, 'accuracy')
    >>> fig.savefig('accuracy_comparison.png')
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create color palette with consensus highlighted
    palette = sns.color_palette("Blues_d", len(df))
    if 'consensus' in df.index.str.lower():
        # Highlight the consensus model with a distinct color
        colors = palette.copy()
        consensus_idx = df.index.str.lower().tolist().index('consensus')
        colors[consensus_idx] = sns.color_palette("Reds_d")[3]
    else:
        colors = palette
    
    # Create the bar plot
    bars = sns.barplot(x=df.index, y=df[metric], palette=colors, ax=ax)
    
    # Annotate bars with values
    for i, bar in enumerate(bars.patches):
        value = df[metric].iloc[i]
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(df[metric]) * 0.02,
            f'{value:.2f}',
            ha='center', va='bottom',
            fontsize=10
        )
    
    # Set title and labels
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    else:
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14, pad=20)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    
    # Rotate x labels if there are many models
    if len(df) > 4:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def create_dashboard(df, metrics, title="Model Performance Dashboard", figsize=(15, 10)):
    """
    Create a single-page dashboard overview of all key metrics.
    
    This function creates a comprehensive dashboard with multiple visualizations
    of different metrics to compare model performances.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the benchmark results with model names as index
        and metrics as columns.
    metrics : list
        List of metric names to include in the dashboard (should be columns in df).
    title : str, optional
        Title for the dashboard.
    figsize : tuple, optional
        Size of the figure (width, height) in inches.
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
        
    Examples
    --------
    >>> results_df = pd.DataFrame({
    ...     'accuracy': [0.78, 0.82, 0.85, 0.91],
    ...     'hallucination_rate': [0.15, 0.12, 0.09, 0.05],
    ...     'latency': [1.2, 0.9, 1.5, 1.1],
    ...     'factual_consistency': [0.81, 0.84, 0.88, 0.92]
    ... }, index=['Model A', 'Model B', 'Model C', 'Consensus'])
    >>> metrics_to_plot = ['accuracy', 'hallucination_rate', 'factual_consistency']
    >>> fig = create_dashboard(results_df, metrics_to_plot)
    >>> fig.savefig('performance_dashboard.png', dpi=300)
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Calculate grid dimensions based on number of metrics
    n_metrics = len(metrics)
    n_rows = max(1, (n_metrics + 1) // 2)  # Add radar chart 
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows + 1, 2, height_ratios=[0.2] + [1] * n_rows)
    
    # Add title to the dashboard
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Create radar chart for overview in the first row (spanning both columns)
    ax_radar = fig.add_subplot(gs[0, :], polar=True)
    _create_radar_chart(df, metrics, ax_radar)
    
    # Create individual bar plots for each metric
    for i, metric in enumerate(metrics):
        row = 1 + i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Create the bar plot
        palette = sns.color_palette("Blues_d", len(df))
        if 'consensus' in df.index.str.lower():
            # Highlight the consensus model
            colors = palette.copy()
            consensus_idx = df.index.str.lower().tolist().index('consensus')
            colors[consensus_idx] = sns.color_palette("Reds_d")[3]
        else:
            colors = palette
            
        bars = sns.barplot(x=df.index, y=df[metric], palette=colors, ax=ax)
        
        # Annotate bars with values
        for j, bar in enumerate(bars.patches):
            value = df[metric].iloc[j]
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(df[metric]) * 0.02,
                f'{value:.2f}',
                ha='center', va='bottom',
                fontsize=9
            )
        
        # Set title and labels
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
        ax.set_xlabel('')
        
        # Rotate x labels if there are many models
        if len(df) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    return fig


def _create_radar_chart(df, metrics, ax):
    """
    Helper function to create a radar chart for the dashboard.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the benchmark results.
    metrics : list
        List of metric names to include in the radar chart.
    ax : matplotlib.axes.Axes
        The axes to draw the radar chart on.
    """
    # Number of metrics (categories)
    N = len(metrics)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Normalize metrics to 0-1 scale for radar chart
    df_normalized = df.copy()
    for metric in metrics:
        if metric.lower() in ['error_rate', 'hallucination_rate', 'latency']:
            # For metrics where lower is better, invert the normalization
            df_normalized[metric] = 1 - ((df[metric] - df[metric].min()) / 
                                         (df[metric].max() - df[metric].min() + 1e-10))
        else:
            # For metrics where higher is better
            df_normalized[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min() + 1e-10)
    
    # Plot each model
    for i, model in enumerate(df.index):
        values = df_normalized.loc[model, metrics].tolist()
        values += values[:1]  # Close the loop
        
        # Choose color based on whether it's the consensus model
        if 'consensus' in model.lower():
            color = sns.color_palette("Reds_d")[3]
            linewidth = 2.5
            alpha = 0.85
        else:
            color = sns.color_palette("Blues_d")[i]
            linewidth = 1.5
            alpha = 0.7
        
        # Plot values
        ax.plot(angles, values, linewidth=linewidth, linestyle='solid', label=model, color=color, alpha=alpha)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    # Add metric labels to the radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=9)
    
    # Remove radial labels and set limits
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=9)


def export_plots(fig, name, output_dir='plots', formats=None, dpi=300):
    """
    Utility for saving plots in different formats.
    
    This function saves a matplotlib figure in multiple formats.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    name : str
        Base name for the saved file (without extension).
    output_dir : str, optional
        Directory to save the plots to. If it doesn't exist, it will be created.
    formats : list, optional
        List of formats to save the plot in. Defaults to ['png', 'pdf'].
    dpi : int, optional
        Resolution for raster formats like PNG.
        
    Returns
    -------
    list
        List of paths to the saved files.
        
    Examples
    --------
    >>> results_df = pd.DataFrame({
    ...     'accuracy': [0.78, 0.82, 0.85, 0.91]
    ... }, index=['Model A', 'Model B', 'Model C', 'Consensus'])
    >>> fig = plot_comparison(results_df, 'accuracy', 'Model Accuracy Comparison')
    >>> paths = export_plots(fig, 'accuracy_comparison', formats=['png', 'pdf', 'svg'])
    >>> print(f"Plots saved to: {', '.join(paths)}")
    """
    if formats is None:
        formats = ['png', 'pdf']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    for fmt in formats:
        filename = f"{name}.{fmt}"
        filepath = os.path.join(output_dir, filename)
        
        # Save with appropriate settings for each format
        if fmt in ['png', 'jpg']:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(filepath, bbox_inches='tight')
        
        saved_paths.append(filepath)
        print(f"Saved plot as {filepath}")
    
    return saved_paths


if __name__ == "__main__":
    # Example usage
    # Create sample data
    example_data = {
        'accuracy': [0.78, 0.82, 0.85, 0.91],
        'hallucination_rate': [0.15, 0.12, 0.09, 0.05],
        'latency': [1.2, 0.9, 1.5, 1.1],
        'factual_consistency': [0.81, 0.84, 0.88, 0.92],
        'reasoning_quality': [0.75, 0.79, 0.83, 0.89]
    }
    
    results_df = pd.DataFrame(example_data, index=['Model A', 'Model B', 'Model C', 'Consensus'])
    
    # Example 1: Plot comparison for a single metric
    fig1 = plot_comparison(results_df, 'accuracy', 'Model Accuracy Comparison')
    export_plots(fig1, 'accuracy_comparison')
    
    # Example 2: Create a dashboard with multiple metrics
    metrics_to_plot = ['accuracy', 'hallucination_rate', 'factual_consistency', 'reasoning_quality']
    fig2 = create_dashboard(results_df, metrics_to_plot, 'Model Performance Dashboard')
    export_plots(fig2, 'performance_dashboard', formats=['png', 'pdf', 'svg'])
    
    plt.show()

