#!/usr/bin/env python3
"""
Create a focused CRS plot with confidence intervals and numerical values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent

# File paths
FILES = {
    'base': BASE_DIR / 'metrics_base.csv',
    'SA': BASE_DIR / 'metrics_single.csv',
    'MAP': BASE_DIR / 'metrics_parallel.csv',
    'MAS': BASE_DIR / 'metrics_sequential.csv'
}

def compute_crs_with_ci(df, confidence=0.95):
    """Compute CRS metrics with confidence intervals."""

    metrics = {}
    crs_cols = ['overall_avg', 'person_avg', 'action_avg', 'location_avg']
    metric_names = ['OCRS', 'PCRS', 'ACRS', 'LCRS']

    for col, name in zip(crs_cols, metric_names):
        values = df[col].values
        mean = np.mean(values)
        sem = stats.sem(values)  # Standard error of the mean
        ci = sem * stats.t.ppf((1 + confidence) / 2, len(values) - 1)

        metrics[name] = {
            'mean': mean,
            'ci': ci,
            'lower': mean - ci,
            'upper': mean + ci
        }

    # Compute CRS as average of the four
    crs_values = df[crs_cols].mean(axis=1).values
    crs_mean = np.mean(crs_values)
    crs_sem = stats.sem(crs_values)
    crs_ci = crs_sem * stats.t.ppf((1 + confidence) / 2, len(crs_values) - 1)

    metrics['CRS'] = {
        'mean': crs_mean,
        'ci': crs_ci,
        'lower': crs_mean - crs_ci,
        'upper': crs_mean + crs_ci
    }

    return metrics

def main():
    """Create CRS plot with confidence intervals."""

    # Load data and compute metrics
    all_metrics = {}
    for pipeline_name, filepath in FILES.items():
        print(f"Processing {pipeline_name}...")
        df = pd.read_csv(filepath)
        all_metrics[pipeline_name] = compute_crs_with_ci(df)

    # Prepare data for plotting (left to right order)
    pipelines = ['base', 'SA', 'MAS', 'MAP']
    metric_names = ['CRS', 'OCRS', 'PCRS', 'ACRS', 'LCRS']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Increase font sizes globally
    plt.rcParams.update({'font.size': 16})

    # Bar settings
    x = np.arange(len(pipelines))
    width = 0.15
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot bars with error bars and values
    for i, metric in enumerate(metric_names):
        means = [all_metrics[p][metric]['mean'] for p in pipelines]
        cis = [all_metrics[p][metric]['ci'] for p in pipelines]

        bars = ax.bar(x + i * width, means, width,
                      yerr=cis, capsize=4,
                      label=metric, color=colors[i],
                      alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add numerical values on top of bars
        for j, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + ci + 0.005,
                   f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=9.5, rotation=0)

    # Customize plot
    ax.set_xlabel('Pipeline', fontsize=18, fontweight='bold')
    ax.set_ylabel('Cultural Relevance Score (CRS)', fontsize=18, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(pipelines, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=16, framealpha=0.9, ncol=5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max([all_metrics[p][m]['mean'] + all_metrics[p][m]['ci']
                        for p in pipelines for m in metric_names]) * 1.15)

    # Tight layout
    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'crs_with_ci.pdf'
    png_path = OUTPUT_DIR / 'crs_with_ci.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"\nSaved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")

    # Also save the data to CSV
    csv_data = []
    for pipeline in pipelines:
        for metric in metric_names:
            csv_data.append({
                'Pipeline': pipeline,
                'Metric': metric,
                'Mean': all_metrics[pipeline][metric]['mean'],
                'CI': all_metrics[pipeline][metric]['ci'],
                'Lower': all_metrics[pipeline][metric]['lower'],
                'Upper': all_metrics[pipeline][metric]['upper']
            })

    csv_df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / 'crs_with_ci.csv'
    csv_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Saved CSV to: {csv_path}")

    plt.close()

if __name__ == '__main__':
    main()
