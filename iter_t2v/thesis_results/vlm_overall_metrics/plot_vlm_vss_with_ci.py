#!/usr/bin/env python3
"""
Create VLM-based VSS (Visual Similarity Score) plot with confidence intervals.
Only for agent pipelines: SA, MAS, MAP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent

# File paths (only agent pipelines)
FILES = {
    'SA': BASE_DIR / 'metrics_single.csv',
    'MAS': BASE_DIR / 'metrics_sequential.csv',
    'MAP': BASE_DIR / 'metrics_parallel.csv'
}

# VLM VSS column name (same for all agent pipelines)
VLM_VSS_COL = 'vlm_visual_similarity_score'

def compute_vlm_vss_with_ci(df, confidence=0.95):
    """Compute VLM VSS with confidence intervals."""

    values = df[VLM_VSS_COL].values
    mean = np.mean(values)
    sem = stats.sem(values)  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(values) - 1)

    return {
        'mean': mean,
        'ci': ci,
        'lower': mean - ci,
        'upper': mean + ci
    }

def main():
    """Create VLM VSS plot with confidence intervals."""

    # Load data and compute metrics
    all_metrics = {}
    for pipeline_name, filepath in FILES.items():
        print(f"Processing {pipeline_name}...")
        df = pd.read_csv(filepath)
        all_metrics[pipeline_name] = compute_vlm_vss_with_ci(df)

    # Prepare data for plotting (left to right order)
    pipelines = ['SA', 'MAS', 'MAP']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Increase font sizes globally
    plt.rcParams.update({'font.size': 16})

    # Bar settings
    x = np.arange(len(pipelines))
    width = 0.6
    color = '#1f77b4'  # Blue color from tab10

    # Extract data
    means = [all_metrics[p]['mean'] for p in pipelines]
    cis = [all_metrics[p]['ci'] for p in pipelines]

    # Plot bars with error bars
    bars = ax.bar(x, means, width,
                  yerr=cis, capsize=8,
                  color=color, alpha=0.8,
                  edgecolor='black', linewidth=0.5)

    # Add numerical values on top of bars
    for i, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + ci + 0.05,
               f'{mean:.2f}',
               ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Customize plot
    ax.set_xlabel('Pipeline', fontsize=18, fontweight='bold')
    ax.set_ylabel('VLM Visual Similarity Score', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(means) + max(cis) + 0.5)

    # Tight layout
    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'vlm_vss_with_ci.pdf'
    png_path = OUTPUT_DIR / 'vlm_vss_with_ci.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"\nSaved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")

    # Also save the data to CSV
    csv_data = []
    for pipeline in pipelines:
        csv_data.append({
            'Pipeline': pipeline,
            'Mean': all_metrics[pipeline]['mean'],
            'CI': all_metrics[pipeline]['ci'],
            'Lower': all_metrics[pipeline]['lower'],
            'Upper': all_metrics[pipeline]['upper']
        })

    csv_df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / 'vlm_vss_with_ci.csv'
    csv_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Saved CSV to: {csv_path}")

    # Print results
    print("\n" + "="*60)
    print("VLM Visual Similarity Score (VLM_VSS) Summary")
    print("="*60)
    print(csv_df.to_string(index=False))
    print("="*60)

    plt.close()

if __name__ == '__main__':
    main()
