#!/usr/bin/env python3
"""
Create visualization for significance test results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set up paths
OUTPUT_DIR = Path(__file__).parent

def main():
    """Create significance test visualization."""

    # Load significance test results
    csv_path = OUTPUT_DIR / 'significance_tests.csv'
    df = pd.read_csv(csv_path)

    # Filter to key comparisons in left-to-right order
    key_comparisons = [
        'SA vs base',
        'MAS vs SA',
        'MAP vs MAS',
        'MAS vs base',
        'MAP vs base',
        'MAP vs SA'
    ]

    # Metrics order
    metrics = ['CRS', 'OCRS', 'PCRS', 'ACRS', 'LCRS']

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Increase font sizes globally
    plt.rcParams.update({'font.size': 14})

    # --- Plot 1: Mean Differences Heatmap ---
    ax1 = axes[0]

    # Prepare data for heatmap
    mean_diff_matrix = np.zeros((len(metrics), len(key_comparisons)))
    sig_matrix = np.zeros((len(metrics), len(key_comparisons)))  # For marking significance

    for i, metric in enumerate(metrics):
        for j, comp in enumerate(key_comparisons):
            row = df[(df['Comparison'] == comp) & (df['Metric'] == metric)]
            if len(row) > 0:
                mean_diff_matrix[i, j] = row['Mean_Diff'].values[0]
                sig_matrix[i, j] = row['p_value'].values[0]

    # Create heatmap
    im1 = ax1.imshow(mean_diff_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.015, vmax=0.015)

    # Set ticks and labels
    ax1.set_xticks(np.arange(len(key_comparisons)))
    ax1.set_yticks(np.arange(len(metrics)))
    ax1.set_xticklabels(key_comparisons, rotation=45, ha='right', fontsize=14)
    ax1.set_yticklabels(metrics, fontsize=16)

    # Add text annotations with significance markers
    for i in range(len(metrics)):
        for j in range(len(key_comparisons)):
            value = mean_diff_matrix[i, j]
            p_val = sig_matrix[i, j]

            # Determine significance marker
            if p_val < 0.001:
                sig_marker = '***'
            elif p_val < 0.01:
                sig_marker = '**'
            elif p_val < 0.05:
                sig_marker = '*'
            else:
                sig_marker = ''

            # Choose text color based on background
            text_color = 'white' if abs(value) > 0.008 else 'black'

            text = f'{value:+.4f}\n{sig_marker}'
            ax1.text(j, i, text, ha='center', va='center',
                    color=text_color, fontsize=11, fontweight='bold')

    # Colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Mean Difference', fontsize=16, fontweight='bold')
    cbar1.ax.tick_params(labelsize=14)

    ax1.set_title('Mean Differences (Right - Left)', fontsize=18, fontweight='bold', pad=15)
    ax1.set_xlabel('Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Metric', fontsize=16, fontweight='bold')

    # --- Plot 2: Effect Sizes (Cohen's d) Heatmap ---
    ax2 = axes[1]

    # Prepare effect size data
    effect_size_matrix = np.zeros((len(metrics), len(key_comparisons)))

    for i, metric in enumerate(metrics):
        for j, comp in enumerate(key_comparisons):
            row = df[(df['Comparison'] == comp) & (df['Metric'] == metric)]
            if len(row) > 0:
                effect_size_matrix[i, j] = row['Cohen_d'].values[0]

    # Create heatmap
    im2 = ax2.imshow(effect_size_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.7, vmax=0.7)

    # Set ticks and labels
    ax2.set_xticks(np.arange(len(key_comparisons)))
    ax2.set_yticks(np.arange(len(metrics)))
    ax2.set_xticklabels(key_comparisons, rotation=45, ha='right', fontsize=14)
    ax2.set_yticklabels(metrics, fontsize=16)

    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(key_comparisons)):
            value = effect_size_matrix[i, j]
            p_val = sig_matrix[i, j]

            # Determine significance marker
            if p_val < 0.001:
                sig_marker = '***'
            elif p_val < 0.01:
                sig_marker = '**'
            elif p_val < 0.05:
                sig_marker = '*'
            else:
                sig_marker = ''

            # Choose text color based on background
            text_color = 'white' if abs(value) > 0.4 else 'black'

            text = f'{value:+.3f}\n{sig_marker}'
            ax2.text(j, i, text, ha='center', va='center',
                    color=text_color, fontsize=11, fontweight='bold')

    # Colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Cohen's d (Effect Size)", fontsize=16, fontweight='bold')
    cbar2.ax.tick_params(labelsize=14)

    ax2.set_title("Effect Sizes (Cohen's d)", fontsize=18, fontweight='bold', pad=15)
    ax2.set_xlabel('Comparison', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Metric', fontsize=16, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'significance_heatmap.pdf'
    png_path = OUTPUT_DIR / 'significance_heatmap.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"Saved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")

    plt.close()

    print("\nSignificance markers: *** p<0.001, ** p<0.01, * p<0.05")

if __name__ == '__main__':
    main()
