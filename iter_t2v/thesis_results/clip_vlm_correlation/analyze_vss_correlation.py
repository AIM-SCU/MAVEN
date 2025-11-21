#!/usr/bin/env python3
"""
Analyze correlation between CLIP-based VSS and VLM-judged VSS.
Only for agent pipelines: SA, MAS, MAP (base doesn't have VSS).
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

# VSS column mapping
CLIP_VSS_COLS = {
    'SA': 'base_vs_single_avg',
    'MAS': 'base_vs_sequential_avg',
    'MAP': 'base_vs_parallel_avg'
}

VLM_VSS_COL = 'vlm_visual_similarity_score'

def compute_correlation(clip_values, vlm_values):
    """Compute Pearson correlation."""
    pearson_r, pearson_p = stats.pearsonr(clip_values, vlm_values)
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p
    }

def main():
    """Analyze VSS correlation between CLIP and VLM."""

    print("="*80)
    print("CLIP vs VLM Visual Similarity Score (VSS) Correlation Analysis")
    print("="*80)

    # Collect all data
    all_data = []
    correlation_results = []

    pipelines = ['SA', 'MAS', 'MAP']

    for pipeline in pipelines:
        print(f"\nProcessing {pipeline}...")
        df = pd.read_csv(FILES[pipeline])

        # Extract CLIP VSS
        clip_vss = df[CLIP_VSS_COLS[pipeline]].values

        # Extract VLM VSS
        vlm_vss = df[VLM_VSS_COL].values

        # Store data for plotting
        for i in range(len(clip_vss)):
            all_data.append({
                'pipeline': pipeline,
                'clip_vss': clip_vss[i],
                'vlm_vss': vlm_vss[i]
            })

        # Compute correlation
        corr = compute_correlation(clip_vss, vlm_vss)

        correlation_results.append({
            'Pipeline': pipeline,
            'Pearson_r': corr['pearson_r'],
            'Pearson_p': corr['pearson_p'],
            'N': len(clip_vss)
        })

        print(f"  VSS: Pearson r={corr['pearson_r']:.4f} (p={corr['pearson_p']:.4e})")

    # Convert to DataFrame
    all_data_df = pd.DataFrame(all_data)
    corr_results_df = pd.DataFrame(correlation_results)

    # Save correlation results
    corr_csv_path = OUTPUT_DIR / 'vss_correlation_results.csv'
    corr_results_df.to_csv(corr_csv_path, index=False, float_format='%.6f')
    print(f"\n✓ Saved correlation results to: {corr_csv_path}")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {'SA': '#ff7f0e', 'MAS': '#2ca02c', 'MAP': '#d62728'}

    # Compute correlations for each pipeline
    pipeline_corrs = {}
    for pipeline in pipelines:
        pipeline_data = all_data_df[all_data_df['pipeline'] == pipeline]
        x = pipeline_data['clip_vss'].values
        y = pipeline_data['vlm_vss'].values

        # Compute correlation for this pipeline
        corr = compute_correlation(x, y)
        pipeline_corrs[pipeline] = corr['pearson_r']

        # Scatter plot
        ax.scatter(x, y, alpha=0.5, s=50, c=colors[pipeline], label=pipeline)

        # Add regression line for this pipeline
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), color=colors[pipeline],
                   linestyle='--', alpha=0.7, linewidth=2)

    # Create title with all Pearson r values
    title_text = 'VSS\n'
    title_text += ', '.join([f'{p}: r={pipeline_corrs[p]:.3f}' for p in pipelines])

    # Labels and title
    ax.set_xlabel('CLIP VSS', fontsize=16, fontweight='bold')
    ax.set_ylabel('VLM VSS', fontsize=16, fontweight='bold')
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12)

    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'vss_correlation_scatter.pdf'
    png_path = OUTPUT_DIR / 'vss_correlation_scatter.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"✓ Saved scatter plot to: {pdf_path}")
    print(f"✓ Saved scatter plot to: {png_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(f"\nVSS:")
    print(f"  Mean Pearson r: {corr_results_df['Pearson_r'].mean():.4f}")
    print(f"  Min Pearson r: {corr_results_df['Pearson_r'].min():.4f} "
          f"({corr_results_df.loc[corr_results_df['Pearson_r'].idxmin(), 'Pipeline']})")
    print(f"  Max Pearson r: {corr_results_df['Pearson_r'].max():.4f} "
          f"({corr_results_df.loc[corr_results_df['Pearson_r'].idxmax(), 'Pipeline']})")
    print(f"  Pipeline values:")
    for pipeline in pipelines:
        pipeline_r = corr_results_df[corr_results_df['Pipeline'] == pipeline]['Pearson_r'].values[0]
        print(f"    {pipeline}: r={pipeline_r:.4f}")

    plt.close()
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    main()
