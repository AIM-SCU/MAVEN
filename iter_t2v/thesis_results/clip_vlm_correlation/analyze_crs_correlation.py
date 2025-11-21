#!/usr/bin/env python3
"""
Analyze correlation between CLIP-based CRS and VLM-judged CRS.
Includes overall CRS and 4 dimensions (OCRS, PCRS, ACRS, LCRS).
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

# Column mappings
CLIP_COLS = {
    'OCRS': 'overall_avg',
    'PCRS': 'person_avg',
    'ACRS': 'action_avg',
    'LCRS': 'location_avg'
}

def get_vlm_cols(pipeline_name):
    """Get VLM column names based on pipeline."""
    if pipeline_name == 'base':
        return {
            'OCRS': 'vlm_overall_score',
            'PCRS': 'vlm_person_score',
            'ACRS': 'vlm_action_score',
            'LCRS': 'vlm_location_score'
        }
    else:
        return {
            'OCRS': 'vlm_v_var_overall_score',
            'PCRS': 'vlm_v_var_person_score',
            'ACRS': 'vlm_v_var_action_score',
            'LCRS': 'vlm_v_var_location_score'
        }

def compute_correlations(clip_values, vlm_values):
    """Compute Pearson correlation."""
    pearson_r, pearson_p = stats.pearsonr(clip_values, vlm_values)

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p
    }

def main():
    """Analyze CRS correlation between CLIP and VLM."""

    print("="*80)
    print("CLIP vs VLM Cultural Relevance Score (CRS) Correlation Analysis")
    print("="*80)

    # Collect all data
    all_data = []
    correlation_results = []

    pipelines = ['base', 'SA', 'MAS', 'MAP']
    dimensions = ['OCRS', 'PCRS', 'ACRS', 'LCRS']

    for pipeline in pipelines:
        print(f"\nProcessing {pipeline}...")
        df = pd.read_csv(FILES[pipeline])
        vlm_cols = get_vlm_cols(pipeline)

        # Extract CLIP scores
        clip_ocrs = df[CLIP_COLS['OCRS']].values
        clip_pcrs = df[CLIP_COLS['PCRS']].values
        clip_acrs = df[CLIP_COLS['ACRS']].values
        clip_lcrs = df[CLIP_COLS['LCRS']].values
        clip_crs = df[[CLIP_COLS['OCRS'], CLIP_COLS['PCRS'],
                      CLIP_COLS['ACRS'], CLIP_COLS['LCRS']]].mean(axis=1).values

        # Extract VLM scores
        vlm_ocrs = df[vlm_cols['OCRS']].values
        vlm_pcrs = df[vlm_cols['PCRS']].values
        vlm_acrs = df[vlm_cols['ACRS']].values
        vlm_lcrs = df[vlm_cols['LCRS']].values
        vlm_crs = df[[vlm_cols['OCRS'], vlm_cols['PCRS'],
                     vlm_cols['ACRS'], vlm_cols['LCRS']]].mean(axis=1).values

        # Store data for plotting
        for i in range(len(clip_crs)):
            all_data.append({
                'pipeline': pipeline,
                'clip_ocrs': clip_ocrs[i],
                'clip_pcrs': clip_pcrs[i],
                'clip_acrs': clip_acrs[i],
                'clip_lcrs': clip_lcrs[i],
                'clip_crs': clip_crs[i],
                'vlm_ocrs': vlm_ocrs[i],
                'vlm_pcrs': vlm_pcrs[i],
                'vlm_acrs': vlm_acrs[i],
                'vlm_lcrs': vlm_lcrs[i],
                'vlm_crs': vlm_crs[i]
            })

        # Compute correlations for each dimension
        for dim in dimensions:
            clip_col = f'clip_{dim.lower()}'
            vlm_col = f'vlm_{dim.lower()}'

            if dim == 'CRS':
                clip_vals = clip_crs
                vlm_vals = vlm_crs
            else:
                clip_vals = df[CLIP_COLS[dim]].values
                vlm_vals = df[vlm_cols[dim]].values

            corr = compute_correlations(clip_vals, vlm_vals)

            correlation_results.append({
                'Pipeline': pipeline,
                'Dimension': dim,
                'Pearson_r': corr['pearson_r'],
                'Pearson_p': corr['pearson_p'],
                'N': len(clip_vals)
            })

            print(f"  {dim}: Pearson r={corr['pearson_r']:.4f} (p={corr['pearson_p']:.4e})")

    # Convert to DataFrame
    all_data_df = pd.DataFrame(all_data)
    corr_results_df = pd.DataFrame(correlation_results)

    # Save correlation results
    corr_csv_path = OUTPUT_DIR / 'crs_correlation_results.csv'
    corr_results_df.to_csv(corr_csv_path, index=False, float_format='%.6f')
    print(f"\n✓ Saved correlation results to: {corr_csv_path}")

    # Create scatter plots for each dimension
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    colors = {'base': '#1f77b4', 'SA': '#ff7f0e', 'MAS': '#2ca02c', 'MAP': '#d62728'}

    for idx, dim in enumerate(dimensions):
        ax = axes[idx]
        clip_col = f'clip_{dim.lower()}'
        vlm_col = f'vlm_{dim.lower()}'

        # Compute correlations for each pipeline
        pipeline_corrs = {}
        for pipeline in pipelines:
            pipeline_data = all_data_df[all_data_df['pipeline'] == pipeline]
            x = pipeline_data[clip_col].values
            y = pipeline_data[vlm_col].values

            # Compute correlation for this pipeline
            corr = compute_correlations(x, y)
            pipeline_corrs[pipeline] = corr['pearson_r']

            # Scatter plot
            ax.scatter(x, y, alpha=0.5, s=30, c=colors[pipeline], label=pipeline)

            # Add regression line for this pipeline
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), color=colors[pipeline],
                       linestyle='--', alpha=0.7, linewidth=2)

        # Create title with all Pearson r values
        title_text = f'{dim}\n'
        title_text += ', '.join([f'{p}: r={pipeline_corrs[p]:.3f}' for p in pipelines])

        # Labels and title
        ax.set_xlabel(f'CLIP {dim}', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'VLM {dim}', fontsize=14, fontweight='bold')
        ax.set_title(title_text, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'crs_correlation_scatter.pdf'
    png_path = OUTPUT_DIR / 'crs_correlation_scatter.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"✓ Saved scatter plots to: {pdf_path}")
    print(f"✓ Saved scatter plots to: {png_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)

    for dim in dimensions:
        dim_corr = corr_results_df[corr_results_df['Dimension'] == dim]
        print(f"\n{dim}:")
        print(f"  Mean Pearson r: {dim_corr['Pearson_r'].mean():.4f}")
        print(f"  Min Pearson r: {dim_corr['Pearson_r'].min():.4f} ({dim_corr.loc[dim_corr['Pearson_r'].idxmin(), 'Pipeline']})")
        print(f"  Max Pearson r: {dim_corr['Pearson_r'].max():.4f} ({dim_corr.loc[dim_corr['Pearson_r'].idxmax(), 'Pipeline']})")
        # Print individual pipeline values
        print(f"  Pipeline values:")
        for pipeline in pipelines:
            pipeline_r = dim_corr[dim_corr['Pipeline'] == pipeline]['Pearson_r'].values[0]
            print(f"    {pipeline}: r={pipeline_r:.4f}")

    plt.close()
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    main()
