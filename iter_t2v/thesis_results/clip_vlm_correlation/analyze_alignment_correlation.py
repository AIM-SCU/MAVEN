#!/usr/bin/env python3
"""
Analyze correlation between CLIP-based and VLM-judged alignment scores.
Two types: AS_orig (alignment with original prompt) and AS_final (alignment with final prompt).
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
    'MAS': BASE_DIR / 'metrics_sequential.csv',
    'MAP': BASE_DIR / 'metrics_parallel.csv'
}

def get_alignment_cols(pipeline_name):
    """Get alignment column names based on pipeline."""
    if pipeline_name == 'base':
        return {
            'clip_as_orig': 'v_base_to_original_prompt_avg_avg',
            'clip_as_final': 'v_base_to_final_prompt_avg_avg',
            'vlm_as_orig': 'vlm_v_base_original_prompt_score',
            'vlm_as_final': 'vlm_v_base_final_prompt_score'
        }
    else:
        return {
            'clip_as_orig': 'v_var_to_original_prompt_avg_avg',
            'clip_as_final': 'v_var_to_final_prompt_avg_avg',
            'vlm_as_orig': 'vlm_v_var_original_prompt_score',
            'vlm_as_final': 'vlm_v_var_final_prompt_score'
        }

def compute_correlation(clip_values, vlm_values):
    """Compute Pearson correlation."""
    pearson_r, pearson_p = stats.pearsonr(clip_values, vlm_values)
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p
    }

def main():
    """Analyze alignment correlation between CLIP and VLM."""

    print("="*80)
    print("CLIP vs VLM Alignment Score Correlation Analysis")
    print("="*80)

    # Collect all data
    all_data = []
    correlation_results = []

    pipelines = ['base', 'SA', 'MAS', 'MAP']
    alignment_types = ['AS_orig', 'AS_final']

    for pipeline in pipelines:
        print(f"\nProcessing {pipeline}...")

        # Load appropriate data
        if pipeline == 'base':
            df = pd.read_csv(FILES['SA'])  # base alignment scores in agent CSVs
        else:
            df = pd.read_csv(FILES[pipeline])

        cols = get_alignment_cols(pipeline)

        # Extract scores
        clip_as_orig = df[cols['clip_as_orig']].values
        clip_as_final = df[cols['clip_as_final']].values
        vlm_as_orig = df[cols['vlm_as_orig']].values
        vlm_as_final = df[cols['vlm_as_final']].values

        # Store data for plotting
        for i in range(len(clip_as_orig)):
            all_data.append({
                'pipeline': pipeline,
                'clip_as_orig': clip_as_orig[i],
                'clip_as_final': clip_as_final[i],
                'vlm_as_orig': vlm_as_orig[i],
                'vlm_as_final': vlm_as_final[i]
            })

        # Compute correlations
        for align_type in alignment_types:
            clip_col_key = f'clip_as_{"orig" if align_type == "AS_orig" else "final"}'
            vlm_col_key = f'vlm_as_{"orig" if align_type == "AS_orig" else "final"}'

            clip_vals = df[cols[clip_col_key]].values
            vlm_vals = df[cols[vlm_col_key]].values

            corr = compute_correlation(clip_vals, vlm_vals)

            correlation_results.append({
                'Pipeline': pipeline,
                'Alignment_Type': align_type,
                'Pearson_r': corr['pearson_r'],
                'Pearson_p': corr['pearson_p'],
                'N': len(clip_vals)
            })

            print(f"  {align_type}: Pearson r={corr['pearson_r']:.4f} (p={corr['pearson_p']:.4e})")

    # Convert to DataFrame
    all_data_df = pd.DataFrame(all_data)
    corr_results_df = pd.DataFrame(correlation_results)

    # Save correlation results
    corr_csv_path = OUTPUT_DIR / 'alignment_correlation_results.csv'
    corr_results_df.to_csv(corr_csv_path, index=False, float_format='%.6f')
    print(f"\n✓ Saved correlation results to: {corr_csv_path}")

    # Create scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = {'base': '#1f77b4', 'SA': '#ff7f0e', 'MAS': '#2ca02c', 'MAP': '#d62728'}

    for idx, align_type in enumerate(alignment_types):
        ax = axes[idx]
        clip_col = f'clip_as_{"orig" if align_type == "AS_orig" else "final"}'
        vlm_col = f'vlm_as_{"orig" if align_type == "AS_orig" else "final"}'

        # Compute correlations for each pipeline
        pipeline_corrs = {}
        for pipeline in pipelines:
            pipeline_data = all_data_df[all_data_df['pipeline'] == pipeline]
            x = pipeline_data[clip_col].values
            y = pipeline_data[vlm_col].values

            # Compute correlation for this pipeline
            corr = compute_correlation(x, y)
            pipeline_corrs[pipeline] = corr['pearson_r']

            # Scatter plot
            ax.scatter(x, y, alpha=0.5, s=40, c=colors[pipeline], label=pipeline)

            # Add regression line for this pipeline
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), color=colors[pipeline],
                       linestyle='--', alpha=0.7, linewidth=2)

        # Create title with all Pearson r values
        align_label = r'$\mathrm{AS}_{\mathrm{orig}}$' if align_type == 'AS_orig' else r'$\mathrm{AS}_{\mathrm{final}}$'
        title_text = f'{align_label}\n'
        title_text += ', '.join([f'{p}: r={pipeline_corrs[p]:.3f}' for p in pipelines])

        # Labels and title
        ax.set_xlabel(f'CLIP {align_label}', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'VLM {align_label}', fontsize=14, fontweight='bold')
        ax.set_title(title_text, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'alignment_correlation_scatter.pdf'
    png_path = OUTPUT_DIR / 'alignment_correlation_scatter.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"✓ Saved scatter plots to: {pdf_path}")
    print(f"✓ Saved scatter plots to: {png_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)

    for align_type in alignment_types:
        align_corr = corr_results_df[corr_results_df['Alignment_Type'] == align_type]
        print(f"\n{align_type}:")
        print(f"  Mean Pearson r: {align_corr['Pearson_r'].mean():.4f}")
        print(f"  Min Pearson r: {align_corr['Pearson_r'].min():.4f} "
              f"({align_corr.loc[align_corr['Pearson_r'].idxmin(), 'Pipeline']})")
        print(f"  Max Pearson r: {align_corr['Pearson_r'].max():.4f} "
              f"({align_corr.loc[align_corr['Pearson_r'].idxmax(), 'Pipeline']})")
        print(f"  Pipeline values:")
        for pipeline in pipelines:
            pipeline_r = align_corr[align_corr['Pipeline'] == pipeline]['Pearson_r'].values[0]
            print(f"    {pipeline}: r={pipeline_r:.4f}")

    plt.close()
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    main()
