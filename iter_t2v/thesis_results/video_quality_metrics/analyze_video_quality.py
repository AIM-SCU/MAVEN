#!/usr/bin/env python3
"""
Analyze Visual Quality and Temporal Consistency metrics for all pipelines.
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

# Column names
VISUAL_QUALITY_COL = 'Visual_Quality'
TEMPORAL_CONSISTENCY_COL = 'Temporal_Consistency'

def compute_metrics_with_ci(df, confidence=0.95):
    """Compute Visual Quality and Temporal Consistency with confidence intervals."""

    metrics = {}

    for metric_name, col_name in [('Visual_Quality', VISUAL_QUALITY_COL),
                                   ('Temporal_Consistency', TEMPORAL_CONSISTENCY_COL)]:
        values = df[col_name].values
        mean = np.mean(values)
        sem = stats.sem(values)
        ci = sem * stats.t.ppf((1 + confidence) / 2, len(values) - 1)

        metrics[metric_name] = {
            'mean': mean,
            'ci': ci,
            'lower': mean - ci,
            'upper': mean + ci,
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    return metrics

def main():
    """Analyze video quality metrics for all pipelines."""

    print("="*80)
    print("Video Quality and Temporal Consistency Analysis")
    print("="*80)

    # Load data and compute metrics
    all_metrics = {}
    for pipeline_name, filepath in FILES.items():
        print(f"\nProcessing {pipeline_name}...")
        df = pd.read_csv(filepath)
        all_metrics[pipeline_name] = compute_metrics_with_ci(df)

        # Print results
        for metric in ['Visual_Quality', 'Temporal_Consistency']:
            m = all_metrics[pipeline_name][metric]
            print(f"  {metric}: {m['mean']:.4f} ± {m['ci']:.4f} "
                  f"(range: [{m['min']:.4f}, {m['max']:.4f}])")

    # Prepare data for plotting
    pipelines = ['base', 'SA', 'MAS', 'MAP']
    metrics_list = ['Visual_Quality', 'Temporal_Consistency']

    # ========== PLOT 1: Grouped Bar Chart ==========
    fig1, ax = plt.subplots(figsize=(10, 7))

    # Increase font sizes
    plt.rcParams.update({'font.size': 14})

    # Bar settings
    x = np.arange(len(pipelines))
    width = 0.35
    colors = ['#1f77b4', '#ff7f0e']  # Blue for VQ, Orange for TC

    # Plot bars for each metric
    for idx, metric in enumerate(metrics_list):
        # Extract data
        means = [all_metrics[p][metric]['mean'] for p in pipelines]
        cis = [all_metrics[p][metric]['ci'] for p in pipelines]

        # Plot bars with error bars
        offset = idx * width - width/2
        bars = ax.bar(x + offset, means, width, yerr=cis, capsize=6,
                      label=metric.replace('_', ' '),
                      color=colors[idx], alpha=0.8,
                      edgecolor='black', linewidth=1)

        # Add numerical values on top of bars
        for i, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + ci + 0.5,
                   f'{mean:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Customize plot
    ax.set_xlabel('Pipeline', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=14, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save bar chart
    pdf_path = OUTPUT_DIR / 'video_quality_metrics.pdf'
    png_path = OUTPUT_DIR / 'video_quality_metrics.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"\n✓ Saved bar chart to: {pdf_path}")
    print(f"✓ Saved bar chart to: {png_path}")

    # ========== PLOT 2: Scatter plot (TC vs VQ) ==========
    fig2, ax_scatter = plt.subplots(figsize=(10, 7))

    # Increase font sizes
    plt.rcParams.update({'font.size': 14})

    # Pipeline colors
    pipeline_colors = {
        'base': '#1f77b4',
        'SA': '#ff7f0e',
        'MAS': '#2ca02c',
        'MAP': '#d62728'
    }

    # Plot each pipeline as a point with error bars
    for pipeline in pipelines:
        tc_mean = all_metrics[pipeline]['Temporal_Consistency']['mean']
        tc_ci = all_metrics[pipeline]['Temporal_Consistency']['ci']
        vq_mean = all_metrics[pipeline]['Visual_Quality']['mean']
        vq_ci = all_metrics[pipeline]['Visual_Quality']['ci']

        # Plot point
        ax_scatter.scatter(tc_mean, vq_mean, s=200, alpha=0.8,
                          color=pipeline_colors[pipeline], label=pipeline,
                          edgecolors='black', linewidth=2, zorder=3)

        # Add error bars
        ax_scatter.errorbar(tc_mean, vq_mean,
                           xerr=tc_ci, yerr=vq_ci,
                           fmt='none', color=pipeline_colors[pipeline],
                           alpha=0.5, capsize=5, capthick=2, zorder=2)

    # Customize plot
    ax_scatter.set_xlabel('Temporal Consistency', fontsize=16, fontweight='bold')
    ax_scatter.set_ylabel('Visual Quality', fontsize=16, fontweight='bold')
    ax_scatter.tick_params(axis='both', labelsize=14)
    ax_scatter.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                     fontsize=14, ncol=4)
    ax_scatter.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save scatter plot
    pdf_path2 = OUTPUT_DIR / 'video_quality_metrics_scatter.pdf'
    png_path2 = OUTPUT_DIR / 'video_quality_metrics_scatter.png'

    plt.savefig(pdf_path2, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path2, format='png', bbox_inches='tight', dpi=300)

    print(f"✓ Saved scatter plot to: {pdf_path2}")
    print(f"✓ Saved scatter plot to: {png_path2}")

    # Save data to CSV
    csv_data = []
    for pipeline in pipelines:
        for metric in metrics_list:
            m = all_metrics[pipeline][metric]
            csv_data.append({
                'Pipeline': pipeline,
                'Metric': metric,
                'Mean': m['mean'],
                'CI': m['ci'],
                'Lower': m['lower'],
                'Upper': m['upper'],
                'Std': m['std'],
                'Min': m['min'],
                'Max': m['max']
            })

    csv_df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / 'video_quality_metrics.csv'
    csv_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"✓ Saved data to: {csv_path}")

    # Print summary table
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    print(f"\n{'Pipeline':<10} {'Metric':<25} {'Mean':<10} {'CI':<10} {'Range':<20}")
    print("-"*80)
    for pipeline in pipelines:
        for metric in metrics_list:
            m = all_metrics[pipeline][metric]
            metric_name = metric.replace('_', ' ')
            range_str = f"[{m['min']:.3f}, {m['max']:.3f}]"
            print(f"{pipeline:<10} {metric_name:<25} {m['mean']:<10.4f} "
                  f"{m['ci']:<10.4f} {range_str:<20}")
    print("="*80)

    # Compute percentage differences relative to base
    print("\n" + "="*80)
    print("Percentage Difference from Base Pipeline")
    print("="*80)
    for metric in metrics_list:
        print(f"\n{metric.replace('_', ' ')}:")
        base_mean = all_metrics['base'][metric]['mean']
        for pipeline in ['SA', 'MAS', 'MAP']:
            pipe_mean = all_metrics[pipeline][metric]['mean']
            diff_pct = ((pipe_mean - base_mean) / base_mean) * 100
            print(f"  {pipeline}: {diff_pct:+.2f}%")
    print("="*80)

    plt.close('all')
    print("\n✓ Analysis complete!")

if __name__ == '__main__':
    main()
