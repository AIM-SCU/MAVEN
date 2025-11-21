#!/usr/bin/env python3
"""
Compute and plot VLM-based alignment scores for all pipelines.
Each pipeline (base, SA, MAS, MAP) has 2 bars:
1. VLM_AS_orig: Alignment with original prompt
2. VLM_AS_final: Alignment with refined prompt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent

# File paths (all pipelines)
FILES = {
    'base': BASE_DIR / 'metrics_base.csv',
    'SA': BASE_DIR / 'metrics_single.csv',
    'MAS': BASE_DIR / 'metrics_sequential.csv',
    'MAP': BASE_DIR / 'metrics_parallel.csv'
}

def compute_vlm_metrics_with_ci(df, pipeline_name, confidence=0.95, all_dfs=None):
    """
    Compute VLM_AS_orig and VLM_AS_final with confidence intervals.

    For agent pipelines:
    VLM_AS_orig = VLM_AS_{v_agent, Pr_orig}
    VLM_AS_final = VLM_AS_{v_agent, Pr_final}

    For base pipeline:
    VLM_AS_orig = VLM_AS_{v_base, Pr_orig}
    VLM_AS_final = average of VLM_AS_{v_base, Pr_final} across SA, MAS, MAP
    """

    metrics = {}

    if pipeline_name == 'base':
        # For base, use base video alignment scores
        as_orig_col = 'vlm_v_base_original_prompt_score'

        # Alignment with original prompt
        as_orig_values = df[as_orig_col].values
        as_orig_mean = np.mean(as_orig_values)
        as_orig_sem = stats.sem(as_orig_values)
        as_orig_ci = as_orig_sem * stats.t.ppf((1 + confidence) / 2, len(as_orig_values) - 1)

        metrics['VLM_AS_orig'] = {
            'mean': as_orig_mean,
            'ci': as_orig_ci,
            'lower': as_orig_mean - as_orig_ci,
            'upper': as_orig_mean + as_orig_ci
        }

        # For VLM_AS_final, average base video alignment across all agent refined prompts
        if all_dfs is not None:
            as_final_col = 'vlm_v_base_final_prompt_score'
            all_as_final_values = []

            for agent_name in ['SA', 'MAS', 'MAP']:
                agent_df = all_dfs[agent_name]
                all_as_final_values.append(agent_df[as_final_col].values)

            # Stack all values: shape (3, n_samples), then average across pipelines
            stacked_values = np.array(all_as_final_values)  # shape: (3, n_samples)
            as_final_values = np.mean(stacked_values, axis=0)  # average across 3 pipelines

            as_final_mean = np.mean(as_final_values)
            as_final_sem = stats.sem(as_final_values)
            as_final_ci = as_final_sem * stats.t.ppf((1 + confidence) / 2, len(as_final_values) - 1)

            metrics['VLM_AS_final'] = {
                'mean': as_final_mean,
                'ci': as_final_ci,
                'lower': as_final_mean - as_final_ci,
                'upper': as_final_mean + as_final_ci
            }
    else:
        # For agent pipelines
        as_orig_col = 'vlm_v_var_original_prompt_score'
        as_final_col = 'vlm_v_var_final_prompt_score'

        # Alignment with original prompt
        as_orig_values = df[as_orig_col].values
        as_orig_mean = np.mean(as_orig_values)
        as_orig_sem = stats.sem(as_orig_values)
        as_orig_ci = as_orig_sem * stats.t.ppf((1 + confidence) / 2, len(as_orig_values) - 1)

        metrics['VLM_AS_orig'] = {
            'mean': as_orig_mean,
            'ci': as_orig_ci,
            'lower': as_orig_mean - as_orig_ci,
            'upper': as_orig_mean + as_orig_ci
        }

        # Alignment with refined prompt
        as_final_values = df[as_final_col].values
        as_final_mean = np.mean(as_final_values)
        as_final_sem = stats.sem(as_final_values)
        as_final_ci = as_final_sem * stats.t.ppf((1 + confidence) / 2, len(as_final_values) - 1)

        metrics['VLM_AS_final'] = {
            'mean': as_final_mean,
            'ci': as_final_ci,
            'lower': as_final_mean - as_final_ci,
            'upper': as_final_mean + as_final_ci
        }

    return metrics

def main():
    """Create grouped bar plot for VLM alignment scores."""

    # Load all dataframes first
    all_dfs = {}
    for pipeline_name, filepath in FILES.items():
        print(f"Loading {pipeline_name}...")
        all_dfs[pipeline_name] = pd.read_csv(filepath)

    # Compute metrics
    all_metrics = {}
    for pipeline_name in FILES.keys():
        print(f"Processing {pipeline_name}...")
        if pipeline_name == 'base':
            # For base, read from SA CSV for VLM_AS_orig, but compute VLM_AS_final as average
            df = all_dfs['SA']
            all_metrics[pipeline_name] = compute_vlm_metrics_with_ci(df, pipeline_name, all_dfs=all_dfs)
        else:
            df = all_dfs[pipeline_name]
            all_metrics[pipeline_name] = compute_vlm_metrics_with_ci(df, pipeline_name)

    # Prepare data for plotting (left to right order)
    pipelines = ['base', 'SA', 'MAS', 'MAP']
    metric_names = ['VLM_AS_orig', 'VLM_AS_final']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Increase font sizes globally
    plt.rcParams.update({'font.size': 16})

    # Bar settings
    x = np.arange(len(pipelines))
    width = 0.35
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange from tab10

    # Plot bars with error bars for each pipeline
    for i, metric in enumerate(metric_names):
        # Create label only for first bar (will apply to all bars of this metric)
        if metric == 'VLM_AS_orig':
            label = r'$\mathrm{VLM\_AS}(v, \mathrm{Pr}_{\mathrm{orig}})$'
        else:  # VLM_AS_final
            label = r'$\mathrm{VLM\_AS}(v, \mathrm{Pr}_{\mathrm{final}})$'

        for j, pipeline in enumerate(pipelines):
            mean = all_metrics[pipeline][metric]['mean']
            ci = all_metrics[pipeline][metric]['ci']

            # Only add label to first bar of each metric type
            bar_label = label if j == 0 else None

            bar = ax.bar(x[j] + i * width - width/2, mean, width,
                        yerr=ci, capsize=6,
                        label=bar_label, color=colors[i],
                        alpha=0.8, edgecolor='black', linewidth=0.5)

            # Add numerical values on top of bars
            height = bar[0].get_height()
            ax.text(x[j] + i * width - width/2, height + ci + 0.05,
                   f'{mean:.2f}',
                   ha='center', va='bottom', fontsize=10, rotation=0)

    # Customize plot
    ax.set_xlabel('Pipeline', fontsize=18, fontweight='bold')
    ax.set_ylabel('VLM Score', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=14, framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Set y-limit
    all_means = [all_metrics[p][m]['mean'] for p in pipelines for m in metric_names]
    all_cis = [all_metrics[p][m]['ci'] for p in pipelines for m in metric_names]
    y_max = max([m + c for m, c in zip(all_means, all_cis)])
    y_min = min([m - c for m, c in zip(all_means, all_cis)])
    ax.set_ylim(y_min - 0.1, y_max + 0.3)

    # Tight layout
    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'vlm_enrichment_and_alignment.pdf'
    png_path = OUTPUT_DIR / 'vlm_enrichment_and_alignment.png'

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
    csv_path = OUTPUT_DIR / 'vlm_enrichment_and_alignment.csv'
    csv_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Saved CSV to: {csv_path}")

    # Print results
    print("\n" + "="*80)
    print("VLM Alignment Scores Summary")
    print("="*80)
    for pipeline in pipelines:
        print(f"\n{pipeline}:")
        for metric in metric_names:
            m = all_metrics[pipeline][metric]
            metric_label = "VLM_AS(v_agent, Pr_orig)" if metric == "VLM_AS_orig" else "VLM_AS(v_agent, Pr_final)"
            print(f"  {metric_label:30s}: {m['mean']:.4f} ± {m['ci']:.4f}")
    print("="*80)

    plt.close()

if __name__ == '__main__':
    main()
