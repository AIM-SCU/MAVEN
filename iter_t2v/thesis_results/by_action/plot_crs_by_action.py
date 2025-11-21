#!/usr/bin/env python3
"""
Plot CRS comparison across action types (food, music, dance).
Uses grouped bars with different colors for each action type.
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

# Action keywords for classification
ACTION_KEYWORDS = {
    'food': ['eating', 'Peking duck', 'mooncakes', 'dumplings', 'hot dogs', 'burgers', 'pizza slice',
             'sarmale', 'mici', 'mămăligă'],
    'music': ['playing', 'guzheng', 'erhu', 'dizi', 'banjo', 'electric guitar', 'saxophone',
              'nai', 'cobză', 'țambal'],
    'dance': ['dancing', 'fan dance', 'ribbon dance', 'umbrella dance', 'hip-hop', 'moonwalk', 'tap dance',
              'Hora', 'Sârba', 'Brâul']
}

def classify_action(prompt):
    """Classify prompt by action type."""
    for action_type, keywords in ACTION_KEYWORDS.items():
        if any(keyword in prompt for keyword in keywords):
            return action_type
    return 'unknown'

def compute_crs_metrics(df, confidence=0.95):
    """Compute CRS and dimension scores with CI."""
    if len(df) == 0:
        return None

    metrics = {}

    # CRS components
    components = {
        'OCRS': 'overall_avg',
        'PCRS': 'person_avg',
        'ACRS': 'action_avg',
        'LCRS': 'location_avg'
    }

    for name, col in components.items():
        values = df[col].values
        mean = np.mean(values)
        sem = stats.sem(values)
        ci = sem * stats.t.ppf((1 + confidence) / 2, len(values) - 1)

        metrics[name] = {
            'mean': mean,
            'ci': ci,
            'lower': mean - ci,
            'upper': mean + ci
        }

    # CRS as average
    crs_mean = np.mean([metrics[k]['mean'] for k in ['OCRS', 'PCRS', 'ACRS', 'LCRS']])
    crs_ci = np.mean([metrics[k]['ci'] for k in ['OCRS', 'PCRS', 'ACRS', 'LCRS']])

    metrics['CRS'] = {
        'mean': crs_mean,
        'ci': crs_ci,
        'lower': crs_mean - crs_ci,
        'upper': crs_mean + crs_ci
    }

    return metrics

def main():
    """Create plot with grouped bars for CRS by action type."""

    # Load and classify data
    all_data = {}
    for pipeline_name, filepath in FILES.items():
        print(f"Processing {pipeline_name}...")
        df = pd.read_csv(filepath)

        # Classify prompts by action type
        df['action_type'] = df['original_prompt'].apply(classify_action)

        # Split by action type
        all_data[pipeline_name] = {}
        for action_type in ['food', 'music', 'dance']:
            df_action = df[df['action_type'] == action_type]
            all_data[pipeline_name][action_type] = compute_crs_metrics(df_action)
            print(f"  {action_type.capitalize()}: {len(df_action)}")

    # Prepare data for plotting
    pipelines = ['base', 'SA', 'MAS', 'MAP']
    metric_names = ['CRS', 'OCRS', 'PCRS', 'ACRS', 'LCRS']
    action_types = ['food', 'music', 'dance']
    action_labels = ['Food', 'Music', 'Dance']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 6))
    plt.rcParams.update({'font.size': 16})

    # Bar settings
    x = np.arange(len(pipelines))
    width = 0.05  # Width for each bar
    group_width = width * 3  # Width for 3 action types
    spacing = 0  # No space between metric groups

    # Use tab10 colors for metrics
    tab10 = plt.cm.tab10.colors

    # Use tab20c colors for action types
    tab20c = plt.cm.tab20c.colors
    # Food: tab20c[0] (darkest), Music: tab20c[1] (medium), Dance: tab20c[2] (lightest)

    # Calculate positions for 5 metric groups
    total_width = (group_width + spacing) * len(metric_names) - spacing
    start_offset = -total_width / 2

    # Plot bars
    for metric_idx, metric in enumerate(metric_names):
        group_offset = start_offset + metric_idx * (group_width + spacing)

        for action_idx, action_type in enumerate(action_types):
            bar_offset = group_offset + action_idx * width

            means = [all_data[p][action_type][metric]['mean'] for p in pipelines]
            cis = [all_data[p][action_type][metric]['ci'] for p in pipelines]

            # Use tab10 for metric color, adjust alpha based on action type
            color = tab10[metric_idx]
            alpha = 0.9 - (action_idx * 0.3)  # 0.9, 0.6, 0.3

            # Only add label for Food (first occurrence of each metric)
            label = metric if action_idx == 0 else None

            bars = ax.bar(x + bar_offset, means, width,
                         yerr=cis, capsize=3,
                         label=label,
                         color=color, alpha=alpha,
                         edgecolor='black', linewidth=0.5)

            # Add numerical values (3 decimals)
            for bar, mean, ci in zip(bars, means, cis):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + ci + 0.003,
                       f'{mean:.3f}',
                       ha='center', va='bottom', fontsize=6, rotation=0)

    # Customize plot
    ax.set_xlabel('Pipeline', fontsize=18, fontweight='bold')
    ax.set_ylabel('Score', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Create custom legend with action type explanation
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=tab10[i], alpha=0.9, edgecolor='black', label=metric_names[i])
                       for i in range(len(metric_names))]

    # Add action type explanation
    legend_elements.append(Patch(facecolor='gray', alpha=0.9, edgecolor='black', label='Dark: Food'))
    legend_elements.append(Patch(facecolor='gray', alpha=0.6, edgecolor='black', label='Medium: Music'))
    legend_elements.append(Patch(facecolor='gray', alpha=0.3, edgecolor='black', label='Light: Dance'))

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
             fontsize=13, framealpha=0.9, ncol=8)

    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'crs_by_action.pdf'
    png_path = OUTPUT_DIR / 'crs_by_action.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"\nSaved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")

    # Save CSV
    csv_data = []
    for pipeline in pipelines:
        for action_type in action_types:
            for metric in metric_names:
                m = all_data[pipeline][action_type][metric]
                csv_data.append({
                    'Pipeline': pipeline,
                    'Action_Type': action_type.capitalize(),
                    'Metric': metric,
                    'Mean': m['mean'],
                    'CI': m['ci'],
                    'Lower': m['lower'],
                    'Upper': m['upper']
                })

    csv_df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / 'crs_by_action.csv'
    csv_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Saved CSV to: {csv_path}")

    plt.close()

if __name__ == '__main__':
    main()
