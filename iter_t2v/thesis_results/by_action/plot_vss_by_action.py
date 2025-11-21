#!/usr/bin/env python3
"""
Plot VSS comparison across action types (food, music, dance).
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

# File paths (only agent-based pipelines have VSS)
FILES = {
    'SA': BASE_DIR / 'metrics_single.csv',
    'MAS': BASE_DIR / 'metrics_sequential.csv',
    'MAP': BASE_DIR / 'metrics_parallel.csv'
}

# VSS column mapping
VSS_COL_MAP = {
    'SA': 'base_vs_single_avg',
    'MAP': 'base_vs_parallel_avg',
    'MAS': 'base_vs_sequential_avg'
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

def compute_vss_metrics(df, pipeline_name, confidence=0.95):
    """Compute VSS with CI."""
    if len(df) == 0:
        return None

    vss_col = VSS_COL_MAP[pipeline_name]
    values = df[vss_col].values
    mean = np.mean(values)
    sem = stats.sem(values)
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(values) - 1)

    return {
        'mean': mean,
        'ci': ci,
        'lower': mean - ci,
        'upper': mean + ci
    }

def main():
    """Create plot with grouped bars for VSS by action type."""

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
            all_data[pipeline_name][action_type] = compute_vss_metrics(df_action, pipeline_name)
            print(f"  {action_type.capitalize()}: {len(df_action)}")

    # Prepare data for plotting
    pipelines = ['SA', 'MAS', 'MAP']
    action_types = ['food', 'music', 'dance']
    action_labels = ['Food', 'Music', 'Dance']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': 16})

    # Bar settings
    x = np.arange(len(pipelines))
    width = 0.25  # Width for each bar

    # Use tab20c colors for action types
    tab20c = plt.cm.tab20c.colors
    # Food: tab20c[0] (darkest), Music: tab20c[1] (medium), Dance: tab20c[2] (lightest)
    colors = [tab20c[0], tab20c[1], tab20c[2]]

    # Plot bars
    for action_idx, action_type in enumerate(action_types):
        means = [all_data[p][action_type]['mean'] for p in pipelines]
        cis = [all_data[p][action_type]['ci'] for p in pipelines]

        offset = (action_idx - 1) * width

        bars = ax.bar(x + offset, means, width,
                      yerr=cis, capsize=6,
                      label=action_labels[action_idx],
                      color=colors[action_idx],
                      alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add numerical values
        for bar, mean, ci in zip(bars, means, cis):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + ci + 0.005,
                   f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=10, rotation=0)

    # Customize plot
    ax.set_xlabel('Pipeline', fontsize=18, fontweight='bold')
    ax.set_ylabel('Visual Similarity Score', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=14, framealpha=0.9, ncol=3)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Set y-limit
    all_means = [all_data[p][a]['mean'] for p in pipelines for a in action_types]
    all_cis = [all_data[p][a]['ci'] for p in pipelines for a in action_types]
    y_max = max([m + c for m, c in zip(all_means, all_cis)])
    y_min = min([m - c for m, c in zip(all_means, all_cis)])
    ax.set_ylim(y_min - 0.02, y_max + 0.06)

    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'vss_by_action.pdf'
    png_path = OUTPUT_DIR / 'vss_by_action.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"\nSaved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")

    # Save CSV
    csv_data = []
    for pipeline in pipelines:
        for action_type in action_types:
            m = all_data[pipeline][action_type]
            csv_data.append({
                'Pipeline': pipeline,
                'Action_Type': action_type.capitalize(),
                'Mean': m['mean'],
                'CI': m['ci'],
                'Lower': m['lower'],
                'Upper': m['upper']
            })

    csv_df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / 'vss_by_action.csv'
    csv_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Saved CSV to: {csv_path}")

    plt.close()

if __name__ == '__main__':
    main()
