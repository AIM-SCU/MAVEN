#!/usr/bin/env python3
"""
Plot CRS comparison between mono-cultural and cross-cultural prompts in single plot.
Uses grouped bars with dark colors for mono, light colors for cross.
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

# Cultural keywords for classification
CULTURAL_KEYWORDS = {
    'Chinese': ['Chinese', 'Peking duck', 'mooncakes', 'dumplings', 'guzheng', 'erhu', 'dizi',
                'fan dance', 'ribbon dance', 'umbrella dance', 'Forbidden City', 'West Lake', 'Potala Palace'],
    'American': ['American', 'hamburger', 'hot dog', 'apple pie', 'guitar', 'banjo', 'harmonica',
                 'square dance', 'line dance', 'tap dance', 'Statue of Liberty', 'Golden Gate Bridge', 'White House'],
    'Romanian': ['Romanian', 'sarmale', 'mici', 'mamaliga', 'panflute', 'cimbalom', 'violin',
                 'hora', 'brau', 'calusari', 'Bran Castle', 'Peles Castle', 'Palace of Parliament']
}

def classify_prompt_type(prompt):
    """Classify prompt as mono-cultural or cross-cultural."""
    cultures_found = []
    for culture, keywords in CULTURAL_KEYWORDS.items():
        if any(keyword in prompt for keyword in keywords):
            cultures_found.append(culture)

    unique_cultures = set(cultures_found)
    if len(unique_cultures) <= 1:
        return 'mono'
    else:
        return 'cross'

def compute_crs_metrics(df, confidence=0.95):
    """Compute CRS and dimension scores with CI."""
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
    """Create single plot with grouped bars for mono vs cross cultural CRS."""

    # Load and classify data
    all_data = {}
    for pipeline_name, filepath in FILES.items():
        print(f"Processing {pipeline_name}...")
        df = pd.read_csv(filepath)

        # Classify prompts
        df['prompt_type'] = df['original_prompt'].apply(classify_prompt_type)

        # Split by prompt type
        df_mono = df[df['prompt_type'] == 'mono']
        df_cross = df[df['prompt_type'] == 'cross']

        all_data[pipeline_name] = {
            'mono': compute_crs_metrics(df_mono),
            'cross': compute_crs_metrics(df_cross)
        }

        print(f"  Mono: {len(df_mono)}, Cross: {len(df_cross)}")

    # Prepare data for plotting
    pipelines = ['base', 'SA', 'MAS', 'MAP']
    metric_names = ['CRS', 'OCRS', 'PCRS', 'ACRS', 'LCRS']
    prompt_types = ['mono', 'cross']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 6))
    plt.rcParams.update({'font.size': 16})

    # Bar settings
    x = np.arange(len(pipelines))
    width = 0.08  # Width for each bar
    group_width = width * 2  # Width for mono+cross pair (no gap between them)
    spacing = 0  # No space between metric groups

    # Use tab10 colors for metrics
    tab10 = plt.cm.tab10.colors

    # Calculate positions for 5 metric groups
    total_width = (group_width + spacing) * len(metric_names) - spacing
    start_offset = -total_width / 2

    # Plot bars
    for metric_idx, metric in enumerate(metric_names):
        group_offset = start_offset + metric_idx * (group_width + spacing)

        for type_idx, prompt_type in enumerate(prompt_types):
            bar_offset = group_offset + type_idx * width

            means = [all_data[p][prompt_type][metric]['mean'] for p in pipelines]
            cis = [all_data[p][prompt_type][metric]['ci'] for p in pipelines]

            # Use darker color for mono, lighter for cross
            if prompt_type == 'mono':
                color = tab10[metric_idx]
                alpha = 0.9
            else:
                color = tab10[metric_idx]
                alpha = 0.4

            # Only add label for mono (first occurrence of each metric)
            label = metric if type_idx == 0 else None

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
                       ha='center', va='bottom', fontsize=6.5, rotation=0)

    # Customize plot
    ax.set_xlabel('Pipeline', fontsize=18, fontweight='bold')
    ax.set_ylabel('Score', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Create custom legend with mono/cross explanation
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=tab10[i], alpha=0.9, edgecolor='black', label=metric_names[i])
                       for i in range(len(metric_names))]

    # Add mono/cross explanation
    legend_elements.append(Patch(facecolor='gray', alpha=0.9, edgecolor='black', label='Dark: Mono'))
    legend_elements.append(Patch(facecolor='gray', alpha=0.4, edgecolor='black', label='Light: Cross'))

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
             fontsize=13, framealpha=0.9, ncol=7)

    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'crs_mono_vs_cross_single.pdf'
    png_path = OUTPUT_DIR / 'crs_mono_vs_cross_single.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"\nSaved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")

    plt.close()

if __name__ == '__main__':
    main()
