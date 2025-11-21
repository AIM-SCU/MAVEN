#!/usr/bin/env python3
"""
Plot VSS comparison between mono-cultural and cross-cultural prompts.
Uses grouped bars in single plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent

# File paths (only agent pipelines for VSS)
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

def compute_vss_metrics(df, pipeline_name, confidence=0.95):
    """Compute VSS with CI."""
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
    """Create grouped bar plot for mono vs cross cultural VSS."""

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
            'mono': compute_vss_metrics(df_mono, pipeline_name),
            'cross': compute_vss_metrics(df_cross, pipeline_name)
        }

        print(f"  Mono: {len(df_mono)}, Cross: {len(df_cross)}")

    # Prepare data for plotting
    pipelines = ['SA', 'MAS', 'MAP']
    prompt_types = ['mono', 'cross']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': 16})

    # Bar settings
    x = np.arange(len(pipelines))
    width = 0.35

    # Use blue colors: dark blue for mono, light blue for cross
    colors = ['#1f77b4', '#aec7e8']  # Dark blue, light blue from tab20

    # Plot bars
    for i, prompt_type in enumerate(prompt_types):
        means = [all_data[p][prompt_type]['mean'] for p in pipelines]
        cis = [all_data[p][prompt_type]['ci'] for p in pipelines]

        bars = ax.bar(x + i * width - width/2, means, width,
                      yerr=cis, capsize=6,
                      label=f'{prompt_type.capitalize()}-cultural',
                      color=colors[i],
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=14, framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Set y-limit
    all_means = [all_data[p][t]['mean'] for p in pipelines for t in prompt_types]
    all_cis = [all_data[p][t]['ci'] for p in pipelines for t in prompt_types]
    y_max = max([m + c for m, c in zip(all_means, all_cis)])
    y_min = min([m - c for m, c in zip(all_means, all_cis)])
    ax.set_ylim(y_min - 0.02, y_max + 0.06)

    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'vss_mono_vs_cross.pdf'
    png_path = OUTPUT_DIR / 'vss_mono_vs_cross.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"\nSaved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")

    # Save CSV
    csv_data = []
    for pipeline in pipelines:
        for prompt_type in prompt_types:
            m = all_data[pipeline][prompt_type]
            csv_data.append({
                'Pipeline': pipeline,
                'Prompt_Type': prompt_type,
                'Mean': m['mean'],
                'CI': m['ci'],
                'Lower': m['lower'],
                'Upper': m['upper']
            })

    csv_df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / 'vss_mono_vs_cross.csv'
    csv_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Saved CSV to: {csv_path}")

    plt.close()

if __name__ == '__main__':
    main()
