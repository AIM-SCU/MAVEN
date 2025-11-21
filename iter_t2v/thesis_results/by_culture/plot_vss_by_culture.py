#!/usr/bin/env python3
"""
Plot VSS comparison across cultures (Chinese, American, Romanian).
Uses grouped bars with different colors for each culture.
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

# Cultural keywords for classification
CULTURAL_KEYWORDS = {
    'chinese': ['Chinese', 'Peking duck', 'mooncakes', 'dumplings', 'guzheng', 'erhu', 'dizi',
                'fan dance', 'ribbon dance', 'umbrella dance', 'Forbidden City', 'West Lake', 'Potala Palace'],
    'american': ['American', 'hot dogs', 'burgers', 'pizza slice', 'banjo', 'electric guitar', 'saxophone',
                 'hip-hop', 'moonwalk', 'tap dance', 'Statue of Liberty', 'Grand Canyon', 'Mount Rushmore'],
    'romanian': ['Romanian', 'sarmale', 'mici', 'mămăligă', 'nai', 'cobză', 'țambal',
                 'Hora', 'Sârba', 'Brâul', 'Bran Castle', 'Palace of Parliament', 'Wooden Churches of Maramureș']
}

def classify_culture(prompt):
    """Classify prompt by culture (person culture for cross-cultural prompts)."""
    for culture, keywords in CULTURAL_KEYWORDS.items():
        # Check if it's a person from this culture
        if f'{culture.capitalize()} person' in prompt or f'a {culture} person' in prompt or f'an {culture} person' in prompt:
            return culture

    # If no person specified, check for other cultural elements
    for culture, keywords in CULTURAL_KEYWORDS.items():
        if any(keyword in prompt for keyword in keywords):
            return culture

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
    """Create plot with grouped bars for VSS by culture."""

    # Load and classify data
    all_data = {}
    for pipeline_name, filepath in FILES.items():
        print(f"Processing {pipeline_name}...")
        df = pd.read_csv(filepath)

        # Classify prompts by culture
        df['culture'] = df['original_prompt'].apply(classify_culture)

        # Split by culture
        all_data[pipeline_name] = {}
        for culture in ['chinese', 'american', 'romanian']:
            df_culture = df[df['culture'] == culture]
            all_data[pipeline_name][culture] = compute_vss_metrics(df_culture, pipeline_name)
            print(f"  {culture.capitalize()}: {len(df_culture)}")

    # Prepare data for plotting
    pipelines = ['SA', 'MAS', 'MAP']
    cultures = ['chinese', 'american', 'romanian']
    culture_labels = ['Chinese', 'American', 'Romanian']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': 16})

    # Bar settings
    x = np.arange(len(pipelines))
    width = 0.25  # Width for each bar

    # Use tab20c colors for cultures
    tab20c = plt.cm.tab20c.colors
    # Chinese: tab20c[0] (darkest), American: tab20c[1] (medium), Romanian: tab20c[2] (lightest)
    colors = [tab20c[0], tab20c[1], tab20c[2]]

    # Plot bars
    for culture_idx, culture in enumerate(cultures):
        means = [all_data[p][culture]['mean'] for p in pipelines]
        cis = [all_data[p][culture]['ci'] for p in pipelines]

        offset = (culture_idx - 1) * width

        bars = ax.bar(x + offset, means, width,
                      yerr=cis, capsize=6,
                      label=culture_labels[culture_idx],
                      color=colors[culture_idx],
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
    all_means = [all_data[p][c]['mean'] for p in pipelines for c in cultures]
    all_cis = [all_data[p][c]['ci'] for p in pipelines for c in cultures]
    y_max = max([m + c for m, c in zip(all_means, all_cis)])
    y_min = min([m - c for m, c in zip(all_means, all_cis)])
    ax.set_ylim(y_min - 0.02, y_max + 0.06)

    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'vss_by_culture.pdf'
    png_path = OUTPUT_DIR / 'vss_by_culture.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"\nSaved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")

    # Save CSV
    csv_data = []
    for pipeline in pipelines:
        for culture in cultures:
            m = all_data[pipeline][culture]
            csv_data.append({
                'Pipeline': pipeline,
                'Culture': culture.capitalize(),
                'Mean': m['mean'],
                'CI': m['ci'],
                'Lower': m['lower'],
                'Upper': m['upper']
            })

    csv_df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / 'vss_by_culture.csv'
    csv_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Saved CSV to: {csv_path}")

    plt.close()

if __name__ == '__main__':
    main()
