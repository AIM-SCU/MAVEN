#!/usr/bin/env python3
"""
Plot CRS comparison between mono-cultural and cross-cultural prompts.
Uses side-by-side subplots.
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

    # If all elements are from same culture or only one culture found, it's mono
    # Otherwise it's cross
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
    """Create side-by-side plots for mono vs cross cultural CRS."""

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
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # tab10 colors

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plt.rcParams.update({'font.size': 16})

    # Plot settings
    x = np.arange(len(pipelines))
    width = 0.15

    # Plot mono-cultural
    for i, metric in enumerate(metric_names):
        means = [all_data[p]['mono'][metric]['mean'] for p in pipelines]
        cis = [all_data[p]['mono'][metric]['ci'] for p in pipelines]

        bars = ax1.bar(x + i * width - 2 * width, means, width,
                       yerr=cis, capsize=4,
                       label=metric, color=colors[i],
                       alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add numerical values
        for j, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + ci + 0.005,
                    f'{mean:.3f}',
                    ha='center', va='bottom', fontsize=9.5, rotation=0)

    ax1.set_xlabel('Pipeline', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=18, fontweight='bold')
    ax1.set_title('Mono-cultural Prompts', fontsize=18, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pipelines, fontsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot cross-cultural
    for i, metric in enumerate(metric_names):
        means = [all_data[p]['cross'][metric]['mean'] for p in pipelines]
        cis = [all_data[p]['cross'][metric]['ci'] for p in pipelines]

        bars = ax2.bar(x + i * width - 2 * width, means, width,
                       yerr=cis, capsize=4,
                       label=metric, color=colors[i],
                       alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add numerical values
        for j, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + ci + 0.005,
                    f'{mean:.3f}',
                    ha='center', va='bottom', fontsize=9.5, rotation=0)

    ax2.set_xlabel('Pipeline', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=18, fontweight='bold')
    ax2.set_title('Cross-cultural Prompts', fontsize=18, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(pipelines, fontsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Shared legend at bottom
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fontsize=14, framealpha=0.9, ncol=5)

    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'crs_mono_vs_cross.pdf'
    png_path = OUTPUT_DIR / 'crs_mono_vs_cross.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"\nSaved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")

    # Save CSV
    csv_data = []
    for pipeline in pipelines:
        for prompt_type in ['mono', 'cross']:
            for metric in metric_names:
                m = all_data[pipeline][prompt_type][metric]
                csv_data.append({
                    'Pipeline': pipeline,
                    'Prompt_Type': prompt_type,
                    'Metric': metric,
                    'Mean': m['mean'],
                    'CI': m['ci'],
                    'Lower': m['lower'],
                    'Upper': m['upper']
                })

    csv_df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / 'crs_mono_vs_cross.csv'
    csv_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Saved CSV to: {csv_path}")

    plt.close()

if __name__ == '__main__':
    main()
