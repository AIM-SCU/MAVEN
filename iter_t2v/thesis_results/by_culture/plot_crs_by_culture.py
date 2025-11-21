#!/usr/bin/env python3
"""
Plot CRS comparison across cultures (Chinese, American, Romanian).
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

# File paths
FILES = {
    'base': BASE_DIR / 'metrics_base.csv',
    'SA': BASE_DIR / 'metrics_single.csv',
    'MAS': BASE_DIR / 'metrics_sequential.csv',
    'MAP': BASE_DIR / 'metrics_parallel.csv'
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
    """Create plot with grouped bars for CRS by culture."""

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
            all_data[pipeline_name][culture] = compute_crs_metrics(df_culture)
            print(f"  {culture.capitalize()}: {len(df_culture)}")

    # Prepare data for plotting
    pipelines = ['base', 'SA', 'MAS', 'MAP']
    metric_names = ['CRS', 'OCRS', 'PCRS', 'ACRS', 'LCRS']
    cultures = ['chinese', 'american', 'romanian']
    culture_labels = ['Chinese', 'American', 'Romanian']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 6))
    plt.rcParams.update({'font.size': 16})

    # Bar settings
    x = np.arange(len(pipelines))
    width = 0.05  # Width for each bar
    group_width = width * 3  # Width for 3 cultures
    spacing = 0  # No space between metric groups

    # Use tab10 colors for metrics
    tab10 = plt.cm.tab10.colors

    # Use tab20c colors for cultures
    tab20c = plt.cm.tab20c.colors
    # Chinese: tab20c[0] (darkest), American: tab20c[1] (medium), Romanian: tab20c[2] (lightest)
    culture_color_indices = [0, 1, 2]

    # Calculate positions for 5 metric groups
    total_width = (group_width + spacing) * len(metric_names) - spacing
    start_offset = -total_width / 2

    # Plot bars
    for metric_idx, metric in enumerate(metric_names):
        group_offset = start_offset + metric_idx * (group_width + spacing)

        for culture_idx, culture in enumerate(cultures):
            bar_offset = group_offset + culture_idx * width

            means = [all_data[p][culture][metric]['mean'] for p in pipelines]
            cis = [all_data[p][culture][metric]['ci'] for p in pipelines]

            # Use tab20c for culture shading on top of tab10 metric color
            # Mix the metric color with culture shade
            color = tab10[metric_idx]
            # Adjust alpha based on culture (darker for Chinese, lighter for Romanian)
            alpha = 0.9 - (culture_idx * 0.3)  # 0.9, 0.6, 0.3

            # Only add label for Chinese (first occurrence of each metric)
            label = metric if culture_idx == 0 else None

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

    # Create custom legend with culture explanation
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=tab10[i], alpha=0.9, edgecolor='black', label=metric_names[i])
                       for i in range(len(metric_names))]

    # Add culture explanation
    legend_elements.append(Patch(facecolor='gray', alpha=0.9, edgecolor='black', label='Dark: Chinese'))
    legend_elements.append(Patch(facecolor='gray', alpha=0.6, edgecolor='black', label='Medium: American'))
    legend_elements.append(Patch(facecolor='gray', alpha=0.3, edgecolor='black', label='Light: Romanian'))

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
             fontsize=13, framealpha=0.9, ncol=8)

    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'crs_by_culture.pdf'
    png_path = OUTPUT_DIR / 'crs_by_culture.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"\nSaved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")

    # Save CSV
    csv_data = []
    for pipeline in pipelines:
        for culture in cultures:
            for metric in metric_names:
                m = all_data[pipeline][culture][metric]
                csv_data.append({
                    'Pipeline': pipeline,
                    'Culture': culture.capitalize(),
                    'Metric': metric,
                    'Mean': m['mean'],
                    'CI': m['ci'],
                    'Lower': m['lower'],
                    'Upper': m['upper']
                })

    csv_df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / 'crs_by_culture.csv'
    csv_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Saved CSV to: {csv_path}")

    plt.close()

if __name__ == '__main__':
    main()
