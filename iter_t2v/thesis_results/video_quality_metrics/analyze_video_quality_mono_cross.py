#!/usr/bin/env python3
"""
Analyze Visual Quality and Temporal Consistency metrics grouped by mono-cultural vs cross-cultural.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import re

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

# Culture keywords
CULTURE_KEYWORDS = {
    'Chinese': ['Chinese', 'China', 'Peking duck', 'hotpot', 'Guzheng', 'Lion dance', 'Forbidden City', 'West Lake', 'Great Wall'],
    'American': ['American', 'America', 'United States', 'hamburger', 'Jazz', 'Hip-hop', 'Statue of Liberty', 'Golden Gate Bridge', 'Times Square'],
    'Romanian': ['Romanian', 'Romania', 'Sarmale', 'Cobza', 'Hora', 'Bran Castle', 'Palace of Parliament', 'Transfagarasan Highway']
}

def identify_cultures_in_prompt(prompt):
    """Identify which cultures are present in the prompt."""
    cultures = set()
    prompt_lower = prompt.lower()

    for culture, keywords in CULTURE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in prompt_lower:
                cultures.add(culture)
                break

    return cultures

def is_mono_cultural(prompt):
    """Determine if a prompt is mono-cultural or cross-cultural."""
    cultures = identify_cultures_in_prompt(prompt)
    return len(cultures) <= 1

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
            'n': len(values)
        }

    return metrics

def main():
    """Analyze video quality metrics grouped by mono vs cross-cultural."""

    print("="*80)
    print("Video Quality: Mono-cultural vs Cross-cultural Analysis")
    print("="*80)

    # Load data and compute metrics
    all_metrics = {}

    pipelines = ['base', 'SA', 'MAS', 'MAP']

    for pipeline_name in pipelines:
        print(f"\nProcessing {pipeline_name}...")
        df = pd.read_csv(FILES[pipeline_name])

        # Classify prompts as mono or cross-cultural
        df['is_mono'] = df['original_prompt'].apply(is_mono_cultural)

        mono_df = df[df['is_mono'] == True]
        cross_df = df[df['is_mono'] == False]

        print(f"  Mono-cultural: {len(mono_df)} samples")
        print(f"  Cross-cultural: {len(cross_df)} samples")

        # Compute metrics for both groups
        all_metrics[pipeline_name] = {
            'mono': compute_metrics_with_ci(mono_df),
            'cross': compute_metrics_with_ci(cross_df)
        }

        # Print results
        for scenario in ['mono', 'cross']:
            print(f"  {scenario.capitalize()}:")
            for metric in ['Visual_Quality', 'Temporal_Consistency']:
                m = all_metrics[pipeline_name][scenario][metric]
                print(f"    {metric}: {m['mean']:.4f} ± {m['ci']:.4f}")

    # Create grouped bar plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Increase font sizes
    plt.rcParams.update({'font.size': 14})

    metrics_list = ['Visual_Quality', 'Temporal_Consistency']

    for idx, metric in enumerate(metrics_list):
        ax = axes[idx]

        # Bar settings
        x = np.arange(len(pipelines))
        width = 0.35
        colors_mono = '#1f77b4'  # Blue
        colors_cross = '#ff7f0e'  # Orange

        # Extract data for mono and cross
        mono_means = [all_metrics[p]['mono'][metric]['mean'] for p in pipelines]
        mono_cis = [all_metrics[p]['mono'][metric]['ci'] for p in pipelines]
        cross_means = [all_metrics[p]['cross'][metric]['mean'] for p in pipelines]
        cross_cis = [all_metrics[p]['cross'][metric]['ci'] for p in pipelines]

        # Plot bars
        bars1 = ax.bar(x - width/2, mono_means, width, yerr=mono_cis, capsize=6,
                       label='Mono-cultural', color=colors_mono, alpha=0.8,
                       edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, cross_means, width, yerr=cross_cis, capsize=6,
                       label='Cross-cultural', color=colors_cross, alpha=0.8,
                       edgecolor='black', linewidth=1)

        # Add numerical values on top of bars
        for bars, means, cis in [(bars1, mono_means, mono_cis), (bars2, cross_means, cross_cis)]:
            for bar, mean, ci in zip(bars, means, cis):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + ci + 0.5,
                       f'{mean:.2f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Customize plot
        ax.set_xlabel('Pipeline', fontsize=16, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' '), fontsize=16, fontweight='bold')
        ax.set_title(metric.replace('_', ' '), fontsize=18, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pipelines, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=14, ncol=2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'video_quality_metrics_mono_cross.pdf'
    png_path = OUTPUT_DIR / 'video_quality_metrics_mono_cross.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"\n✓ Saved plots to: {pdf_path}")
    print(f"✓ Saved plots to: {png_path}")

    # Save data to CSV
    csv_data = []
    for pipeline in pipelines:
        for scenario in ['mono', 'cross']:
            for metric in metrics_list:
                m = all_metrics[pipeline][scenario][metric]
                csv_data.append({
                    'Pipeline': pipeline,
                    'Scenario': scenario,
                    'Metric': metric,
                    'Mean': m['mean'],
                    'CI': m['ci'],
                    'Lower': m['lower'],
                    'Upper': m['upper'],
                    'N': m['n']
                })

    csv_df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / 'video_quality_metrics_mono_cross.csv'
    csv_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"✓ Saved data to: {csv_path}")

    # Print comparison
    print("\n" + "="*80)
    print("Mono vs Cross Comparison")
    print("="*80)
    for pipeline in pipelines:
        print(f"\n{pipeline}:")
        for metric in metrics_list:
            mono_mean = all_metrics[pipeline]['mono'][metric]['mean']
            cross_mean = all_metrics[pipeline]['cross'][metric]['mean']
            diff_pct = ((cross_mean - mono_mean) / mono_mean) * 100
            print(f"  {metric.replace('_', ' ')}:")
            print(f"    Mono: {mono_mean:.4f}")
            print(f"    Cross: {cross_mean:.4f}")
            print(f"    Difference: {diff_pct:+.2f}%")
    print("="*80)

    plt.close()
    print("\n✓ Analysis complete!")

if __name__ == '__main__':
    main()
