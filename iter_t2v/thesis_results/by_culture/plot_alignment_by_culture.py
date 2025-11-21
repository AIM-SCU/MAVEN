#!/usr/bin/env python3
"""
Plot alignment score comparison across cultures (Chinese, American, Romanian).
Uses grouped bars showing both AS_orig and AS_final for each culture.
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

def compute_alignment_metrics(df, pipeline_name, confidence=0.95, all_dfs=None):
    """Compute AS_orig and AS_final with CI."""
    if len(df) == 0:
        return None

    metrics = {}

    if pipeline_name == 'base':
        as_orig_col = 'v_base_to_original_prompt_avg_avg'

        # AS_orig
        as_orig_values = df[as_orig_col].values
        as_orig_mean = np.mean(as_orig_values)
        as_orig_sem = stats.sem(as_orig_values)
        as_orig_ci = as_orig_sem * stats.t.ppf((1 + confidence) / 2, len(as_orig_values) - 1)

        metrics['AS_orig'] = {
            'mean': as_orig_mean,
            'ci': as_orig_ci,
            'lower': as_orig_mean - as_orig_ci,
            'upper': as_orig_mean + as_orig_ci
        }

        # For AS_final, average base video alignment across all agent refined prompts
        if all_dfs is not None:
            as_final_col = 'v_base_to_final_prompt_avg_avg'
            all_as_final_values = []

            for agent_name in ['SA', 'MAS', 'MAP']:
                agent_df = all_dfs[agent_name]
                # Filter for same culture
                agent_df_filtered = agent_df[agent_df['culture'] == df['culture'].iloc[0]]
                all_as_final_values.append(agent_df_filtered[as_final_col].values)

            # Stack and average
            stacked_values = np.array(all_as_final_values)
            as_final_values = np.mean(stacked_values, axis=0)

            as_final_mean = np.mean(as_final_values)
            as_final_sem = stats.sem(as_final_values)
            as_final_ci = as_final_sem * stats.t.ppf((1 + confidence) / 2, len(as_final_values) - 1)

            metrics['AS_final'] = {
                'mean': as_final_mean,
                'ci': as_final_ci,
                'lower': as_final_mean - as_final_ci,
                'upper': as_final_mean + as_final_ci
            }
    else:
        as_orig_col = 'v_var_to_original_prompt_avg_avg'
        as_final_col = 'v_var_to_final_prompt_avg_avg'

        # AS_orig
        as_orig_values = df[as_orig_col].values
        as_orig_mean = np.mean(as_orig_values)
        as_orig_sem = stats.sem(as_orig_values)
        as_orig_ci = as_orig_sem * stats.t.ppf((1 + confidence) / 2, len(as_orig_values) - 1)

        metrics['AS_orig'] = {
            'mean': as_orig_mean,
            'ci': as_orig_ci,
            'lower': as_orig_mean - as_orig_ci,
            'upper': as_orig_mean + as_orig_ci
        }

        # AS_final
        as_final_values = df[as_final_col].values
        as_final_mean = np.mean(as_final_values)
        as_final_sem = stats.sem(as_final_values)
        as_final_ci = as_final_sem * stats.t.ppf((1 + confidence) / 2, len(as_final_values) - 1)

        metrics['AS_final'] = {
            'mean': as_final_mean,
            'ci': as_final_ci,
            'lower': as_final_mean - as_final_ci,
            'upper': as_final_mean + as_final_ci
        }

    return metrics

def main():
    """Create plot with grouped bars for alignment by culture."""

    # Load all dataframes first and classify by culture
    all_dfs = {}
    for pipeline_name, filepath in FILES.items():
        print(f"Loading {pipeline_name}...")
        df = pd.read_csv(filepath)
        df['culture'] = df['original_prompt'].apply(classify_culture)
        all_dfs[pipeline_name] = df

    # Compute metrics
    all_data = {}
    for pipeline_name in FILES.keys():
        print(f"Processing {pipeline_name}...")

        if pipeline_name == 'base':
            # For base, read from SA CSV for AS_orig, but compute AS_final as average
            df = all_dfs['SA']
        else:
            df = all_dfs[pipeline_name]

        # Split by culture
        all_data[pipeline_name] = {}
        for culture in ['chinese', 'american', 'romanian']:
            df_culture = df[df['culture'] == culture]
            if pipeline_name == 'base':
                all_data[pipeline_name][culture] = compute_alignment_metrics(df_culture, pipeline_name, all_dfs=all_dfs)
            else:
                all_data[pipeline_name][culture] = compute_alignment_metrics(df_culture, pipeline_name)
            print(f"  {culture.capitalize()}: {len(df_culture)}")

    # Prepare data for plotting
    pipelines = ['base', 'SA', 'MAS', 'MAP']
    cultures = ['chinese', 'american', 'romanian']
    culture_labels = ['Chinese', 'American', 'Romanian']
    alignment_types = ['AS_orig', 'AS_final']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 6))
    plt.rcParams.update({'font.size': 16})

    # Bar settings
    x = np.arange(len(pipelines))
    width = 0.1  # Width for each bar
    group_width = width * 6  # Width for 3 cultures × 2 alignment types
    spacing = 0.01

    # Use tab20c colors for cultures
    tab20c = plt.cm.tab20c.colors
    # Blue group (0-3) for original, Orange group (4-7) for final
    # Chinese: 0, American: 1, Romanian: 3 (skip 2 for better contrast)
    orig_colors = [tab20c[0], tab20c[1], tab20c[3]]  # Blue group shades
    final_colors = [tab20c[4], tab20c[5], tab20c[7]]  # Orange group shades

    # Calculate positions
    total_width = (group_width + spacing) * 1  # Only one group
    start_offset = -total_width / 2

    # Plot bars for each culture and alignment type
    for culture_idx, culture in enumerate(cultures):
        for align_idx, align_type in enumerate(alignment_types):
            bar_offset = start_offset + (culture_idx * 2 + align_idx) * width

            means = [all_data[p][culture][align_type]['mean'] for p in pipelines]
            cis = [all_data[p][culture][align_type]['ci'] for p in pipelines]

            # Choose color based on alignment type
            if align_type == 'AS_orig':
                color = orig_colors[culture_idx]
                label = f'{culture_labels[culture_idx]}-Original' if culture_idx == 0 else None
            else:
                color = final_colors[culture_idx]
                label = f'{culture_labels[culture_idx]}-Final' if culture_idx == 0 else None

            bars = ax.bar(x + bar_offset, means, width,
                         yerr=cis, capsize=2,
                         label=label,
                         color=color,
                         alpha=0.8, edgecolor='black', linewidth=0.5)

            # Add numerical values (3 decimals)
            for bar, mean, ci in zip(bars, means, cis):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + ci + 0.003,
                       f'{mean:.3f}',
                       ha='center', va='bottom', fontsize=5.5, rotation=0)

    # Customize plot
    ax.set_xlabel('Pipeline', fontsize=18, fontweight='bold')
    ax.set_ylabel('Alignment Score', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = []

    # Add culture groups
    for idx, culture_label in enumerate(culture_labels):
        legend_elements.append(Patch(facecolor=orig_colors[idx], alpha=0.8, edgecolor='black',
                                     label=f'{culture_label}-Original'))
        legend_elements.append(Patch(facecolor=final_colors[idx], alpha=0.8, edgecolor='black',
                                     label=f'{culture_label}-Final'))

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
             fontsize=12, framealpha=0.9, ncol=6)

    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'alignment_by_culture.pdf'
    png_path = OUTPUT_DIR / 'alignment_by_culture.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"\nSaved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")

    # Save CSV
    csv_data = []
    for pipeline in pipelines:
        for culture in cultures:
            for align_type in alignment_types:
                m = all_data[pipeline][culture][align_type]
                csv_data.append({
                    'Pipeline': pipeline,
                    'Culture': culture.capitalize(),
                    'Alignment_Type': align_type,
                    'Mean': m['mean'],
                    'CI': m['ci'],
                    'Lower': m['lower'],
                    'Upper': m['upper']
                })

    csv_df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / 'alignment_by_culture.csv'
    csv_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Saved CSV to: {csv_path}")

    plt.close()

if __name__ == '__main__':
    main()
