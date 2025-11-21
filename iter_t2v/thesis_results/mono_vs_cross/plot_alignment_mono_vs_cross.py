#!/usr/bin/env python3
"""
Plot alignment score comparison between mono-cultural and cross-cultural prompts.
Uses grouped bars in single plot, showing both AS_orig and AS_final.
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

def compute_alignment_metrics(df, pipeline_name, confidence=0.95, all_dfs=None):
    """Compute AS_orig and AS_final with CI."""
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
                # Filter for same prompt type
                agent_df_filtered = agent_df[agent_df['prompt_type'] == df['prompt_type'].iloc[0]]
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
    """Create grouped bar plots for mono vs cross cultural alignment."""

    # Load all dataframes first and classify prompts
    all_dfs = {}
    for pipeline_name, filepath in FILES.items():
        print(f"Loading {pipeline_name}...")
        df = pd.read_csv(filepath)
        df['prompt_type'] = df['original_prompt'].apply(classify_prompt_type)
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

        # Split by prompt type
        df_mono = df[df['prompt_type'] == 'mono']
        df_cross = df[df['prompt_type'] == 'cross']

        if pipeline_name == 'base':
            all_data[pipeline_name] = {
                'mono': compute_alignment_metrics(df_mono, pipeline_name, all_dfs=all_dfs),
                'cross': compute_alignment_metrics(df_cross, pipeline_name, all_dfs=all_dfs)
            }
        else:
            all_data[pipeline_name] = {
                'mono': compute_alignment_metrics(df_mono, pipeline_name),
                'cross': compute_alignment_metrics(df_cross, pipeline_name)
            }

        print(f"  Mono: {len(df_mono)}, Cross: {len(df_cross)}")

    # Prepare data for plotting
    pipelines = ['base', 'SA', 'MAS', 'MAP']
    prompt_types = ['mono', 'cross']
    alignment_types = ['AS_orig', 'AS_final']

    # Create single plot
    fig, ax = plt.subplots(figsize=(14, 6))
    plt.rcParams.update({'font.size': 16})

    # Bar settings
    x = np.arange(len(pipelines))
    width = 0.2  # Narrower bars since we have 4 per pipeline

    # Colors: Blue for original, Yellow for final; Dark for mono, Light for cross
    colors = {
        ('AS_orig', 'mono'): '#1f77b4',    # Dark blue
        ('AS_orig', 'cross'): '#aec7e8',   # Light blue
        ('AS_final', 'mono'): '#ff7f0e',   # Dark orange/yellow
        ('AS_final', 'cross'): '#ffbb78'   # Light orange/yellow
    }

    # Plot all 4 bar groups
    bar_positions = [-1.5, -0.5, 0.5, 1.5]  # Positions for 4 bars per pipeline
    bar_labels = []

    for idx, (alignment_type, prompt_type) in enumerate([
        ('AS_orig', 'mono'),
        ('AS_orig', 'cross'),
        ('AS_final', 'mono'),
        ('AS_final', 'cross')
    ]):
        means = [all_data[p][prompt_type][alignment_type]['mean'] for p in pipelines]
        cis = [all_data[p][prompt_type][alignment_type]['ci'] for p in pipelines]

        # Create label
        align_label = 'Original' if alignment_type == 'AS_orig' else 'Final'
        prompt_label = 'Mono' if prompt_type == 'mono' else 'Cross'
        label = f'{align_label}-{prompt_label}'
        bar_labels.append(label)

        bars = ax.bar(x + bar_positions[idx] * width, means, width,
                     yerr=cis, capsize=4,
                     label=label,
                     color=colors[(alignment_type, prompt_type)],
                     alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add numerical values
        for bar, mean, ci in zip(bars, means, cis):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + ci + 0.003,
                   f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=9, rotation=0)

    ax.set_xlabel('Pipeline', fontsize=18, fontweight='bold')
    ax.set_ylabel('Alignment Score', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=13, framealpha=0.9, ncol=4)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save plots
    pdf_path = OUTPUT_DIR / 'alignment_mono_vs_cross.pdf'
    png_path = OUTPUT_DIR / 'alignment_mono_vs_cross.png'

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

    print(f"\nSaved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")

    # Save CSV
    csv_data = []
    for pipeline in pipelines:
        for prompt_type in prompt_types:
            for alignment_type in alignment_types:
                m = all_data[pipeline][prompt_type][alignment_type]
                csv_data.append({
                    'Pipeline': pipeline,
                    'Prompt_Type': prompt_type,
                    'Alignment_Type': alignment_type,
                    'Mean': m['mean'],
                    'CI': m['ci'],
                    'Lower': m['lower'],
                    'Upper': m['upper']
                })

    csv_df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / 'alignment_mono_vs_cross.csv'
    csv_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Saved CSV to: {csv_path}")

    plt.close()

if __name__ == '__main__':
    main()
