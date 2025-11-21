#!/usr/bin/env python3
"""
Merge EvalCrafter metrics from partition log files into the main CSV.

This script:
1. Reads all evalcrafter_metrics_log_part*.txt files
2. Parses metrics from each JSONL entry
3. Matches videos to CSV rows by video filename
4. Adds 12 EvalCrafter metrics as new columns
5. Saves the updated CSV

Usage:
    python evaluations/ec_video/merge_evalcrafter_to_csv.py --mode sequential
    python evaluations/ec_video/merge_evalcrafter_to_csv.py --mode base --results_dir /path/to/results
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import sys
import os

# EvalCrafter metrics to add (12 total)
EVALCRAFTER_METRICS = [
    'VQA_A',
    'VQA_T',
    'IS',
    'clip_temp_score',
    'warping_error',
    'face_consistency_score',
    'flow_score',
    'clip_score',
    'blip_bleu',
    'sd_score',
    'Visual_Quality',
    'Temporal_Consistency'
]


def load_metrics_from_logs(results_dir, mode):
    """Load all metrics from partition log files.

    Returns:
        dict: Mapping from video filename to metrics dict
    """
    results_path = Path(results_dir) / mode
    log_pattern = f"evalcrafter_metrics_log_part*.txt"
    log_files = sorted(results_path.glob(log_pattern))

    if not log_files:
        print(f"⚠ No log files found matching: {results_path / log_pattern}")
        return {}

    print(f"Found {len(log_files)} log file(s):")
    for log_file in log_files:
        print(f"  • {log_file.name}")

    # Collect metrics from all log files
    metrics_map = {}
    total_entries = 0

    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)
                    total_entries += 1

                    # Extract video filename (not full path to handle path differences)
                    video_path = entry.get('video_path', '')
                    video_filename = Path(video_path).name

                    if entry.get('status') != 'success':
                        print(f"  ⚠ Skipping {video_filename}: status={entry.get('status')}")
                        continue

                    # Extract metrics
                    metrics = entry.get('metrics', {})

                    # Check if we have all expected metrics
                    missing_metrics = [m for m in EVALCRAFTER_METRICS if m not in metrics]
                    if missing_metrics:
                        print(f"  ⚠ {video_filename} missing metrics: {missing_metrics}")

                    # Store metrics
                    metrics_map[video_filename] = metrics

                except json.JSONDecodeError as e:
                    print(f"  ✗ Error parsing line {line_num} in {log_file.name}: {e}")
                    continue

    print(f"\n✓ Loaded metrics for {len(metrics_map)}/{total_entries} videos")
    return metrics_map


def merge_metrics_to_csv(csv_path, metrics_map):
    """Merge EvalCrafter metrics into the CSV file.

    Args:
        csv_path: Path to the CSV file
        metrics_map: Dict mapping video filename to metrics
    """
    # Load CSV
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    # Create backup of ORIGINAL CSV before any modifications
    backup_path = csv_path.parent / f"{csv_path.stem}_backup.csv"
    print(f"\n✓ Creating backup of original CSV: {backup_path.name}")
    df.to_csv(backup_path, index=False)

    # Extract video filenames from video_path column
    df['_video_filename'] = df['video_path'].apply(lambda x: Path(x).name)

    # Add EvalCrafter metric columns (initialize with NaN)
    for metric in EVALCRAFTER_METRICS:
        if metric not in df.columns:
            df[metric] = float('nan')

    # Match and fill metrics
    matched_count = 0
    unmatched_videos = []

    for idx, row in df.iterrows():
        video_filename = row['_video_filename']

        if video_filename in metrics_map:
            metrics = metrics_map[video_filename]

            # Fill in each metric
            for metric in EVALCRAFTER_METRICS:
                if metric in metrics:
                    df.at[idx, metric] = metrics[metric]

            matched_count += 1
        else:
            unmatched_videos.append(video_filename)

    # Remove temporary column
    df = df.drop(columns=['_video_filename'])

    # Report results
    print(f"\n✓ Matched {matched_count}/{len(df)} videos")

    if unmatched_videos:
        print(f"\n⚠ {len(unmatched_videos)} videos not found in log files:")
        for video in unmatched_videos[:10]:  # Show first 10
            print(f"  • {video}")
        if len(unmatched_videos) > 10:
            print(f"  ... and {len(unmatched_videos) - 10} more")

    # Save updated CSV
    print(f"\n✓ Saving updated CSV: {csv_path.name}")
    df.to_csv(csv_path, index=False)

    # Show sample of added metrics
    print(f"\n✓ Sample of added metrics (first 3 rows):")
    print(df[['video_path'] + EVALCRAFTER_METRICS].head(3).to_string())

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Merge EvalCrafter metrics from log files into CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--mode', type=str, required=True,
                        help='Evaluation mode (base, single, parallel, sequential)')
    parser.add_argument('--results_dir', type=str,
                        default='/workspace/t2v_secret/iter_t2v/results',
                        help='Results directory (default: /workspace/t2v_secret/iter_t2v/results)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    mode_dir = results_dir / args.mode
    csv_path = mode_dir / f"metrics_{args.mode}.csv"

    # Check if CSV exists
    if not csv_path.exists():
        print(f"✗ CSV file not found: {csv_path}")
        sys.exit(1)

    print("=" * 60)
    print("EvalCrafter Metrics Merger")
    print("=" * 60)
    print(f"Mode:        {args.mode}")
    print(f"Results dir: {results_dir}")
    print(f"CSV file:    {csv_path.name}")
    print()

    # Load metrics from log files
    metrics_map = load_metrics_from_logs(results_dir, args.mode)

    if not metrics_map:
        print("\n✗ No metrics found in log files. Exiting.")
        sys.exit(1)

    # Merge metrics into CSV
    df = merge_metrics_to_csv(csv_path, metrics_map)

    print("\n" + "=" * 60)
    print("✓ Merge completed successfully!")
    print("=" * 60)
    print(f"\nMetrics added: {', '.join(EVALCRAFTER_METRICS)}")
    print(f"\nUpdated CSV: {csv_path}")
    print(f"Backup saved: {csv_path.parent / f'{csv_path.stem}_backup.csv'}")


if __name__ == '__main__':
    main()
