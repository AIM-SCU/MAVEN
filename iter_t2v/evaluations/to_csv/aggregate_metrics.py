#!/usr/bin/env python3
"""
JSONL to CSV Aggregator Script
=============================

This script aggregates key metrics from JSONL files and outputs them to CSV format.

For base mode:
- Input: vlm_base.jsonl (preferred) or video_culture_relevance_base.jsonl (fallback)
- Metrics: overall_avg, person_avg, action_avg, location_avg, vlm_*_score

For variant modes (single, parallel, sequential):
- Input: vlm_{mode}.jsonl (preferred) or video_text_alignment_{mode}.jsonl (fallback)
- Metrics: overall_avg, person_avg, action_avg, location_avg, base_vs_{mode}_avg, 
           vlm_*_score fields, and all *_avg_avg fields from video-text alignments

Usage:
    # Use default output paths (saves to results/{mode}/metrics_{mode}.csv)
    python evaluations/to_csv/aggregate_metrics.py --mode base
    python evaluations/to_csv/aggregate_metrics.py --mode single
    python evaluations/to_csv/aggregate_metrics.py --mode parallel
    python evaluations/to_csv/aggregate_metrics.py --mode sequential
    
    # Or specify custom output path
    python evaluations/to_csv/aggregate_metrics.py --mode base --output custom_path.csv
"""
import argparse
import json
import csv
from pathlib import Path
import sys

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
MODES = ["base", "single", "parallel", "sequential"]


def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def get_text_fields(entry):
    """Extract only the core text fields we want to keep."""
    text_fields = {}
    # Core identification fields
    core_fields = [
        'original_prompt', 'final_prompt', 'video_path', 'image_id', 
        'overall_cultural_relevance', 'person_cultural_relevance', 
        'action_cultural_relevance', 'location_cultural_relevance'
    ]
    
    for field in core_fields:
        if field in entry:
            text_fields[field] = entry[field]
    
    return text_fields


def get_base_metrics(entry):
    """Extract metrics for base mode."""
    metrics = {
        'overall_avg': entry.get('overall_avg', 0.0),
        'person_avg': entry.get('person_avg', 0.0),
        'action_avg': entry.get('action_avg', 0.0),
        'location_avg': entry.get('location_avg', 0.0)
    }
    
    # Add VLM scores (4 dimensions for base mode)
    vlm_metrics = [
        'vlm_overall_score', 'vlm_person_score', 'vlm_action_score', 'vlm_location_score'
    ]
    
    for metric in vlm_metrics:
        if metric in entry:
            metrics[metric] = entry.get(metric, 0.0)
    
    # Add EvalCrafter metrics if available
    ec_metrics = [
        'ec_VQA_A', 'ec_VQA_T', 'ec_IS', 'ec_clip_temp_score', 'ec_warping_error',
        'ec_face_consistency_score', 'ec_flow_score', 'ec_clip_score', 'ec_blip_bleu',
        'ec_sd_score', 'ec_Visual_Quality', 'ec_Temporal_Consistency'
    ]
    
    for metric in ec_metrics:
        if metric in entry:
            metrics[metric] = entry.get(metric, 0.0)
    
    return metrics


def get_variant_metrics(entry, mode):
    """Extract metrics for variant modes (single, parallel, sequential)."""
    metrics = {
        'overall_avg': entry.get('overall_avg', 0.0),
        'person_avg': entry.get('person_avg', 0.0),
        'action_avg': entry.get('action_avg', 0.0),
        'location_avg': entry.get('location_avg', 0.0),
        f'base_vs_{mode}_avg': entry.get(f'base_vs_{mode}_avg', 0.0)
    }
    
    # Add all video-text alignment avg_avg metrics
    alignment_metrics = [
        'v_base_to_overall_cultural_relevance_avg_avg',
        'v_base_to_person_cultural_relevance_avg_avg',
        'v_base_to_action_cultural_relevance_avg_avg', 
        'v_base_to_location_cultural_relevance_avg_avg',
        'v_base_to_original_prompt_avg_avg',
        'v_base_to_final_prompt_avg_avg',
        'v_var_to_overall_cultural_relevance_avg_avg',
        'v_var_to_person_cultural_relevance_avg_avg',
        'v_var_to_action_cultural_relevance_avg_avg',
        'v_var_to_location_cultural_relevance_avg_avg',
        'v_var_to_original_prompt_avg_avg',
        'v_var_to_final_prompt_avg_avg'
    ]
    
    for metric in alignment_metrics:
        metrics[metric] = entry.get(metric, 0.0)
    
    # Add EvalCrafter metrics if available
    ec_metrics = [
        'ec_VQA_A', 'ec_VQA_T', 'ec_IS', 'ec_clip_temp_score', 'ec_warping_error',
        'ec_face_consistency_score', 'ec_flow_score', 'ec_clip_score', 'ec_blip_bleu',
        'ec_sd_score', 'ec_Visual_Quality', 'ec_Temporal_Consistency'
    ]
    
    for metric in ec_metrics:
        if metric in entry:
            metrics[metric] = entry.get(metric, 0.0)
    
    # Add VLM scores after all video metrics (cultural relevance, visual similarity, and text-image alignment)
    vlm_metrics = [
        # Cultural relevance scores (4 dimensions)
        'vlm_overall_score', 'vlm_person_score', 'vlm_action_score', 'vlm_location_score',
        # Visual similarity score (1 dimension)
        'vlm_visual_similarity_score',
        # Text-image alignment scores (12 dimensions)
        'vlm_v_base_overall_score', 'vlm_v_base_person_score', 'vlm_v_base_action_score', 'vlm_v_base_location_score',
        'vlm_v_base_original_prompt_score', 'vlm_v_base_final_prompt_score',
        'vlm_v_var_overall_score', 'vlm_v_var_person_score', 'vlm_v_var_action_score', 'vlm_v_var_location_score',
        'vlm_v_var_original_prompt_score', 'vlm_v_var_final_prompt_score'
    ]
    
    for metric in vlm_metrics:
        if metric in entry:
            metrics[metric] = entry.get(metric, 0.0)
    
    return metrics


def save_to_csv(data, output_path, fieldnames):
    """Save data to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved {len(data)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate key metrics from JSONL to CSV")
    parser.add_argument("--mode", choices=MODES, required=True, help="Mode to process")
    parser.add_argument("--output", help="Output CSV file path (optional, defaults to results/{mode}/metrics_{mode}.csv)")
    args = parser.parse_args()

    # Determine input file based on mode
    if args.mode == "base":
        # Try VLM combined version first, then fallback to original
        vlm_file = RESULTS_DIR / "base" / "vlm_base.jsonl"
        orig_file = RESULTS_DIR / "base" / "video_culture_relevance_base.jsonl"
        input_file = vlm_file if vlm_file.exists() else orig_file
    else:
        # Try VLM combined version first, then fallback to original
        vlm_file = RESULTS_DIR / args.mode / f"vlm_{args.mode}.jsonl"
        orig_file = RESULTS_DIR / args.mode / f"video_text_alignment_{args.mode}.jsonl"
        input_file = vlm_file if vlm_file.exists() else orig_file

    # Determine output file - default to results folder with mode suffix
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = RESULTS_DIR / args.mode / f"metrics_{args.mode}.csv"
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        if args.mode != "base":
            print(f"Make sure to run the video text alignment script for mode '{args.mode}' first.")
        sys.exit(1)

    print(f"Loading data from: {input_file}")
    data = load_jsonl(input_file)
    
    if not data:
        print("Error: No data found in input file")
        sys.exit(1)

    # Extract metrics based on mode
    aggregated_data = []
    
    if args.mode == "base":
        for entry in data:
            metrics = get_base_metrics(entry)
            # Keep only core text fields and specified numerical metrics
            result = get_text_fields(entry)
            result.update(metrics)
            aggregated_data.append(result)
        
        # Use all fieldnames from the first entry
        fieldnames = list(aggregated_data[0].keys()) if aggregated_data else []
    
    else:
        for entry in data:
            metrics = get_variant_metrics(entry, args.mode)
            # Keep only core text fields and specified numerical metrics
            result = get_text_fields(entry)
            result.update(metrics)
            aggregated_data.append(result)
        
        # Use all fieldnames from the first entry
        fieldnames = list(aggregated_data[0].keys()) if aggregated_data else []

    # Save to CSV
    save_to_csv(aggregated_data, output_file, fieldnames)
    
    # Print summary statistics
    print(f"\nSummary for mode '{args.mode}':")
    print(f"Total entries: {len(aggregated_data)}")
    
    if args.mode == "base":
        avg_overall = sum(entry['overall_avg'] for entry in aggregated_data) / len(aggregated_data)
        print(f"Average overall_avg: {avg_overall:.4f}")
        
        # Print VLM score averages if available
        vlm_metrics = ['vlm_overall_score', 'vlm_person_score', 'vlm_action_score', 'vlm_location_score']
        for metric in vlm_metrics:
            if metric in aggregated_data[0]:
                avg_vlm = sum(entry.get(metric, 0) for entry in aggregated_data) / len(aggregated_data)
                print(f"Average {metric}: {avg_vlm:.4f}")
    else:
        avg_overall = sum(entry['overall_avg'] for entry in aggregated_data) / len(aggregated_data)
        avg_base_vs_var = sum(entry[f'base_vs_{args.mode}_avg'] for entry in aggregated_data) / len(aggregated_data)
        print(f"Average overall_avg: {avg_overall:.4f}")
        print(f"Average base_vs_{args.mode}_avg: {avg_base_vs_var:.4f}")
        
        # Print key VLM score averages if available
        key_vlm_metrics = ['vlm_overall_score', 'vlm_visual_similarity_score']
        for metric in key_vlm_metrics:
            if metric in aggregated_data[0]:
                avg_vlm = sum(entry.get(metric, 0) for entry in aggregated_data) / len(aggregated_data)
                print(f"Average {metric}: {avg_vlm:.4f}")


if __name__ == "__main__":
    main()
