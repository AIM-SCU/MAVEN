#!/usr/bin/env python3
"""
VLM Results Combiner Script

This script combines all VLM evaluation dimensions for a given mode and merges them with the original data.

For base mode:
- Combines cultural relevance scores (4 dimensions)
- Merges with video_culture_relevance_base.jsonl

For other modes (single, sequential, parallel):
- Combines cultural relevance scores (4 dimensions)
- Combines visual similarity scores (1 dimension)  
- Combines text-image alignment scores (12 dimensions)
- Merges with video_text_alignment_{mode}.jsonl

Usage:
    python evaluations/vlm/combine_vlm_results.py --mode base
    python evaluations/vlm/combine_vlm_results.py --mode single --max_lines 2
    python evaluations/vlm/combine_vlm_results.py --mode sequential
    python evaluations/vlm/combine_vlm_results.py --mode parallel

Parameters:
    --mode: Processing mode (base, single, sequential, parallel)
    --max_lines: Maximum number of lines to process (optional, processes all if not specified)

Output files:
    /workspace/t2v_self/iter_t2v/results/{mode}/vlm_{mode}.jsonl
"""

import argparse
import json
import os
from typing import Dict, List

RESULTS_DIR = "/workspace/t2v_self/iter_t2v/results"

def get_file_paths(mode):
    """Get all relevant file paths for a given mode"""
    mode_dir = os.path.join(RESULTS_DIR, mode)
    
    if mode == "base":
        return {
            "original_data": os.path.join(mode_dir, "video_culture_relevance_base.jsonl"),
            "culture_relevance": os.path.join(mode_dir, "vlm_eval_culture_relevance_log.jsonl"),
            "output": os.path.join(mode_dir, "vlm_base.jsonl")
        }
    else:
        return {
            "original_data": os.path.join(mode_dir, f"video_text_alignment_{mode}.jsonl"),
            "culture_relevance": os.path.join(mode_dir, "vlm_eval_culture_relevance_log.jsonl"),
            "visual_similarity": os.path.join(mode_dir, "vlm_visual_similarity_log.jsonl"),
            "text_alignment": os.path.join(mode_dir, "vlm_text_image_alignment_log.jsonl"),
            "output": os.path.join(mode_dir, f"vlm_{mode}.jsonl")
        }

def load_jsonl_by_video_id(file_path, video_id_field="video_id"):
    """Load JSONL file and return dictionary keyed by video_id"""
    data = {}
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return data
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                video_id = entry.get(video_id_field)
                if video_id:
                    data[video_id] = entry
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num} in {file_path}: {e}")
    
    return data

def combine_base_mode(file_paths, max_lines=None):
    """Combine VLM results for base mode"""
    print("Processing base mode...")
    
    # Load original data
    print(f"Loading original data: {file_paths['original_data']}")
    original_data = load_jsonl_by_video_id(file_paths["original_data"])
    print(f"Loaded {len(original_data)} original entries")
    
    # Apply max_lines limit if specified
    if max_lines is not None and max_lines > 0:
        original_items = list(original_data.items())[:max_lines]
        original_data = dict(original_items)
        print(f"Limited to first {len(original_data)} entries due to max_lines={max_lines}")
    
    # Load cultural relevance scores
    print(f"Loading cultural relevance scores: {file_paths['culture_relevance']}")
    culture_data = load_jsonl_by_video_id(file_paths["culture_relevance"])
    print(f"Loaded {len(culture_data)} cultural relevance entries")
    
    # Combine data
    combined_entries = []
    matched_count = 0
    
    for video_id, original_entry in original_data.items():
        combined_entry = original_entry.copy()
        
        # Add cultural relevance scores
        if video_id in culture_data:
            culture_entry = culture_data[video_id]
            # Add VLM cultural relevance scores (scores only)
            combined_entry.update({
                "vlm_overall_score": culture_entry.get("overall_score", 0),
                "vlm_person_score": culture_entry.get("person_score", 0),
                "vlm_action_score": culture_entry.get("action_score", 0),
                "vlm_location_score": culture_entry.get("location_score", 0)
            })
            matched_count += 1
        else:
            print(f"Warning: No cultural relevance data found for video_id: {video_id}")
        
        combined_entries.append(combined_entry)
    
    print(f"Combined {matched_count}/{len(original_data)} entries with cultural relevance scores")
    return combined_entries

def combine_other_mode(file_paths, mode, max_lines=None):
    """Combine VLM results for non-base modes"""
    print(f"Processing {mode} mode...")
    
    # Load original data
    print(f"Loading original data: {file_paths['original_data']}")
    original_data = load_jsonl_by_video_id(file_paths["original_data"])
    print(f"Loaded {len(original_data)} original entries")
    
    # Apply max_lines limit if specified
    if max_lines is not None and max_lines > 0:
        original_items = list(original_data.items())[:max_lines]
        original_data = dict(original_items)
        print(f"Limited to first {len(original_data)} entries due to max_lines={max_lines}")
    
    # Load all VLM evaluation results
    print(f"Loading cultural relevance scores: {file_paths['culture_relevance']}")
    culture_data = load_jsonl_by_video_id(file_paths["culture_relevance"])
    print(f"Loaded {len(culture_data)} cultural relevance entries")
    
    print(f"Loading visual similarity scores: {file_paths['visual_similarity']}")
    visual_data = load_jsonl_by_video_id(file_paths["visual_similarity"])
    print(f"Loaded {len(visual_data)} visual similarity entries")
    
    print(f"Loading text-image alignment scores: {file_paths['text_alignment']}")
    text_alignment_data = load_jsonl_by_video_id(file_paths["text_alignment"])
    print(f"Loaded {len(text_alignment_data)} text-image alignment entries")
    
    # Combine data
    combined_entries = []
    culture_matched = 0
    visual_matched = 0
    text_matched = 0
    
    for video_id, original_entry in original_data.items():
        combined_entry = original_entry.copy()
        
        # Sanity check: verify original_prompt consistency across all evaluation files
        original_prompt = original_entry.get("original_prompt", "")
        
        # Add cultural relevance scores
        if video_id in culture_data:
            culture_entry = culture_data[video_id]
            culture_prompt = culture_entry.get("original_prompt", "")
            if culture_prompt and culture_prompt != original_prompt:
                raise ValueError(f"Mismatched original_prompt for video_id {video_id}: "
                               f"original='{original_prompt}' vs culture='{culture_prompt}'")
            
            combined_entry.update({
                "vlm_overall_score": culture_entry.get("overall_score", 0),
                "vlm_person_score": culture_entry.get("person_score", 0),
                "vlm_action_score": culture_entry.get("action_score", 0),
                "vlm_location_score": culture_entry.get("location_score", 0)
            })
            culture_matched += 1
        
        # Add visual similarity scores
        if video_id in visual_data:
            visual_entry = visual_data[video_id]
            visual_prompt = visual_entry.get("original_prompt", "")
            if visual_prompt and visual_prompt != original_prompt:
                raise ValueError(f"Mismatched original_prompt for video_id {video_id}: "
                               f"original='{original_prompt}' vs visual='{visual_prompt}'")
            
            combined_entry.update({
                "vlm_visual_similarity_score": visual_entry.get("score", 0)
            })
            visual_matched += 1
        
        # Add text-image alignment scores (scores only)
        if video_id in text_alignment_data:
            text_entry = text_alignment_data[video_id]
            text_prompt = text_entry.get("original_prompt", "")
            if text_prompt and text_prompt != original_prompt:
                raise ValueError(f"Mismatched original_prompt for video_id {video_id}: "
                               f"original='{original_prompt}' vs text_alignment='{text_prompt}'")
            
            # Add all the text-image alignment score fields
            for key, value in text_entry.items():
                if key.endswith("_score") and (key.startswith("v_base_") or key.startswith("v_var_")):
                    combined_entry[f"vlm_{key}"] = value
            text_matched += 1
        
        combined_entries.append(combined_entry)
    
    print(f"Combined {culture_matched}/{len(original_data)} entries with cultural relevance scores")
    print(f"Combined {visual_matched}/{len(original_data)} entries with visual similarity scores")
    print(f"Combined {text_matched}/{len(original_data)} entries with text-image alignment scores")
    
    return combined_entries

def save_combined_results(combined_entries, output_path):
    """Save combined results to JSONL file"""
    print(f"Saving {len(combined_entries)} combined entries to: {output_path}")
    
    with open(output_path, 'w') as f:
        for entry in combined_entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Successfully saved combined results to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Combine VLM evaluation results for a given mode")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["base", "single", "sequential", "parallel"],
                        help="Processing mode")
    parser.add_argument("--max_lines", type=int, default=None,
                        help="Maximum number of lines to process (default: process all lines)")
    args = parser.parse_args()
    
    print("VLM Results Combiner")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    if args.max_lines:
        print(f"Max lines: {args.max_lines}")
    
    # Get file paths
    file_paths = get_file_paths(args.mode)
    
    # Check if original data file exists
    if not os.path.exists(file_paths["original_data"]):
        print(f"Error: Original data file not found: {file_paths['original_data']}")
        return
    
    # Combine results based on mode
    if args.mode == "base":
        combined_entries = combine_base_mode(file_paths, args.max_lines)
    else:
        combined_entries = combine_other_mode(file_paths, args.mode, args.max_lines)
    
    # Save combined results
    save_combined_results(combined_entries, file_paths["output"])
    
    print("\nCombination completed successfully!")

if __name__ == "__main__":
    main()
