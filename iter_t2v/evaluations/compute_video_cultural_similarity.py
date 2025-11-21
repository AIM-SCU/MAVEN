#!/usr/bin/env python3
"""
Video-Cultural Similarity Computation Script
============================================

This script computes similarity scores between video frames and cultural relevance texts.
For each video, it extracts 5 evenly spaced frames and computes similarity with 4 cultural
relevance text fields, resulting in 24 scores (5 frames × 4 texts + 4 averages).

Usage:
    python evaluations/compute_video_cultural_similarity.py --input /workspace/t2v_self/iter_t2v/evaluations/prompts_base_cultural_relevance.jsonl --output video_similarity_scores.jsonl --num-frames 5
    ### update 0625 [input and ouput file paths changed, have not tested yet]
    python evaluations/compute_video_cultural_similarity.py --input /workspace/t2v_self/iter_t2v/results/base/prompts_base_merged.jsonl --output video_culture_relevance_base.jsonl --num-frames 5
"""

import json
import argparse
import numpy as np
from typing import Dict, List
from pathlib import Path
import sys
import os

# Add the evaluations directory to path to import clip_video_similarity
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clip_video_similarity import CLIPVideoSimilarity


def process_video_entry(entry: Dict, clip_model: CLIPVideoSimilarity, num_frames: int = 5) -> Dict:
    """
    Process a single video entry and compute all similarity scores.
    
    Args:
        entry: JSONL entry dictionary
        clip_model: CLIPVideoSimilarity instance
        num_frames: Number of frames to extract
        
    Returns:
        Entry with added similarity scores
    """
    video_path = entry.get("video_path", "")
    
    if not video_path or not Path(video_path).exists():
        print(f"Warning: Video not found: {video_path}")
        # Add empty scores
        entry.update(get_empty_scores(num_frames))
        return entry
    
    # Extract cultural relevance texts
    cultural_texts = {
        "overall": entry.get("overall_cultural_relevance", ""),
        "person": entry.get("person_cultural_relevance", ""),
        "action": entry.get("action_cultural_relevance", ""),
        "location": entry.get("location_cultural_relevance", "")
    }
    
    # Check if cultural texts exist
    if not all(cultural_texts.values()):
        print(f"Warning: Missing cultural relevance texts for {video_path}")
        entry.update(get_empty_scores(num_frames))
        return entry
    
    # Compute similarities using existing CLIP functionality
    similarities = {}
    
    # For each cultural dimension
    for dim_name, text in cultural_texts.items():
        try:
            # Use the correct method name from clip_video_similarity.py
            frame_scores = clip_model.get_video_text_similarity(video_path, text, num_frames)
            
            if frame_scores is None or len(frame_scores) == 0:
                print(f"Warning: No similarity scores computed for {video_path} - {dim_name}")
                for frame_idx in range(num_frames):
                    similarities[f"{dim_name}_frame_{frame_idx}"] = 0.0
                similarities[f"{dim_name}_avg"] = 0.0
                continue
            
            for frame_idx, score in enumerate(frame_scores):
                similarities[f"{dim_name}_frame_{frame_idx}"] = float(score)
            for frame_idx in range(len(frame_scores), num_frames):
                similarities[f"{dim_name}_frame_{frame_idx}"] = 0.0
            similarities[f"{dim_name}_avg"] = float(np.mean(frame_scores))
            
        except Exception as e:
            print(f"Error computing similarity for {video_path} - {dim_name}: {e}")
            for frame_idx in range(num_frames):
                similarities[f"{dim_name}_frame_{frame_idx}"] = 0.0
            similarities[f"{dim_name}_avg"] = 0.0
    entry.update(similarities)
    return entry


def get_empty_scores(num_frames: int) -> Dict:
    """Generate empty similarity scores dictionary."""
    scores = {}
    
    dimensions = ["overall", "person", "action", "location"]
    
    for dim in dimensions:
        # Frame-level scores
        for frame_idx in range(num_frames):
            scores[f"{dim}_frame_{frame_idx}"] = 0.0
        
        # Average score
        scores[f"{dim}_avg"] = 0.0
    
    return scores


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save list of dictionaries to JSONL file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {len(data)} entries to {file_path}")
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")


def print_similarity_statistics(data: List[Dict], num_frames: int):
    """Print statistics about similarity scores."""
    dimensions = ["overall", "person", "action", "location"]
    
    print("\n" + "="*50)
    print("SIMILARITY SCORE STATISTICS")
    print("="*50)
    
    for dim in dimensions:
        avg_scores = [entry.get(f"{dim}_avg", 0) for entry in data if f"{dim}_avg" in entry]
        
        if avg_scores:
            print(f"\n{dim.upper()} DIMENSION:")
            print(f"  Mean: {np.mean(avg_scores):.3f}")
            print(f"  Std:  {np.std(avg_scores):.3f}")
            print(f"  Min:  {np.min(avg_scores):.3f}")
            print(f"  Max:  {np.max(avg_scores):.3f}")
    
    # Show example scores
    print(f"\nEXAMPLE SCORES (first entry):")
    if data:
        example = data[0]
        print(f"Video: {example.get('video_path', 'N/A')}")
        for dim in dimensions:
            avg_score = example.get(f"{dim}_avg", 0)
            print(f"  {dim.capitalize()} avg: {avg_score:.3f}")
            
            frame_scores = [example.get(f"{dim}_frame_{i}", 0) for i in range(num_frames)]
            print(f"  {dim.capitalize()} frames: {[f'{s:.3f}' for s in frame_scores]}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute similarity between video frames and cultural relevance texts"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file with cultural relevance texts")
    parser.add_argument("--output", required=True, help="Output JSONL file with similarity scores")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to extract per video (default: 5)")
    parser.add_argument("--device", default="auto", help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Set device properly before initializing CLIP model
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Initialize CLIP model using existing functionality
    print("Loading CLIP model...")
    try:
        clip_model = CLIPVideoSimilarity(device=device)
    except Exception as e:
        print(f"Error initializing CLIP model: {e}")
        return
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Load data
    print(f"Loading data from: {args.input}")
    data = load_jsonl(args.input)
    if not data:
        print("Error: Could not load data")
        return
    
    print(f"Processing {len(data)} videos with {args.num_frames} frames each...")
    
    # Process each entry
    processed_data = []
    for i, entry in enumerate(data):
        if i % 10 == 0:
            print(f"Processing video {i+1}/{len(data)}")
        
        processed_entry = process_video_entry(entry, clip_model, args.num_frames)
        processed_data.append(processed_entry)
    
    # Save processed data
    save_jsonl(processed_data, args.output)
    
    # Print statistics
    print_similarity_statistics(processed_data, args.num_frames)


if __name__ == "__main__":
    main()
