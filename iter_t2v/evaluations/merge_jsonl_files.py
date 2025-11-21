#!/usr/bin/env python3
"""
JSONL File Merger Script
========================

This script merges two JSONL files by matching the 'final_prompt' field from the source file
with the 'prompt' field from the target file. The merged output contains all fields from both files.

Usage:
    python utils/merge_jsonl_files.py --source /workspace/t2v_self/iter_t2v/results/base/prompts_base.jsonl --target /workspace/t2v_self/iter_t2v/templates/pal_prompts_v1.jsonl --output /workspace/t2v_self/iter_t2v/results/base/prompts_base_merged.jsonl
    python merge_jsonl_files.py --source source.jsonl --target target.jsonl --output merged.jsonl --merge-strategy append
"""

import json
import argparse
from typing import Dict, List, Optional, Set
from pathlib import Path
from collections import defaultdict


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
                    print(f"Warning: Invalid JSON on line {line_num} in {file_path}: {e}")
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


def create_target_lookup(target_data: List[Dict], target_field: str = "prompt") -> Dict[str, List[Dict]]:
    """
    Create a lookup dictionary for target data based on the target field.
    
    Args:
        target_data: List of target entries
        target_field: Field name to use as key (default: "prompt")
        
    Returns:
        Dictionary mapping prompt text to list of matching entries
    """
    lookup = defaultdict(list)
    
    for entry in target_data:
        if target_field in entry:
            prompt_text = entry[target_field]
            lookup[prompt_text].append(entry)
        else:
            print(f"Warning: '{target_field}' field missing in target entry")
    
    return dict(lookup)


def merge_entries(source_entry: Dict, target_entries: List[Dict], merge_strategy: str = "first") -> Dict:
    """
    Merge source entry with target entries.
    
    Args:
        source_entry: Source entry dictionary
        target_entries: List of matching target entries
        merge_strategy: How to handle multiple matches ("first", "last", "append", "merge_all")
        
    Returns:
        Merged dictionary
    """
    merged = source_entry.copy()
    
    if not target_entries:
        # No matches found
        merged["_merge_status"] = "no_match"
        return merged
    
    if merge_strategy == "first":
        # Use only the first matching target entry
        target_entry = target_entries[0]
        merged.update(target_entry)
        merged["_merge_status"] = "matched_first"
        if len(target_entries) > 1:
            merged["_multiple_matches"] = len(target_entries)
            
    elif merge_strategy == "last":
        # Use only the last matching target entry
        target_entry = target_entries[-1]
        merged.update(target_entry)
        merged["_merge_status"] = "matched_last"
        if len(target_entries) > 1:
            merged["_multiple_matches"] = len(target_entries)
            
    elif merge_strategy == "append":
        # Add target entries as a list
        merged["_target_matches"] = target_entries
        merged["_merge_status"] = f"matched_{len(target_entries)}"
        
    elif merge_strategy == "merge_all":
        # Merge all target entries, with later ones overriding earlier ones
        for target_entry in target_entries:
            merged.update(target_entry)
        merged["_merge_status"] = f"merged_{len(target_entries)}"
        if len(target_entries) > 1:
            merged["_multiple_matches"] = len(target_entries)
    
    return merged


def merge_jsonl_files(
    source_data: List[Dict], 
    target_data: List[Dict],
    source_field: str = "final_prompt",
    target_field: str = "prompt",
    merge_strategy: str = "first"
) -> List[Dict]:
    """
    Merge source and target JSONL data based on matching prompt fields.
    
    Args:
        source_data: Source JSONL data
        target_data: Target JSONL data
        source_field: Field name in source to match (default: "final_prompt")
        target_field: Field name in target to match (default: "prompt")
        merge_strategy: How to handle multiple matches
        
    Returns:
        List of merged entries
    """
    # Create lookup dictionary for target data
    target_lookup = create_target_lookup(target_data, target_field)
    
    merged_data = []
    stats = {
        "total_source": len(source_data),
        "matched": 0,
        "no_match": 0,
        "multiple_matches": 0,
        "missing_source_field": 0
    }
    
    for source_entry in source_data:
        if source_field not in source_entry:
            print(f"Warning: '{source_field}' field missing in source entry")
            stats["missing_source_field"] += 1
            merged_entry = source_entry.copy()
            merged_entry["_merge_status"] = "missing_source_field"
            merged_data.append(merged_entry)
            continue
        
        prompt_text = source_entry[source_field]
        target_entries = target_lookup.get(prompt_text, [])
        
        # Merge entries
        merged_entry = merge_entries(source_entry, target_entries, merge_strategy)
        merged_data.append(merged_entry)
        
        # Update statistics
        if target_entries:
            stats["matched"] += 1
            if len(target_entries) > 1:
                stats["multiple_matches"] += 1
        else:
            stats["no_match"] += 1
    
    return merged_data, stats


def print_merge_statistics(stats: Dict):
    """Print merge statistics."""
    print("\n" + "="*50)
    print("MERGE STATISTICS")
    print("="*50)
    print(f"Total source entries: {stats['total_source']}")
    print(f"Successfully matched: {stats['matched']}")
    print(f"No matches found: {stats['no_match']}")
    print(f"Multiple matches found: {stats['multiple_matches']}")
    print(f"Missing source field: {stats['missing_source_field']}")
    
    if stats['total_source'] > 0:
        match_rate = (stats['matched'] / stats['total_source']) * 100
        print(f"Match rate: {match_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Merge two JSONL files based on matching prompt fields"
    )
    parser.add_argument("--source", required=True, help="Source JSONL file (with 'original_prompt' field)")
    parser.add_argument("--target", required=True, help="Target JSONL file (with 'prompt' field)")
    parser.add_argument("--output", required=True, help="Output JSONL file for merged results")
    parser.add_argument("--source-field", default="original_prompt", 
                       help="Field name in source file (default: original_prompt)")
    parser.add_argument("--target-field", default="prompt",
                       help="Field name in target file (default: prompt)")
    parser.add_argument("--merge-strategy", default="first",
                       choices=["first", "last", "append", "merge_all"],
                       help="Strategy for handling multiple matches (default: first)")
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.source).exists():
        print(f"Error: Source file not found: {args.source}")
        return
    
    if not Path(args.target).exists():
        print(f"Error: Target file not found: {args.target}")
        return
    
    # Load data
    print(f"Loading source file: {args.source}")
    source_data = load_jsonl(args.source)
    if not source_data:
        print("Error: Could not load source data")
        return
    
    print(f"Loading target file: {args.target}")
    target_data = load_jsonl(args.target)
    if not target_data:
        print("Error: Could not load target data")
        return
    
    # Merge data
    print(f"Merging files using strategy: {args.merge_strategy}")
    merged_data, stats = merge_jsonl_files(
        source_data, 
        target_data,
        source_field=args.source_field,
        target_field=args.target_field,
        merge_strategy=args.merge_strategy
    )
    
    # Save merged data
    save_jsonl(merged_data, args.output)
    
    # Print statistics
    print_merge_statistics(stats)


if __name__ == "__main__":
    main()
