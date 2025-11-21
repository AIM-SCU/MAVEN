#!/usr/bin/env python3
"""
Runner Script for Evaluation Pipeline
====================================

This script runs the evaluation pipeline in three steps:
1. Merge JSONL files
2. Add cultural relevance
3. Compute CLIP score

Usage:
    CUDA_VISIBLE_DEVICES=0 python evaluations/runner.py --mode base
    CUDA_VISIBLE_DEVICES=0 python evaluations/runner.py --mode single
    CUDA_VISIBLE_DEVICES=0 python evaluations/runner.py --mode parallel
    CUDA_VISIBLE_DEVICES=0 python evaluations/runner.py --mode sequential

Assumptions:
- The merge step uses merge_jsonl_files.py in this directory.
- The cultural relevance and CLIP score steps are placeholders; replace with actual script paths/commands.
"""
import argparse
import subprocess
from pathlib import Path

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR.parent / "results"

MODES = ["base", "single", "parallel", "sequential"]

# These are example file patterns; adjust as needed for your project
MODE_CONFIG = {
    "base": {
        "source": RESULTS_DIR / "base" / "prompts_base.jsonl",
        "target": EVAL_DIR.parent / "templates" / "pal_prompts_v1.jsonl",
        "merged": RESULTS_DIR / "base" / "prompts_base_merged.jsonl",
    },
    "single": {
        "source": RESULTS_DIR / "single" / "prompts_single.jsonl",
        "target": EVAL_DIR.parent / "templates" / "pal_prompts_v1.jsonl",
        "merged": RESULTS_DIR / "single" / "prompts_single_merged.jsonl",
    },
    "parallel": {
        "source": RESULTS_DIR / "parallel" / "prompts_parallel.jsonl",
        "target": EVAL_DIR.parent / "templates" / "pal_prompts_v1.jsonl",
        "merged": RESULTS_DIR / "parallel" / "prompts_parallel_merged.jsonl",
    },
    "sequential": {
        "source": RESULTS_DIR / "sequential" / "prompts_sequential.jsonl",
        "target": EVAL_DIR.parent / "templates" / "pal_prompts_v1.jsonl",
        "merged": RESULTS_DIR / "sequential" / "prompts_sequential_merged.jsonl",
    },
}

def run_merge(mode):
    cfg = MODE_CONFIG[mode]
    cmd = [
        "python3", str(EVAL_DIR / "merge_jsonl_files.py"),
        "--source", str(cfg["source"]),
        "--target", str(cfg["target"]),
        "--output", str(cfg["merged"]),
    ]
    print(f"[Step 1] Merging JSONL files for mode '{mode}'...")
    subprocess.run(cmd, check=True)

def run_cultural_relevance(mode):
    cfg = MODE_CONFIG[mode]
    input_path = cfg["merged"]
    output_path = input_path.parent / f"prompts_{mode}_cultural_relevance.jsonl"
    cmd = [
        "python3", str(EVAL_DIR / "add_cultural_relevance_fields.py"),
        "--input", str(input_path),
        "--output", str(output_path),
    ]
    print(f"[Step 2] Adding cultural relevance for mode '{mode}'...")
    subprocess.run(cmd, check=True)
    return output_path

def run_clip_score(mode, input_path):
    # TODO: 0625 [I changed output file name, haven't tested yet]
    output_path = input_path.parent / f"video_culture_relevance_{mode}.jsonl"
    cmd = [
        "python3", str(EVAL_DIR / "compute_video_cultural_similarity.py"),
        "--input", str(input_path),
        "--output", str(output_path),
        "--num-frames", "5"
    ]
    print(f"[Step 3] Computing CLIP score for mode '{mode}'...")
    subprocess.run(cmd, check=True)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline for different modes.")
    parser.add_argument("--mode", choices=MODES, default="base", help="Evaluation mode")
    args = parser.parse_args()

    if args.mode not in MODE_CONFIG:
        print(f"Error: Mode '{args.mode}' not configured.")
        return

    run_merge(args.mode)
    cultural_path = run_cultural_relevance(args.mode)
    run_clip_score(args.mode, cultural_path)
    print("Pipeline completed.")

if __name__ == "__main__":
    main()
