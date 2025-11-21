#!/usr/bin/env python3
"""
EvalCrafter Video Metrics Runner
===============================

This script prepares data for EvalCrafter and runs video evaluation metrics.

For base mode:
- Input: video_similarity_scores_base.jsonl
- Copies final_prompt to my_prompts/0000.txt
- Copies video to my_videos/0000.mp4
- Runs EvalCrafter evaluation
- Extracts 12 metrics from results
- Appends metrics to JSONL entries

Features:
- Supports partitioned processing for parallel runs
- Uses specified GPU for evaluation
- Logs metrics to tab-separated file
- Processes videos one at a time with cleanup
- Resumable: skips already completed entries based on log file

Usage:
    # Process all videos
    python evaluations/ec_video/run_evalcrafter.py --mode base --gpu 0

    # Process partition 1 of 2 on GPU 0
    python evaluations/ec_video/run_evalcrafter.py --mode base --partition 21 --gpu 0
    python evaluations/ec_video/run_evalcrafter.py --mode base --partition 22 --gpu 1

    # Process partition 1 of 10 on GPU 0
    python evaluations/ec_video/run_evalcrafter.py --mode base --partition 101 --gpu 0
    python evaluations/ec_video/run_evalcrafter.py --mode base --partition 102 --gpu 1
    ...
    python evaluations/ec_video/run_evalcrafter.py --mode base --partition 1010 --gpu 9
"""
import argparse
import json
import subprocess
import shutil
import re
from pathlib import Path
import sys
import os
import time

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
EVALCRAFTER_DIR = Path("/workspace/t2v_secret/evaluation/video/EvalCrafter")
MODES = ["base", "single", "parallel", "sequential"]


def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """Save list of dictionaries to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def prepare_evalcrafter_single_video(prompt, video_path, gpu_id=0):
    """Prepare a single video and prompt for EvalCrafter evaluation.

    Uses GPU-specific directories to avoid race conditions when running
    multiple partitions in parallel on different GPUs.
    """
    prompts_dir = EVALCRAFTER_DIR / f"my_prompts_gpu{gpu_id}"
    videos_dir = EVALCRAFTER_DIR / f"my_videos_gpu{gpu_id}"

    # Create directories
    prompts_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)

    # Clear existing files
    for f in prompts_dir.glob("*"):
        f.unlink()
    for f in videos_dir.glob("*"):
        f.unlink()

    # Always use 0000 as filename
    prompt_file = prompts_dir / "0000.txt"
    video_file = videos_dir / "0000.mp4"

    # Write prompt to file
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)

    # Copy video file
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    shutil.copy2(video_path, video_file)


def run_evalcrafter(gpu_id=0):
    """Run EvalCrafter evaluation.

    Uses GPU-specific directories to avoid race conditions when running
    multiple partitions in parallel on different GPUs.

    Returns:
        tuple: (stdout, stderr) - The captured output from the evaluation
    """
    # Remove existing GPU-specific results directory
    results_dir = EVALCRAFTER_DIR / f"results_gpu{gpu_id}"
    if results_dir.exists():
        print(f"Removing existing results directory for GPU {gpu_id}...")
        shutil.rmtree(results_dir)

    # Create GPU-specific results directory
    results_dir.mkdir(exist_ok=True)

    # Change to EvalCrafter directory
    original_cwd = os.getcwd()
    os.chdir(EVALCRAFTER_DIR)

    try:
        print(f"Running EvalCrafter evaluation on GPU {gpu_id}...")
        cmd = [
            "bash", "t2v_eval_video_metrics_2.sh",
            str(EVALCRAFTER_DIR),
            str(EVALCRAFTER_DIR / f"my_videos_gpu{gpu_id}"),
            str(EVALCRAFTER_DIR / f"my_prompts_gpu{gpu_id}"),
            str(gpu_id)  # Pass GPU ID to script
        ]

        # Set environment variable with specified GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Run the command - merge stderr into stdout to preserve chronological order
        result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
        print("EvalCrafter evaluation completed successfully")
        # Show a preview of the output
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) > 10:
            print("Output preview (last 10 lines):")
            for line in output_lines[-10:]:
                print(f"  {line}")
        else:
            print("Output:", result.stdout)

        return result.stdout

    except subprocess.CalledProcessError as e:
        print(f"Error running EvalCrafter: {e}")
        print("Output:", e.stdout)
        raise
    finally:
        # Change back to original directory
        os.chdir(original_cwd)


def archive_evalcrafter_results(entry_index, video_name, gpu_id=0, console_log=""):
    """Archive all EvalCrafter results for a video to permanent storage.

    Saves results_gpu, SDXL_Imgs, DOVER file, and console logs before they get cleaned up.

    Args:
        entry_index: Absolute index of the video in the dataset
        video_name: Name of the video file (without extension)
        gpu_id: GPU ID used for evaluation
        console_log: Captured console output from EvalCrafter run (merged stdout+stderr)
    """
    # Create archive directory structure
    archive_base = EVALCRAFTER_DIR / "archived_results"
    video_archive = archive_base / f"video_{entry_index:04d}_{video_name}"
    video_archive.mkdir(parents=True, exist_ok=True)

    print(f"Archiving results to: {video_archive}")

    # 1. Archive results_gpu directory
    results_gpu_dir = EVALCRAFTER_DIR / f"results_gpu{gpu_id}"
    if results_gpu_dir.exists():
        archive_results = video_archive / "results"
        if archive_results.exists():
            shutil.rmtree(archive_results)
        shutil.copytree(results_gpu_dir, archive_results)
        print(f"  ✓ Archived results directory")

    # 2. Archive SDXL_Imgs for this video
    sdxl_dir = EVALCRAFTER_DIR / f"SDXL_Imgs_gpu{gpu_id}"
    if sdxl_dir.exists():
        # Copy only the images for this video (named 0000_*.png)
        archive_sdxl = video_archive / "SDXL_Imgs"
        archive_sdxl.mkdir(exist_ok=True)
        for img_file in sdxl_dir.glob("0000_*.png"):
            shutil.copy2(img_file, archive_sdxl / img_file.name)
        if list(archive_sdxl.glob("*.png")):
            print(f"  ✓ Archived SDXL images")

    # 3. Archive DOVER results file
    dover_file = EVALCRAFTER_DIR / "metrics" / "DOVER" / f"zero_shot_res_sensehdr_gpu{gpu_id}.txt"
    if dover_file.exists():
        # Copy the entire DOVER file
        archive_dover = video_archive / "dover_results.txt"
        shutil.copy2(dover_file, archive_dover)
        print(f"  ✓ Archived DOVER results")

    # 4. Archive console logs (in chronological order)
    if console_log:
        console_log_file = video_archive / "console_output.log"
        with open(console_log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EvalCrafter Console Output (chronological order)\n")
            f.write("=" * 80 + "\n\n")
            f.write(console_log)
            if not console_log.endswith('\n'):
                f.write("\n")
        print(f"  ✓ Archived console logs")

    print(f"Archive complete for video {entry_index}")


def parse_evalcrafter_results(gpu_id=0):
    """Parse EvalCrafter results and extract 12 metrics.

    Uses GPU-specific results directory to avoid race conditions when running
    multiple partitions in parallel on different GPUs.
    """
    results_file = EVALCRAFTER_DIR / f"results_gpu{gpu_id}" / "final_result.txt"

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, 'r') as f:
        content = f.read()

    print(f"Results file content:\n{content}")
    
    # Extract metrics from the first line (10 metrics)
    metrics_pattern = r"Metrics: ({.*?})"
    metrics_match = re.search(metrics_pattern, content)
    
    if not metrics_match:
        raise ValueError("Could not find metrics in results file")
    
    # Parse the metrics dictionary
    metrics_str = metrics_match.group(1)
    # Clean up the string and evaluate as Python dict
    metrics_str = metrics_str.replace("'", '"').replace('nan', 'null')
    
    try:
        import json
        metrics_dict = json.loads(metrics_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse metrics dictionary: {e}")
    
    # Extract the 10 specific metrics from first line
    first_line_metrics = [
        'VQA_A', 'VQA_T', 'IS', 'clip_temp_score', 'warping_error', 
        'face_consistency_score', 'flow_score', 'clip_score', 'blip_bleu', 'sd_score'
    ]
    
    # Extract metrics from the second line (2 metrics)
    results_pattern = r"Results: Visual Quality ([\d\.-]+), Text-Video Alignment ([\w\.-]+), Motion Quality ([\w\.-]+), Temporal Consistency ([\d\.-]+), Total ([\w\.-]+)"
    results_match = re.search(results_pattern, content)
    
    if not results_match:
        raise ValueError("Could not find results in results file")
    
    visual_quality = results_match.group(1)
    temporal_consistency = results_match.group(4)
    
    # Convert to float, handling 'nan' strings
    def safe_float(value):
        if value == 'nan' or value is None:
            return float('nan')
        return float(value)
    
    # Collect all 12 metrics
    all_metrics = {}
    
    # Add 10 metrics from first line
    for metric in first_line_metrics:
        value = metrics_dict.get(metric)
        all_metrics[metric] = safe_float(value) if value is not None else float('nan')
    
    # Add 2 metrics from second line
    all_metrics['Visual_Quality'] = safe_float(visual_quality)
    all_metrics['Temporal_Consistency'] = safe_float(temporal_consistency)
    
    # Check for NaN values
    nan_metrics = [k for k, v in all_metrics.items() if str(v) == 'nan']
    if nan_metrics:
        raise ValueError(f"Found NaN values in metrics: {nan_metrics}")
    
    print("\nExtracted metrics:")
    for metric, value in all_metrics.items():
        print(f"  {metric}: {value}")
    
    return all_metrics


def load_existing_results(output_file):
    """Load existing results from output JSONL file."""
    existing_results = {}
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        entry = json.loads(line)
                        existing_results[idx] = entry
            print(f"Loaded {len(existing_results)} existing results from {output_file}")
        except Exception as e:
            print(f"Warning: Could not read existing results file: {e}")
    return existing_results


def load_completed_entries(log_file):
    """Load already completed entries from log file (JSONL format)."""
    completed_indices = set()
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        # Only consider entries that have metrics (not FAILED)
                        if 'status' not in entry or entry['status'] == 'success':
                            completed_indices.add(entry['entry_index'])
        except Exception as e:
            print(f"Warning: Could not read existing log file: {e}")
    return completed_indices


def main():
    parser = argparse.ArgumentParser(description="Run EvalCrafter video evaluation")
    parser.add_argument("--mode", choices=MODES, required=True, help="Evaluation mode")
    parser.add_argument("--partition", type=int, help="Partition as concatenation of total_partitions and partition_index (1-based). Examples: 21 (2 total, partition 1), 101 (10 total, partition 1), 1010 (10 total, partition 10)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    args = parser.parse_args()
    
    # Parse partition argument
    partition_idx = None
    total_partitions = None
    if args.partition is not None:
        if args.partition < 10:
            print("Error: Partition format should be concatenation of total_partitions and partition_index")
            print("Examples: 21 (2 total, partition 1), 101 (10 total, partition 1), 1010 (10 total, partition 10)")
            sys.exit(1)

        partition_str = str(args.partition)

        # Try different split points to find a valid partition format
        # For "101", try: "1"+"01", "10"+"1"
        # For "1010", try: "1"+"010", "10"+"10", "101"+"0"
        valid_split = None
        for split_pos in range(1, len(partition_str)):
            try:
                total_str = partition_str[:split_pos]
                idx_str = partition_str[split_pos:]

                total = int(total_str)
                idx = int(idx_str)

                # Validate: total >= 2, idx >= 1 and idx <= total
                if total >= 2 and idx >= 1 and idx <= total:
                    valid_split = (total, idx)
                    break
            except (ValueError, IndexError):
                continue

        if valid_split is None:
            print(f"Error: Could not parse partition format '{partition_str}'")
            print("Partition format should be concatenation of total_partitions and partition_index")
            print("Examples: 21 (2 total, partition 1), 101 (10 total, partition 1), 1010 (10 total, partition 10)")
            sys.exit(1)

        total_partitions, partition_idx = valid_split
        partition_idx = partition_idx - 1  # Convert to 0-based

        print(f"Using partition {partition_idx + 1} of {total_partitions}")

    # Determine input and output files based on mode
    if args.mode == "base":
        input_file = RESULTS_DIR / "base" / "video_similarity_scores_base.jsonl"
        output_file = RESULTS_DIR / "base" / "video_similarity_scores_base_with_ec.jsonl"
    else:
        input_file = RESULTS_DIR / args.mode / f"video_text_alignment_{args.mode}.jsonl"
        output_file = RESULTS_DIR / args.mode / f"video_text_alignment_{args.mode}_with_ec.jsonl"

    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Check if EvalCrafter directory exists
    if not EVALCRAFTER_DIR.exists():
        print(f"Error: EvalCrafter directory not found: {EVALCRAFTER_DIR}")
        sys.exit(1)

    print(f"Loading data from: {input_file}")
    data = load_jsonl(input_file)
    
    if not data:
        print("Error: No data found in input file")
        sys.exit(1)

    # Apply partitioning if specified
    # Track the absolute start index for logging
    absolute_start_idx = 0

    if partition_idx is not None and total_partitions is not None:
        total_items = len(data)
        items_per_partition = total_items // total_partitions
        remainder = total_items % total_partitions

        # Calculate start and end indices for this partition
        start_idx = partition_idx * items_per_partition
        if partition_idx < remainder:
            start_idx += partition_idx
            end_idx = start_idx + items_per_partition + 1
        else:
            start_idx += remainder
            end_idx = start_idx + items_per_partition

        # Store absolute start index before slicing
        absolute_start_idx = start_idx

        data = data[start_idx:end_idx]
        print(f"Processing partition {partition_idx + 1}/{total_partitions}: items {start_idx}-{end_idx-1} ({len(data)} items)")

        # Update output filename to include partition info
        if args.mode == "base":
            output_file = RESULTS_DIR / "base" / f"video_similarity_scores_base_with_ec_part{partition_idx + 1}.jsonl"
        else:
            output_file = RESULTS_DIR / args.mode / f"video_text_alignment_{args.mode}_with_ec_part{partition_idx + 1}.jsonl"
    else:
        print(f"Processing all {len(data)} items")

    # Set up logging file
    log_file = RESULTS_DIR / args.mode / f"evalcrafter_metrics_log.txt"
    if partition_idx is not None:
        log_file = RESULTS_DIR / args.mode / f"evalcrafter_metrics_log_part{partition_idx + 1}.txt"

    # Create results directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Clear/override the log file at the start of each run
    if log_file.exists():
        log_file.unlink()
        print(f"Cleared existing log file: {log_file}")

    # Load already completed entries for resumability (from output file only)
    existing_results = load_existing_results(output_file)

    # Since we cleared the log file, load completed indices from output file instead
    completed_indices = set(existing_results.keys()) if existing_results else set()

    if completed_indices:
        print(f"Found {len(completed_indices)} already completed entries in output file, will skip them")

    # Process each entry individually
    results = []

    # Open log file for writing (will create new file)
    with open(log_file, 'w', encoding='utf-8') as log_f:
        for idx, entry in enumerate(data):
            # Calculate absolute index for logging
            absolute_idx = absolute_start_idx + idx

            # Check if this entry has already been completed
            if absolute_idx in completed_indices:
                print(f"Skipping entry {absolute_idx+1} (already completed)")
                # Use existing result if available, otherwise use original entry
                if absolute_idx in existing_results:
                    result_entry = existing_results[absolute_idx]
                    results.append(result_entry)
                    # Log the already-completed entry to the fresh log file
                    log_entry = {
                        "entry_index": absolute_idx,
                        "video_path": result_entry.get("video_path", ""),
                        "prompt": result_entry.get("final_prompt", ""),
                        "status": "skipped_already_completed",
                        "metrics": {k.replace("ec_", ""): v for k, v in result_entry.items() if k.startswith("ec_")} if any(k.startswith("ec_") for k in result_entry.keys()) else None
                    }
                    log_f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                    log_f.flush()
                else:
                    results.append(entry.copy())
                continue
                
            print(f"\nProcessing entry {absolute_idx+1}")

            # Start timing
            start_time = time.time()

            # Get prompt and video path
            if args.mode == "base":
                prompt = entry.get("final_prompt", "")
            else:
                prompt = entry.get("final_prompt", "")

            video_path = entry.get("video_path", "")

            # Fix path if it contains the old workspace name
            if video_path and "/workspace/t2v_self/" in video_path:
                video_path = video_path.replace("/workspace/t2v_self/", "/workspace/t2v_secret/")

            if not prompt or not video_path:
                print(f"Warning: Missing prompt or video path for entry {absolute_idx}, skipping...")
                # Add entry without EC metrics
                results.append(entry.copy())
                # Log failure in JSONL format
                elapsed_time = time.time() - start_time
                log_entry = {
                    "entry_index": absolute_idx,
                    "video_path": video_path,
                    "prompt": prompt,
                    "status": "failed",
                    "error": "Missing data",
                    "elapsed_time_seconds": round(elapsed_time, 2)
                }
                log_f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                log_f.flush()
                continue

            if not Path(video_path).exists():
                print(f"Warning: Video file not found: {video_path}, skipping...")
                # Add entry without EC metrics
                results.append(entry.copy())
                # Log failure in JSONL format
                elapsed_time = time.time() - start_time
                log_entry = {
                    "entry_index": absolute_idx,
                    "video_path": video_path,
                    "prompt": prompt,
                    "status": "failed",
                    "error": "Video not found",
                    "elapsed_time_seconds": round(elapsed_time, 2)
                }
                log_f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                log_f.flush()
                continue

            try:
                # Prepare single video for EvalCrafter
                prepare_evalcrafter_single_video(prompt, video_path, args.gpu)

                # Run EvalCrafter evaluation and capture logs
                console_log = run_evalcrafter(args.gpu)

                # Parse and validate results
                metrics = parse_evalcrafter_results(args.gpu)

                # Archive results before they get cleaned up
                video_name = Path(video_path).stem  # Get filename without extension
                archive_evalcrafter_results(absolute_idx, video_name, args.gpu, console_log)

                # Calculate elapsed time
                elapsed_time = time.time() - start_time

                # Add EC metrics to entry
                result_entry = entry.copy()
                for metric, value in metrics.items():
                    result_entry[f"ec_{metric}"] = value

                results.append(result_entry)
                print(f"Successfully processed entry {absolute_idx+1} in {elapsed_time:.2f} seconds")

                # Log metrics in JSONL format with full prompt
                log_entry = {
                    "entry_index": absolute_idx,
                    "video_path": video_path,
                    "prompt": prompt,
                    "status": "success",
                    "metrics": metrics,
                    "elapsed_time_seconds": round(elapsed_time, 2)
                }
                log_f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                log_f.flush()

            except Exception as e:
                print(f"Error processing entry {absolute_idx+1}: {e}")
                # Add entry without EC metrics
                results.append(entry.copy())
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                # Log failure in JSONL format
                log_entry = {
                    "entry_index": absolute_idx,
                    "video_path": video_path,
                    "prompt": prompt,
                    "status": "failed",
                    "error": str(e),
                    "elapsed_time_seconds": round(elapsed_time, 2)
                }
                log_f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                log_f.flush()
                continue
    
    # Save results
    # TEMPORARILY COMMENTED OUT - not saving output JSONL with EC metrics
    # save_jsonl(results, output_file)
    # print(f"\nSaved results to: {output_file}")
    print(f"\nSkipped saving output JSONL (temporarily disabled)")
    print(f"Saved metrics log to: {log_file}")
    print(f"Processed {len(results)} entries total")
    if partition_idx is not None:
        print(f"Partition {partition_idx + 1}/{total_partitions} completed")


if __name__ == "__main__":
    main()
