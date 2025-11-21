#!/usr/bin/env python3
"""
VLM Visual Similarity Evaluation Script

This script evaluates the visual similarity between video frames using Gemini Vision Language Model.
It compares frames from the base mode against frames from single/sequential/parallel modes.

Usage:
    python evaluations/vlm/vlm_eval_visual_similarity.py --mode single
    python evaluations/vlm/vlm_eval_visual_similarity.py --mode single --max_lines 2
    python evaluations/vlm/vlm_eval_visual_similarity.py --mode single --max_lines 1 --debug

    python evaluations/vlm/vlm_eval_visual_similarity.py --mode sequential --max_lines 5
    python evaluations/vlm/vlm_eval_visual_similarity.py --mode parallel

Arguments:
    --mode: Comparison mode (single, sequential, parallel) - always compared against base
    --max_lines: Maximum number of total lines to read from input file, including both skipped and processed entries (default: process all lines)
    --debug: Print Gemini responses for debugging (default: False)

For 'single' mode: Compares base vs single frames using video_text_alignment_single.jsonl
For 'sequential' mode: Compares base vs sequential frames using video_text_alignment_sequential.jsonl
For 'parallel' mode: Compares base vs parallel frames using video_text_alignment_parallel.jsonl

The script will:
1. Read the input JSONL file based on the mode
2. For each entry, extract the video_id and locate both base and comparison frame images
3. Build a prompt for visual similarity evaluation
4. Call Gemini API to evaluate visual similarity
5. Log detailed reasoning and scores
"""

import argparse
import os
import json
from google import genai
from google.genai import types

RESULTS_DIR = "/workspace/t2v_self/iter_t2v/results"

MODE_TO_FILE = {
    "single": os.path.join(RESULTS_DIR, "single/video_text_alignment_single.jsonl"),
    "sequential": os.path.join(RESULTS_DIR, "sequential/video_text_alignment_sequential.jsonl"),
    "parallel": os.path.join(RESULTS_DIR, "parallel/video_text_alignment_parallel.jsonl"),
}

BASE_FILE = os.path.join(RESULTS_DIR, "base/video_culture_relevance_base.jsonl")

def get_log_file_path(mode):
    return f"/workspace/t2v_self/iter_t2v/results/{mode}/vlm_visual_similarity_log.jsonl"

PROMPT_TEMPLATE = '''You will be shown two images. Your task is to evaluate how visually similar the second image is compared to the first image.

Consider all visible aspects, including the person (appearance, pose, clothing), the action (what is happening), the location (environment, background), and any cultural or stylistic details.

Please explain your reasoning step by step, and then assign a score between 1 and 5:

1 = Very different visually
5 = Nearly identical visually

image_1: (reference image)
image_2: (comparison image)
The output should be a JSON object ONLY with the following format:
{
  "reasoning": "...",
  "score": number
}'''

def check_required_files(mode):
    mode_file_path = MODE_TO_FILE.get(mode)
    if not mode_file_path or not os.path.exists(mode_file_path):
        raise FileNotFoundError(f"Required file for mode '{mode}' not found: {mode_file_path}")
    
    if not os.path.exists(BASE_FILE):
        raise FileNotFoundError(f"Required base file not found: {BASE_FILE}")
    
    return mode_file_path, BASE_FILE

def load_base_entries():
    """Load base entries into a dictionary keyed by original_prompt"""
    base_entries = {}
    with open(BASE_FILE, 'r') as f:
        for line in f:
            entry = json.loads(line)
            original_prompt = entry.get("original_prompt")
            if original_prompt:
                base_entries[original_prompt] = entry
    return base_entries

def get_frame_path(video_id, mode):
    frames_dir = os.path.join(RESULTS_DIR, mode, "frames")
    return os.path.join(frames_dir, video_id, "frame_0002.jpg")

def build_prompt():
    return PROMPT_TEMPLATE

def call_gemini_vlm(base_image_path, comparison_image_path, prompt, debug=False):
    """Call Gemini API for visual similarity evaluation with retry logic"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)

    # Read both image files
    with open(base_image_path, 'rb') as f:
        base_image_bytes = f.read()
    
    with open(comparison_image_path, 'rb') as f:
        comparison_image_bytes = f.read()

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',    # [gemini-2.5-pro]
                contents=[
                    types.Part.from_bytes(
                        data=base_image_bytes,
                        mime_type='image/jpeg',
                    ),
                    types.Part.from_bytes(
                        data=comparison_image_bytes,
                        mime_type='image/jpeg',
                    ),
                    prompt
                ]
            )

            # Extract and clean up text content
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[len("```json"):].strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()

            if debug:
                print(f"Gemini raw response (attempt {attempt + 1}):")
                print(response_text)

            result = json.loads(response_text)
            return result

        except (json.JSONDecodeError, AttributeError, IndexError, Exception) as e:
            if debug:
                print(f"Attempt {attempt + 1}/{max_retries} failed with error: {e}")
                if hasattr(response, 'text'):
                    print("Raw response:")
                    print(response.text)

            # If this is not the last attempt, continue to retry
            if attempt < max_retries - 1:
                continue
            
            # If all retries failed, return fallback dummy response
            if debug:
                print(f"All {max_retries} attempts failed. Returning fallback response.")

            return {
                "reasoning": f"Failed to parse response after {max_retries} attempts",
                "score": 1
            }

def get_processed_video_ids(log_file_path):
    """Read the log file and return a set of processed video IDs"""
    processed_ids = set()
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'video_id' in entry:
                        processed_ids.add(entry['video_id'])
                except json.JSONDecodeError:
                    continue
    return processed_ids

def main():
    parser = argparse.ArgumentParser(description="VLM Visual Similarity Evaluation Script")
    parser.add_argument("--mode", type=str, required=True, choices=["single", "sequential", "parallel"],
                        help="Comparison mode (single, sequential, parallel) - always compared against base")
    parser.add_argument("--max_lines", type=int, default=None,
                        help="Maximum number of total lines to read from input file, including both skipped and processed entries (default: process all lines)")
    parser.add_argument("--debug", action="store_true",
                        help="Print Gemini responses for debugging (default: False)")
    args = parser.parse_args()

    mode_file_path, base_file_path = check_required_files(args.mode)
    log_file = get_log_file_path(args.mode)
    
    print(f"Processing mode file: {mode_file_path}")
    print(f"Processing base file: {base_file_path}")
    print(f"Log file: {log_file}")
    print(f"Comparing base frames vs {args.mode} frames")

    # Load base entries first
    print("Loading base entries...")
    base_entries = load_base_entries()
    print(f"Loaded {len(base_entries)} base entries")

    # Read processed video_ids once
    processed_video_ids = get_processed_video_ids(log_file)
    print(f"Found {len(processed_video_ids)} already processed entries in log file.")

    with open(mode_file_path, "r") as fin, open(log_file, "a") as flog:
        processed_count = 0
        skipped_count = 0
        total_lines_read = 0
        for line in fin:
            if args.max_lines is not None and total_lines_read >= args.max_lines:
                break
            total_lines_read += 1

            entry = json.loads(line)
            video_id = entry.get("video_id")
            if not video_id:
                raise ValueError("Missing video_id in entry.")

            # Only process if video_id not already processed
            if video_id in processed_video_ids:
                print(f"Skipping already processed video_id: {video_id}")
                skipped_count += 1
                continue

            # Get original_prompt to find matching base entry
            mode_original_prompt = entry.get("original_prompt")
            if not mode_original_prompt:
                print(f"Warning: Missing original_prompt for video_id: {video_id}, skipping...")
                continue

            # Check if corresponding base entry exists (by original_prompt)
            if mode_original_prompt not in base_entries:
                print(f"Warning: No matching base entry found for original_prompt, skipping video_id: {video_id}")
                continue

            base_entry = base_entries[mode_original_prompt]
            base_video_id = base_entry.get("video_id")

            if not base_video_id:
                print(f"Warning: Missing video_id in base entry for original_prompt, skipping...")
                continue

            # Get both base and comparison image paths (using different video_ids)
            base_image_path = get_frame_path(base_video_id, "base")
            comparison_image_path = get_frame_path(video_id, args.mode)
            
            # Check if both images exist
            if not os.path.exists(base_image_path):
                print(f"Warning: Base frame not found: {base_image_path}, skipping...")
                continue
                
            if not os.path.exists(comparison_image_path):
                print(f"Warning: Comparison frame not found: {comparison_image_path}, skipping...")
                continue

            prompt = build_prompt()
            vlm_result = call_gemini_vlm(base_image_path, comparison_image_path, prompt, debug=args.debug)

            # Add metadata to the result
            vlm_result["video_id"] = video_id
            vlm_result["base_video_id"] = base_video_id
            vlm_result["base_image_path"] = base_image_path
            vlm_result["comparison_image_path"] = comparison_image_path
            vlm_result["comparison_mode"] = args.mode
            vlm_result["original_prompt"] = mode_original_prompt
            vlm_result["llm_prompt"] = prompt

            flog.write(json.dumps(vlm_result) + "\n")
            flog.flush()
            processed_video_ids.add(video_id)

            processed_count += 1
            if args.max_lines is not None:
                print(f"Read {total_lines_read}/{args.max_lines} lines - Processed {processed_count}, Skipped {skipped_count}")

        print(f"Total lines read: {total_lines_read}, Total processed: {processed_count}, Total skipped: {skipped_count}")

if __name__ == "__main__":
    main()
