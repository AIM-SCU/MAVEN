#!/usr/bin/env python3
"""
VLM Text-Image Alignment Evaluation Script

This script evaluates the alignment between text descriptions and video frames using Gemini Vision Language Model.
It takes a mode parameter and processes text-image alignment data to generate VLM scores.

Usage:
    python evaluations/vlm/vlm_eval_text_image_alignment.py --mode single
    python evaluations/vlm/vlm_eval_text_image_alignment.py --mode single --max_lines 2
    python evaluations/vlm/vlm_eval_text_image_alignment.py --mode single --max_lines 1 --debug

    python evaluations/vlm/vlm_eval_text_image_alignment.py --mode single --max_lines 5
    python evaluations/vlm/vlm_eval_text_image_alignment.py --mode sequential
    python evaluations/vlm/vlm_eval_text_image_alignment.py --mode parallel

Arguments:
    --mode: Processing mode (single, sequential, parallel)
    --max_lines: Maximum number of total lines to read from input file, including both skipped and processed entries (default: process all lines)
    --debug: Print Gemini responses for debugging (default: False)

For 'single' mode: Checks for video_text_alignment_single.jsonl  
For 'sequential' mode: Checks for video_text_alignment_sequential.jsonl
For 'parallel' mode: Checks for video_text_alignment_parallel.jsonl

The script will:
1. Read the input JSONL file based on the mode
2. For each entry, extract the video_id and locate the corresponding frame image
3. Build a prompt for text-image alignment evaluation
4. Call Gemini API to evaluate text-image alignment
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
    return f"/workspace/t2v_self/iter_t2v/results/{mode}/vlm_text_image_alignment_log.jsonl"

PROMPT_TEMPLATE = '''You will be shown one image and multiple sentences that describe what the image is expected to show.
Your task is to evaluate how well the visual content of the image aligns with each description.

Focus on whether the image clearly reflects the elements described — such as people, actions, locations, objects, or any culturally or visually specific features.

For each sentence, please explain your reasoning step by step, and assign a score between 1 and 5, where:

1 = The image does not match the description
5 = The image clearly and fully matches the description

Please evaluate all sentences and return your response as a JSON object with this exact format:
{
  "text1_reasoning": "...",
  "text1_score": number,
  "text2_reasoning": "...", 
  "text2_score": number,
  "text3_reasoning": "...",
  "text3_score": number,
  "text4_reasoning": "...",
  "text4_score": number,
  "text5_reasoning": "...",
  "text5_score": number,
  "text6_reasoning": "...",
  "text6_score": number
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

def build_prompt(text_descriptions):
    """Build prompt with multiple text descriptions"""
    prompt = PROMPT_TEMPLATE + "\n\n"
    for i, text in enumerate(text_descriptions, 1):
        prompt += f"text{i}: \"{text}\"\n"
    return prompt

def evaluate_image_with_multiple_texts(image_path, text_descriptions, image_type, debug=False):
    """Evaluate a single image with multiple text descriptions"""
    prompt = build_prompt(text_descriptions)
    vlm_result = call_gemini_vlm(image_path, prompt, debug=debug)
    
    # Add evaluation metadata
    vlm_result["image_type"] = image_type
    vlm_result["text_descriptions"] = text_descriptions
    vlm_result["image_path"] = image_path
    vlm_result["llm_prompt"] = prompt
    
    return vlm_result

def call_gemini_vlm(image_path, prompt, debug=False):
    """Call Gemini API for text-image alignment evaluation with retry logic"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)

    # Read the image file
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',    # [gemini-2.5-pro]
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
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
                "text1_reasoning": f"Failed to parse response after {max_retries} attempts",
                "text1_score": 1,
                "text2_reasoning": f"Failed to parse response after {max_retries} attempts", 
                "text2_score": 1,
                "text3_reasoning": f"Failed to parse response after {max_retries} attempts",
                "text3_score": 1,
                "text4_reasoning": f"Failed to parse response after {max_retries} attempts",
                "text4_score": 1,
                "text5_reasoning": f"Failed to parse response after {max_retries} attempts",
                "text5_score": 1,
                "text6_reasoning": f"Failed to parse response after {max_retries} attempts",
                "text6_score": 1
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
    parser = argparse.ArgumentParser(description="VLM Text-Image Alignment Evaluation Script")
    parser.add_argument("--mode", type=str, required=True, choices=["single", "sequential", "parallel"],
                        help="Processing mode (single, sequential, parallel)")
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

            # Get both base and mode image paths
            base_image_path = get_frame_path(base_video_id, "base")
            mode_image_path = get_frame_path(video_id, args.mode)
            
            # Check if both images exist
            if not os.path.exists(base_image_path):
                print(f"Warning: Base frame not found: {base_image_path}, skipping...")
                continue
                
            if not os.path.exists(mode_image_path):
                print(f"Warning: Mode frame not found: {mode_image_path}, skipping...")
                continue

            # Perform batched evaluations (2 API calls instead of 12)
            all_results = {
                "video_id": video_id,
                "base_video_id": base_video_id,
                "comparison_mode": args.mode,
                "original_prompt": mode_original_prompt,
                "base_image_path": base_image_path,
                "mode_image_path": mode_image_path
            }
            
            # Prepare text descriptions for v_base image (6 texts)
            v_base_texts = []
            v_base_text_labels = []
            
            # 1-4: Cultural relevance aspects from base entry
            cultural_aspects = ["overall_cultural_relevance", "person_cultural_relevance", 
                              "action_cultural_relevance", "location_cultural_relevance"]
            
            for aspect in cultural_aspects:
                if aspect in base_entry:
                    v_base_texts.append(base_entry[aspect])
                    clean_aspect = aspect.replace("_cultural_relevance", "")
                    v_base_text_labels.append(f"v_base_{clean_aspect}")
            
            # 5: original_prompt from base entry  
            v_base_texts.append(base_entry.get("original_prompt", ""))
            v_base_text_labels.append("v_base_original_prompt")
            
            # 6: final_prompt from mode entry (if exists)
            if "final_prompt" in entry:
                v_base_texts.append(entry["final_prompt"])
                v_base_text_labels.append("v_base_final_prompt")
            else:
                v_base_texts.append("")  # Empty text if no final_prompt
                v_base_text_labels.append("v_base_final_prompt")
            
            # Call Gemini for v_base image with all 6 texts
            if len(v_base_texts) == 6:
                v_base_result = evaluate_image_with_multiple_texts(
                    base_image_path, v_base_texts, "v_base", debug=args.debug
                )
                
                # Map results to individual fields
                for i, label in enumerate(v_base_text_labels, 1):
                    all_results[f"{label}_reasoning"] = v_base_result.get(f"text{i}_reasoning", "")
                    all_results[f"{label}_score"] = v_base_result.get(f"text{i}_score", 1)
                    all_results[f"{label}_text"] = v_base_texts[i-1]
            
            # Prepare text descriptions for v_var image (6 texts)
            v_var_texts = []
            v_var_text_labels = []
            
            # 1-4: Cultural relevance aspects from mode entry
            for aspect in cultural_aspects:
                if aspect in entry:
                    v_var_texts.append(entry[aspect])
                    clean_aspect = aspect.replace("_cultural_relevance", "")
                    v_var_text_labels.append(f"v_var_{clean_aspect}")
            
            # 5: original_prompt from mode entry
            v_var_texts.append(entry.get("original_prompt", ""))
            v_var_text_labels.append("v_var_original_prompt")
            
            # 6: final_prompt from mode entry (if exists)
            if "final_prompt" in entry:
                v_var_texts.append(entry["final_prompt"])
                v_var_text_labels.append("v_var_final_prompt")
            else:
                v_var_texts.append("")  # Empty text if no final_prompt
                v_var_text_labels.append("v_var_final_prompt")
            
            # Call Gemini for v_var image with all 6 texts
            if len(v_var_texts) == 6:
                v_var_result = evaluate_image_with_multiple_texts(
                    mode_image_path, v_var_texts, "v_var", debug=args.debug
                )
                
                # Map results to individual fields
                for i, label in enumerate(v_var_text_labels, 1):
                    all_results[f"{label}_reasoning"] = v_var_result.get(f"text{i}_reasoning", "")
                    all_results[f"{label}_score"] = v_var_result.get(f"text{i}_score", 1)
                    all_results[f"{label}_text"] = v_var_texts[i-1]

            # Write single combined result to log file
            flog.write(json.dumps(all_results) + "\n")
            flog.flush()
            processed_video_ids.add(video_id)

            processed_count += 1
            if args.max_lines is not None:
                print(f"Read {total_lines_read}/{args.max_lines} lines - Processed {processed_count}, Skipped {skipped_count}")

        print(f"Total lines read: {total_lines_read}, Total processed: {processed_count}, Total skipped: {skipped_count}")

if __name__ == "__main__":
    main()
