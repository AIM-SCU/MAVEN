#!/usr/bin/env python3
"""
VLM Cultural Relevance Evaluation Script

This script evaluates the cultural relevance of video frames using Gemini Vision Language Model.
It takes a mode parameter and processes cultural relevance data to generate VLM scores.

Usage:
    python evaluations/vlm/vlm_eval_culture_relevance.py --mode single
    python evaluations/vlm/vlm_eval_culture_relevance.py --mode single --max_lines 2
    python evaluations/vlm/vlm_eval_culture_relevance.py --mode base --max_lines 1 --debug

    python evaluations/vlm/vlm_eval_culture_relevance.py --mode single --max_lines 5
    python evaluations/vlm/vlm_eval_culture_relevance.py --mode sequential
    python evaluations/vlm/vlm_eval_culture_relevance.py --mode parallel

Arguments:
    --mode: Processing mode (base, single, sequential, parallel)
    --max_lines: Maximum number of total lines to read from input file, including both skipped and processed entries (default: process all lines)
    --debug: Print Gemini responses for debugging (default: False)

For 'base' mode: Checks for video_culture_relevance_base.jsonl
For 'single' mode: Checks for video_text_alignment_single.jsonl  
For 'sequential' mode: Checks for video_text_alignment_sequential.jsonl
For 'parallel' mode: Checks for video_text_alignment_parallel.jsonl

The script will:
1. Read the input JSONL file based on the mode
2. For each entry, extract the video_id and locate the corresponding frame image
3. Build a prompt with cultural relevance descriptions
4. Call Gemini API to evaluate cultural alignment
5. Output results with additional VLM score fields
6. Log detailed reasoning and scores separately
"""

import argparse
import os
import json
from google import genai
from google.genai import types

RESULTS_DIR = "/workspace/t2v_self/iter_t2v/results"

MODE_TO_FILE = {
    "base": os.path.join(RESULTS_DIR, "base/video_culture_relevance_base.jsonl"),
    "single": os.path.join(RESULTS_DIR, "single/video_text_alignment_single.jsonl"),
    "sequential": os.path.join(RESULTS_DIR, "sequential/video_text_alignment_sequential.jsonl"),
    "parallel": os.path.join(RESULTS_DIR, "parallel/video_text_alignment_parallel.jsonl"),
}

def get_log_file_path(mode):
    return f"/workspace/t2v_self/iter_t2v/results/{mode}/vlm_eval_culture_relevance_log.jsonl"

PROMPT_TEMPLATE = '''You will be given one image and four culturally annotated sentences describing different aspects of it (overall scene, person, action, and location).

Your task is to evaluate how culturally aligned the image is with respect to each sentence — that is, whether the visual features in the image reflect the cultural cues or identities expressed in the text.

Specifically, consider appearance, clothing, architecture, symbols, traditions, or any other visual cues linked to a specific country or culture.

For each sentence, reason step by step and assign a score between 1 and 5, where:

1 = The image is not culturally relevant to the description
5 = The image is highly culturally relevant to the description
'''

def check_required_file(mode):
    file_path = MODE_TO_FILE.get(mode)
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file for mode '{mode}' not found: {file_path}")
    return file_path

def get_frame_path(video_id, mode):
    frames_dir = os.path.join(RESULTS_DIR, mode, "frames")
    return os.path.join(frames_dir, video_id, "frame_0002.jpg")

def build_prompt(image_path, entry):
    prompt = f"{PROMPT_TEMPLATE}\nimage: {image_path}\n" \
             f"text_overall: \"{entry['overall_cultural_relevance']}\"\n" \
             f"text_person: \"{entry['person_cultural_relevance']}\"\n" \
             f"text_action: \"{entry['action_cultural_relevance']}\"\n" \
             f"text_location: \"{entry['location_cultural_relevance']}\"\n"
    
    # Append the output format instruction
    prompt += '''
The output should be a JSON object ONLY with the following format:
{
  "overall_reasoning": "...",
  "overall_score": number,
  "person_reasoning": "...",
  "person_score": number,
  "action_reasoning": "...",
  "action_score": number,
  "location_reasoning": "...",
  "location_score": number
}'''
    
    return prompt

def call_gemini_vlm(image_path, prompt, debug=False):
    """Call Gemini API for VLM evaluation with retry logic"""
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
                "overall_reasoning": f"Failed to parse response after {max_retries} attempts",
                "overall_score": 1,
                "person_reasoning": f"Failed to parse response after {max_retries} attempts",
                "person_score": 1,
                "action_reasoning": f"Failed to parse response after {max_retries} attempts",
                "action_score": 1,
                "location_reasoning": f"Failed to parse response after {max_retries} attempts",
                "location_score": 1
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
    parser = argparse.ArgumentParser(description="VLM Cultural Relevance Evaluation Script")
    parser.add_argument("--mode", type=str, required=True, choices=["base", "single", "sequential", "parallel"],
                        help="Processing mode (base, single, sequential, parallel)")
    parser.add_argument("--max_lines", type=int, default=None,
                        help="Maximum number of total lines to read from input file, including both skipped and processed entries (default: process all lines)")
    parser.add_argument("--debug", action="store_true",
                        help="Print Gemini responses for debugging (default: False)")
    args = parser.parse_args()

    file_path = check_required_file(args.mode)
    log_file = get_log_file_path(args.mode)
    
    print(f"Processing file: {file_path}")
    print(f"Log file: {log_file}")

    # Read processed video_ids once
    processed_video_ids = get_processed_video_ids(log_file)
    print(f"Found {len(processed_video_ids)} already processed entries in log file.")

    with open(file_path, "r") as fin, open(log_file, "a") as flog:
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

            image_path = get_frame_path(video_id, args.mode)
            if not os.path.exists(image_path):
                print(f"Warning: Frame not found: {image_path}, skipping...")
                continue

            prompt = build_prompt(image_path, entry)
            vlm_result = call_gemini_vlm(image_path, prompt, debug=args.debug)

            vlm_result["llm_prompt"] = prompt
            vlm_result["video_id"] = video_id
            vlm_result["original_prompt"] = entry.get("original_prompt", "")

            flog.write(json.dumps(vlm_result) + "\n")
            flog.flush()
            processed_video_ids.add(video_id)

            processed_count += 1
            if args.max_lines is not None:
                print(f"Read {total_lines_read}/{args.max_lines} lines - Processed {processed_count}, Skipped {skipped_count}")

        print(f"Total lines read: {total_lines_read}, Total processed: {processed_count}, Total skipped: {skipped_count}")

if __name__ == "__main__":
    main()
