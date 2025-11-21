#!/usr/bin/env python3
"""
Create metadata.jsonl from dataset.jsonl with proper column ordering for HF VideoFolder format.
Columns are ordered with file_name (video) first.
"""
import json
from pathlib import Path

# Pipeline names mapping
PIPELINES = {
    'base': 'base',
    'single': 'sa',
    'sequential': 'mas',
    'parallel': 'map'
}

RESULTS_DIR = Path('/Users/codebear/ms_thesis/t2v_secret/iter_t2v/results')
INPUT_FILE = Path(__file__).parent / 'dataset.jsonl'
OUTPUT_FILE = Path(__file__).parent / 'metadata.jsonl'

def load_metadata(pipeline_name):
    """Load metadata from video_culture_relevance file for a pipeline."""
    metadata_file = RESULTS_DIR / pipeline_name / f'video_culture_relevance_{pipeline_name}.jsonl'

    if not metadata_file.exists():
        print(f"Warning: {metadata_file} not found")
        return {}

    metadata = {}
    with open(metadata_file, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                video_id = entry.get('video_id')
                if video_id:
                    metadata[video_id] = entry

    return metadata

def main():
    print("Loading metadata from all pipelines...")

    # Create dataset entries - one per pipeline per video
    dataset_entries = []

    for pipeline_dir, pipeline_name in PIPELINES.items():
        print(f"  Loading {pipeline_dir}...")
        metadata = load_metadata(pipeline_dir)

        for video_id, entry in metadata.items():
            # Extract fields
            culture = entry.get('person_culture', 'unknown')
            action_category = entry.get('action_category', 'other')
            action_segment = entry.get('action_segment', 'unknown')
            person_segment = entry.get('person_segment', 'unknown')
            location_segment = entry.get('location_segment', 'unknown')
            alignment = entry.get('alignment', 'unknown')
            original_prompt = entry.get('original_prompt', '')
            refined_prompt = entry.get('final_prompt', '')

            # Determine if mono or cross culture
            is_cross_culture = alignment != 'same_culture' if alignment else False

            # Create dataset entry with file_name first (HF VideoFolder convention)
            dataset_entry = {
                'file_name': f'{pipeline_dir}/videos/{video_id}.mp4',
                'video_id': video_id,
                'culture': culture,
                'person': person_segment,
                'action': action_segment,
                'action_type': action_category,
                'location': location_segment,
                'is_cross_culture': is_cross_culture,
                'original_prompt': original_prompt,
                'refined_prompt': refined_prompt,
                'pipeline': pipeline_name,
            }

            dataset_entries.append(dataset_entry)

    print(f"Created {len(dataset_entries)} entries from {len(dataset_entries) // 4} unique videos x 4 pipelines")

    # Write dataset
    print(f"Writing {len(dataset_entries)} entries to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + '\n')

    # Print sample entries from each pipeline
    if dataset_entries:
        print("\nSample entries (one from each pipeline):")
        for pipeline_name in ['base', 'sa', 'mas', 'map']:
            sample = next((e for e in dataset_entries if e['pipeline'] == pipeline_name), None)
            if sample:
                print(f"\n{pipeline_name}:")
                print(f"  file_name: {sample['file_name']}")
                print(f"  culture: {sample['culture']}")
                print(f"  person: {sample['person']}")
                print(f"  action: {sample['action']}")
                print(f"  action_type: {sample['action_type']}")

    print(f"\n✅ metadata.jsonl created successfully: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
