---
dataset_info:
  features:
  - name: video_id
    dtype: string
  - name: culture
    dtype: string
  - name: person
    dtype: string
  - name: action
    dtype: string
  - name: action_type
    dtype: string
  - name: location
    dtype: string
  - name: is_cross_culture
    dtype: bool
  - name: original_prompt
    dtype: string
  - name: refined_prompt
    dtype: string
  - name: video
    dtype: video
  - name: pipeline
    dtype: string
  splits:
  - name: train
    num_bytes: 676000000
    num_examples: 972
  download_size: 676000000
  dataset_size: 676000000
language:
- en
license: cc0-1.0
pretty_name: Multicultural Multiagent Videos
size_categories:
- 1K<n<10K
task_categories:
- video-classification
---

# Multicultural Multiagent Videos Dataset

A comprehensive dataset of AI-generated videos showcasing multicultural content across different cultures, action types, and locations. The dataset includes videos generated using multiple pipeline approaches with both original and refined prompts.

## Dataset Overview

- **Total Entries**: 972 (243 unique videos × 4 pipelines)
- **Cultures**: Chinese, American, Romanian
- **Mono-culture Videos**: 324 entries (81 unique videos × 4 pipelines)
- **Cross-culture Videos**: 648 entries (162 unique videos × 4 pipelines)
- **Action Types**: food, music, dance (324 entries each)
- **Pipelines**: base, sa (single-agent), mas (multi-agent sequential), map (multi-agent parallel)

## Dataset Structure

Each video entry contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | string | Unique identifier for the video |
| `culture` | string | Primary culture represented in the video (Chinese, American, or Romanian) |
| `person` | string | Description of the person/character in the video |
| `action` | string | Description of the action being performed |
| `action_type` | string | Type of action: `food`, `dance`, or `music` |
| `location` | string | Location where the action takes place |
| `is_cross_culture` | bool | Whether the video represents cross-cultural content (mono=false, cross=true) |
| `original_prompt` | string | Original prompt used to generate the video |
| `refined_prompt` | string | Refined prompt with enhanced cultural details |
| `video_path` | string | Path to the video file relative to the results directory |
| `pipeline` | string | Pipeline used for generation: `base`, `sa`, `mas`, or `map` |

## Pipelines

The dataset includes videos generated using four different pipeline approaches:

1. **base** - Baseline single-agent approach
2. **sa** - Single Agent pipeline
3. **mas** - Multi-Agent Sequential pipeline
4. **map** - Multi-Agent Parallel pipeline

## Action Types

- **food**: Food preparation, eating, and culinary practices
- **dance**: Dancing and traditional dance performances
- **music**: Music playing and musical performances

## Usage Example

### Loading with Hugging Face Datasets

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('guinea-pig/multicultural_multiagent_videos')

# Access an example
example = dataset['train'][0]
print(f"Video ID: {example['video_id']}")
print(f"Culture: {example['culture']}")
print(f"Action Type: {example['action_type']}")
print(f"Original Prompt: {example['original_prompt']}")
print(f"Refined Prompt: {example['refined_prompt']}")
```

### Filtering by Action Type

```python
# Get only food-related videos
food_dataset = dataset.filter(lambda x: x['action_type'] == 'food')

# Get only cross-cultural videos
cross_culture_dataset = dataset.filter(lambda x: x['is_cross_culture'])

# Get videos from a specific culture
chinese_dataset = dataset.filter(lambda x: x['culture'] == 'Chinese')
```

### Filtering by Pipeline

```python
# Get videos from a specific pipeline
base_videos = dataset.filter(lambda x: x['pipeline'] == 'base')
sa_videos = dataset.filter(lambda x: x['pipeline'] == 'sa')
mas_videos = dataset.filter(lambda x: x['pipeline'] == 'mas')
map_videos = dataset.filter(lambda x: x['pipeline'] == 'map')

# Get mono vs cross-culture videos
mono_culture = dataset.filter(lambda x: not x['is_cross_culture'])
cross_culture = dataset.filter(lambda x: x['is_cross_culture'])
```

## Dataset Statistics

- **Action Type Distribution**:
  - Food: 324 entries
  - Music: 324 entries
  - Dance: 324 entries

- **Culture Distribution**:
  - Chinese: 324 entries
  - American: 324 entries
  - Romanian: 324 entries

- **Pipeline Distribution**:
  - Each of the 4 pipelines: 243 entries

## License

This dataset is released under the CC0 license (public domain).

## Notes

- Videos are stored in MP4 format
- Each video is exactly 5 seconds long
- Prompts have been refined to enhance cultural representation and detail
- The dataset supports research into culturally-aware video generation and representation
