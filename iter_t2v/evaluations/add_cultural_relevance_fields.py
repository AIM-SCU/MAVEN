#!/usr/bin/env python3
"""
Cultural Relevance Text Fields Generator
========================================

This script adds 4 cultural relevance text fields to a JSONL file based on the
person_culture, action_culture, and location_culture dimensions.

Usage:
    python evaluations/add_cultural_relevance_fields.py --input /workspace/t2v_self/iter_t2v/results/base/prompts_base_merged.jsonl --output /workspace/t2v_self/iter_t2v/evaluations/prompts_base_cultural_relevance.jsonl
"""

import json
import argparse
from typing import Dict, List
from pathlib import Path


def get_culture_name_mapping() -> Dict[str, str]:
    """Map culture codes to proper country names."""
    return {
        "Chinese": "China",
        "American": "United States", 
        "Romanian": "Romania"
    }


def get_culture_adjective_mapping() -> Dict[str, str]:
    """Map culture codes to proper adjectives for cultural practices."""
    return {
        "Chinese": "Chinese",
        "American": "American", 
        "Romanian": "Romanian"
    }


def get_culture_article_mapping() -> Dict[str, str]:
    """Map culture codes to proper articles (a/an) for grammar."""
    return {
        "Chinese": "a",
        "American": "an", 
        "Romanian": "a"
    }


def generate_overall_cultural_relevance(person_culture: str, action_culture: str, location_culture: str) -> str:
    """
    Generate overall cultural relevance text based on the three cultural dimensions.
    
    Args:
        person_culture: Culture of the person
        action_culture: Culture of the action
        location_culture: Culture of the location
        
    Returns:
        Text describing overall cultural relevance
    """
    culture_mapping = get_culture_name_mapping()
    
    # Convert to country names
    person_country = culture_mapping.get(person_culture, person_culture)
    action_country = culture_mapping.get(action_culture, action_culture)
    location_country = culture_mapping.get(location_culture, location_culture)
    
    # Get unique cultures
    cultures = [person_culture, action_culture, location_culture]
    unique_cultures = list(set(cultures))
    
    if len(unique_cultures) == 1:
        # All same culture
        return f"This image belongs to {person_country}."
    elif len(unique_cultures) == 3:
        # All different cultures
        countries = [culture_mapping.get(c, c) for c in unique_cultures]
        countries.sort()  # For consistent ordering
        return f"This image represents a cultural fusion of {', '.join(countries[:-1])}, and {countries[-1]}."
    else:
        # Two dimensions same, one different
        culture_counts = {c: cultures.count(c) for c in unique_cultures}
        primary_culture = max(culture_counts.keys(), key=lambda x: culture_counts[x])
        secondary_culture = min(culture_counts.keys(), key=lambda x: culture_counts[x])
        
        primary_country = culture_mapping.get(primary_culture, primary_culture)
        secondary_country = culture_mapping.get(secondary_culture, secondary_culture)
        
        return f"This image primarily belongs to {primary_country} with {secondary_country} cultural elements."


def generate_person_cultural_relevance(person_culture: str, action_culture: str, location_culture: str, person_segment: str, location_segment: str) -> str:
    """Generate person-specific cultural relevance text."""
    # Fix grammar by replacing "a" with proper article if needed
    culture_article_mapping = get_culture_article_mapping()
    correct_article = culture_article_mapping.get(person_culture, "a")
    
    # Replace the article in person_segment if it starts with "a "
    if person_segment.startswith("a "):
        corrected_segment = person_segment.replace("a ", f"{correct_article} ", 1)
    else:
        corrected_segment = person_segment
    
    return f"This image shows {corrected_segment}."


def generate_action_cultural_relevance(person_culture: str, action_culture: str, location_culture: str, action_segment: str, location_segment: str) -> str:
    """Generate action-specific cultural relevance text."""
    culture_adjective_mapping = get_culture_adjective_mapping()
    
    action_adjective = culture_adjective_mapping.get(action_culture, action_culture)
    
    return f"This image depicts {action_segment}, a practice associated with {action_adjective} culture."


def generate_location_cultural_relevance(person_culture: str, action_culture: str, location_culture: str, person_segment: str, location_segment: str) -> str:
    """Generate location-specific cultural relevance text."""
    culture_mapping = get_culture_name_mapping()
    
    location_country = culture_mapping.get(location_culture, location_culture)
    
    return f"This image shows {location_segment} in {location_country}."


def process_jsonl_entry(entry: Dict) -> Dict:
    """
    Process a single JSONL entry and add cultural relevance fields.
    
    Args:
        entry: Dictionary representing one line from JSONL
        
    Returns:
        Dictionary with added cultural relevance fields
    """
    # Extract existing cultural dimensions
    person_culture = entry.get("person_culture", "")
    action_culture = entry.get("action_culture", "")  
    location_culture = entry.get("location_culture", "")
    
    # Extract segment information
    person_segment = entry.get("person_segment", "")
    action_segment = entry.get("action_segment", "")
    location_segment = entry.get("location_segment", "")
    
    if not all([person_culture, action_culture, location_culture]):
        print(f"Warning: Missing cultural dimensions in entry: {entry.get('prompt', 'Unknown')}")
        # Add empty fields if dimensions are missing
        entry["overall_cultural_relevance"] = ""
        entry["person_cultural_relevance"] = ""
        entry["action_cultural_relevance"] = ""
        entry["location_cultural_relevance"] = ""
        return entry
    
    # Generate cultural relevance texts
    entry["overall_cultural_relevance"] = generate_overall_cultural_relevance(
        person_culture, action_culture, location_culture
    )
    entry["person_cultural_relevance"] = generate_person_cultural_relevance(
        person_culture, action_culture, location_culture, person_segment, location_segment
    )
    entry["action_cultural_relevance"] = generate_action_cultural_relevance(
        person_culture, action_culture, location_culture, action_segment, location_segment
    )
    entry["location_cultural_relevance"] = generate_location_cultural_relevance(
        person_culture, action_culture, location_culture, person_segment, location_segment
    )
    
    return entry


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


def print_statistics(data: List[Dict]):
    """Print statistics about the cultural dimensions."""
    alignment_counts = {}
    culture_combinations = {}
    
    for entry in data:
        alignment = entry.get("alignment", "unknown")
        alignment_counts[alignment] = alignment_counts.get(alignment, 0) + 1
        
        person_culture = entry.get("person_culture", "")
        action_culture = entry.get("action_culture", "")
        location_culture = entry.get("location_culture", "")
        
        if all([person_culture, action_culture, location_culture]):
            combo = f"{person_culture}-{action_culture}-{location_culture}"
            culture_combinations[combo] = culture_combinations.get(combo, 0) + 1
    
    print("\n" + "="*50)
    print("CULTURAL RELEVANCE STATISTICS")
    print("="*50)
    print("Alignment Distribution:")
    for alignment, count in alignment_counts.items():
        print(f"  {alignment}: {count}")
    
    print("\nTop Cultural Combinations:")
    sorted_combos = sorted(culture_combinations.items(), key=lambda x: x[1], reverse=True)
    for combo, count in sorted_combos[:10]:
        print(f"  {combo}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Add cultural relevance text fields to JSONL file"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file with cultural relevance fields")
    
    args = parser.parse_args()
    
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
    
    print(f"Processing {len(data)} entries...")
    
    # Process each entry
    processed_data = []
    for entry in data:
        processed_entry = process_jsonl_entry(entry)
        processed_data.append(processed_entry)
    
    # Save processed data
    save_jsonl(processed_data, args.output)
    
    # Print statistics
    print_statistics(processed_data)
    
    # Show examples
    print("\n" + "="*50)
    print("EXAMPLE CULTURAL RELEVANCE TEXTS")
    print("="*50)
    
    # Show examples for different alignment types
    alignment_examples = {}
    for entry in processed_data:
        alignment = entry.get("alignment", "unknown")
        if alignment not in alignment_examples and len(alignment_examples) < 3:
            alignment_examples[alignment] = entry
    
    for alignment, entry in alignment_examples.items():
        print(f"\n{alignment.upper()} CULTURE EXAMPLE:")
        print(f"Prompt: {entry.get('prompt', 'N/A')}")
        print(f"Overall: {entry.get('overall_cultural_relevance', 'N/A')}")
        print(f"Person: {entry.get('person_cultural_relevance', 'N/A')}")
        print(f"Action: {entry.get('action_cultural_relevance', 'N/A')}")
        print(f"Location: {entry.get('location_cultural_relevance', 'N/A')}")


if __name__ == "__main__":
    main()
