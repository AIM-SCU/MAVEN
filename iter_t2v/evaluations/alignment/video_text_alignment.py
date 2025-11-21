#!/usr/bin/env python3
"""
Video-Text Alignment Computation Script
======================================

This script computes CLIP similarity scores between videos and texts from different modes.
It computes 12 different alignment scores:

1-4:  V_base → [overall, person, action, location]_cultural_relevance
5-8:  V_var → [overall, person, action, location]_cultural_relevance  
9:    V_base → original_prompt (base only)
10:   V_base → final_prompt (variant only)
11:   V_var → original_prompt (base only)
12:   V_var → final_prompt (variant only)

Usage:
    CUDA_VISIBLE_DEVICES=0 python evaluations/alignment/video_text_alignment.py --mode single
    CUDA_VISIBLE_DEVICES=0 python evaluations/alignment/video_text_alignment.py --mode parallel
    CUDA_VISIBLE_DEVICES=0 python evaluations/alignment/video_text_alignment.py --mode sequential

Output is written to the corresponding results/<mode>/video_text_alignment_<mode>.jsonl file.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
MODES = ["single", "parallel", "sequential"]


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_clip_model(device="cuda"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def split_text_for_clip(text, processor, max_tokens=77):
    """Split text into head and tail for CLIP processing."""
    # Tokenize the text with special tokens to match what the processor will do
    tokens = processor.tokenizer.encode(text, add_special_tokens=True)
    
    # Check if text is too long for dual encoding (accounting for special tokens)
    if len(tokens) > 2 * max_tokens:
        raise ValueError(f"Text is too long ({len(tokens)} tokens) for dual encoding. Maximum supported is {2 * max_tokens} tokens.")
    
    # Account for special tokens (usually 2: start and end tokens)
    # Leave room for special tokens in each segment - using 4 for safety
    effective_max_tokens = max_tokens - 4
    
    if len(tokens) <= max_tokens:
        # Text is short enough, use as-is
        return text, None
    else:
        # Text is too long, split into head and tail
        # Re-tokenize without special tokens for splitting
        tokens_no_special = processor.tokenizer.encode(text, add_special_tokens=False)
        
        head_tokens = tokens_no_special[:effective_max_tokens]
        tail_tokens = tokens_no_special[-effective_max_tokens:]
        
        head_text = processor.tokenizer.decode(head_tokens, skip_special_tokens=True)
        tail_text = processor.tokenizer.decode(tail_tokens, skip_special_tokens=True)
        
        return head_text, tail_text


def extract_frames(video_path, num_frames=5):
    """Extract frames from video for processing."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    
    frames = []
    frame_count = 0
    
    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
        frame_count += 1
    
    cap.release()
    
    # If not enough frames, pad with last frame
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])
    
    return frames


def compute_video_text_similarity(model, processor, video_path, text, device="cuda", num_frames=5):
    """Compute CLIP similarity between video frames and text."""
    try:
        frames = extract_frames(video_path, num_frames)
        if not frames:
            return 0.0
        
        frame_similarities = []
        for frame in frames:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame)
            
            # Process inputs
            inputs = processor(
                text=[text], 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
                # Get normalized features
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                # Normalize features
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
                
                # Compute cosine similarity
                similarity = (image_features @ text_features.T).item()
                frame_similarities.append(similarity)
        
        # Return average similarity across frames
        return float(np.mean(frame_similarities))
    
    except Exception as e:
        print(f"Error computing similarity for {video_path} with text '{text[:50]}...': {e}")
        return 0.0


def compute_video_text_similarity_dual(model, processor, video_path, text, device="cuda", num_frames=5):
    """Compute CLIP similarity between video frames and text, handling long texts with dual encoding."""
    try:
        frames = extract_frames(video_path, num_frames)
        if not frames:
            empty_frames = [0.0] * num_frames
            return {
                "head_frames": empty_frames, "head_avg": 0.0,
                "tail_frames": None, "tail_avg": None,
                "avg_frames": empty_frames, "avg_avg": 0.0
            }
        
        # Split text into head and tail
        head_text, tail_text = split_text_for_clip(text, processor)
        
        # Compute similarity for head
        head_similarities = []
        for frame in frames:
            image = Image.fromarray(frame)
            inputs = processor(
                text=[head_text], 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
                similarity = (image_features @ text_features.T).item()
                head_similarities.append(similarity)
        
        head_score = float(np.mean(head_similarities))
        
        if tail_text is None:
            # Short text, no tail
            return {
                "head_frames": head_similarities, "head_avg": head_score,
                "tail_frames": None, "tail_avg": None,
                "avg_frames": head_similarities, "avg_avg": head_score
            }
        else:
            # Long text, compute tail similarity
            tail_similarities = []
            for frame in frames:
                image = Image.fromarray(frame)
                inputs = processor(
                    text=[tail_text], 
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds
                    image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
                    similarity = (image_features @ text_features.T).item()
                    tail_similarities.append(similarity)
            
            tail_score = float(np.mean(tail_similarities))
            
            # Compute average of head and tail for each frame
            avg_similarities = [(h + t) / 2.0 for h, t in zip(head_similarities, tail_similarities)]
            avg_score = float(np.mean(avg_similarities))
            
            return {
                "head_frames": head_similarities, "head_avg": head_score,
                "tail_frames": tail_similarities, "tail_avg": tail_score,
                "avg_frames": avg_similarities, "avg_avg": avg_score
            }
    
    except Exception as e:
        print(f"Error computing similarity for {video_path} with text '{text[:50]}...': {e}")
        empty_frames = [0.0] * num_frames
        return {
            "head_frames": empty_frames, "head_avg": 0.0,
            "tail_frames": None, "tail_avg": None,
            "avg_frames": empty_frames, "avg_avg": 0.0
        }


def main():
    parser = argparse.ArgumentParser(description="Compute video-text alignment scores between base and variant modes.")
    parser.add_argument("--mode", choices=MODES, required=True, help="Variant mode (single, parallel, sequential)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for CLIP")
    args = parser.parse_args()

    base_file = RESULTS_DIR / "base" / "video_similarity_scores_base.jsonl"
    variant_file = RESULTS_DIR / args.mode / f"video_similarity_scores_{args.mode}_with_base_sim.jsonl"
    output_file = RESULTS_DIR / args.mode / f"video_text_alignment_{args.mode}.jsonl"

    if not base_file.exists():
        print(f"Error: {base_file} does not exist.")
        sys.exit(1)
    if not variant_file.exists():
        print(f"Error: {variant_file} does not exist.")
        print(f"Please run the video frame similarity script first to generate this file.")
        sys.exit(1)

    print(f"Loading base: {base_file}")
    print(f"Loading variant: {variant_file}")
    base_data = load_jsonl(base_file)
    variant_data = load_jsonl(variant_file)

    if len(base_data) != len(variant_data):
        print("Warning: base and variant files have different number of entries. Matching by index.")

    print("Loading CLIP model...")
    model, processor = get_clip_model(args.device)

    results = []
    for idx, (base_entry, variant_entry) in enumerate(zip(base_data, variant_data)):
        if idx % 10 == 0:
            print(f"Processing entry {idx+1}/{len(base_data)}")
        
        base_video = base_entry.get("video_path")
        variant_video = variant_entry.get("video_path")
        
        if not base_video or not variant_video:
            print(f"Skipping entry {idx} due to missing video_path.")
            continue
        
        # Get texts
        cultural_texts = {
            "overall": base_entry.get("overall_cultural_relevance", ""),
            "person": base_entry.get("person_cultural_relevance", ""),
            "action": base_entry.get("action_cultural_relevance", ""),
            "location": base_entry.get("location_cultural_relevance", "")
        }
        
        original_prompt = base_entry.get("original_prompt", "")
        final_prompt = variant_entry.get("final_prompt", "")
        
        # Compute alignments with dual encoding
        alignment_scores = {}
        
        # 1-4: V_base → cultural_relevance texts
        for cultural_type, text in cultural_texts.items():
            if text:
                scores = compute_video_text_similarity_dual(
                    model, processor, base_video, text, args.device
                )
                # Store frame-level scores
                for i in range(5):
                    alignment_scores[f"v_base_to_{cultural_type}_cultural_relevance_head_frame_{i}"] = scores["head_frames"][i]
                    if scores["tail_frames"] is not None:
                        alignment_scores[f"v_base_to_{cultural_type}_cultural_relevance_tail_frame_{i}"] = scores["tail_frames"][i]
                    else:
                        alignment_scores[f"v_base_to_{cultural_type}_cultural_relevance_tail_frame_{i}"] = None
                    alignment_scores[f"v_base_to_{cultural_type}_cultural_relevance_avg_frame_{i}"] = scores["avg_frames"][i]
                
                # Store average scores
                alignment_scores[f"v_base_to_{cultural_type}_cultural_relevance_head_avg"] = scores["head_avg"]
                alignment_scores[f"v_base_to_{cultural_type}_cultural_relevance_tail_avg"] = scores["tail_avg"]
                alignment_scores[f"v_base_to_{cultural_type}_cultural_relevance_avg_avg"] = scores["avg_avg"]
            else:
                # Store empty scores
                for i in range(5):
                    alignment_scores[f"v_base_to_{cultural_type}_cultural_relevance_head_frame_{i}"] = 0.0
                    alignment_scores[f"v_base_to_{cultural_type}_cultural_relevance_tail_frame_{i}"] = None
                    alignment_scores[f"v_base_to_{cultural_type}_cultural_relevance_avg_frame_{i}"] = 0.0
                alignment_scores[f"v_base_to_{cultural_type}_cultural_relevance_head_avg"] = 0.0
                alignment_scores[f"v_base_to_{cultural_type}_cultural_relevance_tail_avg"] = None
                alignment_scores[f"v_base_to_{cultural_type}_cultural_relevance_avg_avg"] = 0.0
        
        # 5-8: V_var → cultural_relevance texts  
        for cultural_type, text in cultural_texts.items():
            if text:
                scores = compute_video_text_similarity_dual(
                    model, processor, variant_video, text, args.device
                )
                # Store frame-level scores
                for i in range(5):
                    alignment_scores[f"v_var_to_{cultural_type}_cultural_relevance_head_frame_{i}"] = scores["head_frames"][i]
                    if scores["tail_frames"] is not None:
                        alignment_scores[f"v_var_to_{cultural_type}_cultural_relevance_tail_frame_{i}"] = scores["tail_frames"][i]
                    else:
                        alignment_scores[f"v_var_to_{cultural_type}_cultural_relevance_tail_frame_{i}"] = None
                    alignment_scores[f"v_var_to_{cultural_type}_cultural_relevance_avg_frame_{i}"] = scores["avg_frames"][i]
                
                # Store average scores
                alignment_scores[f"v_var_to_{cultural_type}_cultural_relevance_head_avg"] = scores["head_avg"]
                alignment_scores[f"v_var_to_{cultural_type}_cultural_relevance_tail_avg"] = scores["tail_avg"]
                alignment_scores[f"v_var_to_{cultural_type}_cultural_relevance_avg_avg"] = scores["avg_avg"]
            else:
                # Store empty scores
                for i in range(5):
                    alignment_scores[f"v_var_to_{cultural_type}_cultural_relevance_head_frame_{i}"] = 0.0
                    alignment_scores[f"v_var_to_{cultural_type}_cultural_relevance_tail_frame_{i}"] = None
                    alignment_scores[f"v_var_to_{cultural_type}_cultural_relevance_avg_frame_{i}"] = 0.0
                alignment_scores[f"v_var_to_{cultural_type}_cultural_relevance_head_avg"] = 0.0
                alignment_scores[f"v_var_to_{cultural_type}_cultural_relevance_tail_avg"] = None
                alignment_scores[f"v_var_to_{cultural_type}_cultural_relevance_avg_avg"] = 0.0
        
        # 9: V_base → original_prompt
        if original_prompt:
            scores = compute_video_text_similarity_dual(
                model, processor, base_video, original_prompt, args.device
            )
            # Store frame-level scores
            for i in range(5):
                alignment_scores[f"v_base_to_original_prompt_head_frame_{i}"] = scores["head_frames"][i]
                if scores["tail_frames"] is not None:
                    alignment_scores[f"v_base_to_original_prompt_tail_frame_{i}"] = scores["tail_frames"][i]
                else:
                    alignment_scores[f"v_base_to_original_prompt_tail_frame_{i}"] = None
                alignment_scores[f"v_base_to_original_prompt_avg_frame_{i}"] = scores["avg_frames"][i]
            
            # Store average scores
            alignment_scores["v_base_to_original_prompt_head_avg"] = scores["head_avg"]
            alignment_scores["v_base_to_original_prompt_tail_avg"] = scores["tail_avg"]
            alignment_scores["v_base_to_original_prompt_avg_avg"] = scores["avg_avg"]
        else:
            # Store empty scores
            for i in range(5):
                alignment_scores[f"v_base_to_original_prompt_head_frame_{i}"] = 0.0
                alignment_scores[f"v_base_to_original_prompt_tail_frame_{i}"] = None
                alignment_scores[f"v_base_to_original_prompt_avg_frame_{i}"] = 0.0
            alignment_scores["v_base_to_original_prompt_head_avg"] = 0.0
            alignment_scores["v_base_to_original_prompt_tail_avg"] = None
            alignment_scores["v_base_to_original_prompt_avg_avg"] = 0.0
        
        # 10: V_base → final_prompt
        if final_prompt:
            scores = compute_video_text_similarity_dual(
                model, processor, base_video, final_prompt, args.device
            )
            # Store frame-level scores
            for i in range(5):
                alignment_scores[f"v_base_to_final_prompt_head_frame_{i}"] = scores["head_frames"][i]
                if scores["tail_frames"] is not None:
                    alignment_scores[f"v_base_to_final_prompt_tail_frame_{i}"] = scores["tail_frames"][i]
                else:
                    alignment_scores[f"v_base_to_final_prompt_tail_frame_{i}"] = None
                alignment_scores[f"v_base_to_final_prompt_avg_frame_{i}"] = scores["avg_frames"][i]
            
            # Store average scores
            alignment_scores["v_base_to_final_prompt_head_avg"] = scores["head_avg"]
            alignment_scores["v_base_to_final_prompt_tail_avg"] = scores["tail_avg"]
            alignment_scores["v_base_to_final_prompt_avg_avg"] = scores["avg_avg"]
        else:
            # Store empty scores
            for i in range(5):
                alignment_scores[f"v_base_to_final_prompt_head_frame_{i}"] = 0.0
                alignment_scores[f"v_base_to_final_prompt_tail_frame_{i}"] = None
                alignment_scores[f"v_base_to_final_prompt_avg_frame_{i}"] = 0.0
            alignment_scores["v_base_to_final_prompt_head_avg"] = 0.0
            alignment_scores["v_base_to_final_prompt_tail_avg"] = None
            alignment_scores["v_base_to_final_prompt_avg_avg"] = 0.0
        
        # 11: V_var → original_prompt
        if original_prompt:
            scores = compute_video_text_similarity_dual(
                model, processor, variant_video, original_prompt, args.device
            )
            # Store frame-level scores
            for i in range(5):
                alignment_scores[f"v_var_to_original_prompt_head_frame_{i}"] = scores["head_frames"][i]
                if scores["tail_frames"] is not None:
                    alignment_scores[f"v_var_to_original_prompt_tail_frame_{i}"] = scores["tail_frames"][i]
                else:
                    alignment_scores[f"v_var_to_original_prompt_tail_frame_{i}"] = None
                alignment_scores[f"v_var_to_original_prompt_avg_frame_{i}"] = scores["avg_frames"][i]
            
            # Store average scores
            alignment_scores["v_var_to_original_prompt_head_avg"] = scores["head_avg"]
            alignment_scores["v_var_to_original_prompt_tail_avg"] = scores["tail_avg"]
            alignment_scores["v_var_to_original_prompt_avg_avg"] = scores["avg_avg"]
        else:
            # Store empty scores
            for i in range(5):
                alignment_scores[f"v_var_to_original_prompt_head_frame_{i}"] = 0.0
                alignment_scores[f"v_var_to_original_prompt_tail_frame_{i}"] = None
                alignment_scores[f"v_var_to_original_prompt_avg_frame_{i}"] = 0.0
            alignment_scores["v_var_to_original_prompt_head_avg"] = 0.0
            alignment_scores["v_var_to_original_prompt_tail_avg"] = None
            alignment_scores["v_var_to_original_prompt_avg_avg"] = 0.0
        
        # 12: V_var → final_prompt
        if final_prompt:
            scores = compute_video_text_similarity_dual(
                model, processor, variant_video, final_prompt, args.device
            )
            # Store frame-level scores
            for i in range(5):
                alignment_scores[f"v_var_to_final_prompt_head_frame_{i}"] = scores["head_frames"][i]
                if scores["tail_frames"] is not None:
                    alignment_scores[f"v_var_to_final_prompt_tail_frame_{i}"] = scores["tail_frames"][i]
                else:
                    alignment_scores[f"v_var_to_final_prompt_tail_frame_{i}"] = None
                alignment_scores[f"v_var_to_final_prompt_avg_frame_{i}"] = scores["avg_frames"][i]
            
            # Store average scores
            alignment_scores["v_var_to_final_prompt_head_avg"] = scores["head_avg"]
            alignment_scores["v_var_to_final_prompt_tail_avg"] = scores["tail_avg"]
            alignment_scores["v_var_to_final_prompt_avg_avg"] = scores["avg_avg"]
        else:
            # Store empty scores
            for i in range(5):
                alignment_scores[f"v_var_to_final_prompt_head_frame_{i}"] = 0.0
                alignment_scores[f"v_var_to_final_prompt_tail_frame_{i}"] = None
                alignment_scores[f"v_var_to_final_prompt_avg_frame_{i}"] = 0.0
            alignment_scores["v_var_to_final_prompt_head_avg"] = 0.0
            alignment_scores["v_var_to_final_prompt_tail_avg"] = None
            alignment_scores["v_var_to_final_prompt_avg_avg"] = 0.0
        
        # Create result entry with original variant data plus alignment scores
        result = variant_entry.copy()
        result.update(alignment_scores)
        results.append(result)

    save_jsonl(results, output_file)
    print(f"Saved {len(results)} entries to {output_file}")


if __name__ == "__main__":
    main()
