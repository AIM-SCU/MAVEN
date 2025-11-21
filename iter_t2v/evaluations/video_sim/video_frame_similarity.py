#!/usr/bin/env python3
"""
Video Frame-to-Frame Similarity Comparison Script
================================================

This script compares frame-level CLIP similarities between base and another mode (single, parallel, sequential).
It computes the similarity between each corresponding frame (1-5) of base and the comparison mode, and outputs 5 frame-level similarities plus an overall average per entry.

Usage:
    CUDA_VISIBLE_DEVICES=0 python evaluations/video_sim/video_frame_similarity.py --mode single
    CUDA_VISIBLE_DEVICES=0 python evaluations/video_sim/video_frame_similarity.py --mode parallel
    CUDA_VISIBLE_DEVICES=0 python evaluations/video_sim/video_frame_similarity.py --mode sequential

Output is written to the corresponding results/base/video_similarity_scores_base_vs_<mode>.jsonl file.
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
BASE_FILE = RESULTS_DIR / "base" / "video_similarity_scores_base.jsonl"

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


def compute_clip_similarity(model, processor, img1, img2, device="cuda"):
    # img1, img2: numpy arrays (RGB)
    image1 = Image.fromarray(img1)
    image2 = Image.fromarray(img2)
    inputs = processor(images=[image1, image2], return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        sim = (image_features[0] @ image_features[1]).item()
    return sim


def extract_frames(video_path, num_frames=5):
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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_count += 1
    cap.release()
    # If not enough frames, pad with last frame
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])
    return frames


def main():
    parser = argparse.ArgumentParser(description="Compare base vs other mode frame-level CLIP similarities.")
    parser.add_argument("--mode", choices=MODES, required=True, help="Comparison mode (single, parallel, sequential)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for CLIP")
    args = parser.parse_args()

    compare_file = RESULTS_DIR / args.mode / f"video_similarity_scores_{args.mode}.jsonl"
    output_file = RESULTS_DIR / args.mode / f"video_similarity_scores_{args.mode}_with_base_sim.jsonl"

    if not BASE_FILE.exists():
        print(f"Error: {BASE_FILE} does not exist.")
        sys.exit(1)
    if not compare_file.exists():
        print(f"Error: {compare_file} does not exist.")
        sys.exit(1)

    print(f"Loading base: {BASE_FILE}")
    print(f"Loading compare: {compare_file}")
    base_data = load_jsonl(BASE_FILE)
    compare_data = load_jsonl(compare_file)

    if len(base_data) != len(compare_data):
        print("Warning: base and compare files have different number of entries. Matching by index.")

    model, processor = get_clip_model(args.device)

    results = []
    for idx, (base_entry, cmp_entry) in enumerate(zip(base_data, compare_data)):
        base_video = base_entry.get("video_path")
        cmp_video = cmp_entry.get("video_path")
        if not base_video or not cmp_video:
            print(f"Skipping entry {idx} due to missing video_path.")
            results.append(cmp_entry)
            continue
        try:
            base_frames = extract_frames(base_video, 5)
            cmp_frames = extract_frames(cmp_video, 5)
        except Exception as e:
            print(f"Skipping entry {idx} due to frame extraction error: {e}")
            results.append(cmp_entry)
            continue
        frame_sims = []
        for i in range(5):
            try:
                img1 = base_frames[i]
                img2 = cmp_frames[i]
                sim = compute_clip_similarity(model, processor, img1, img2, device=args.device)
            except Exception as e:
                print(f"Error comparing frame {i} for entry {idx}: {e}")
                sim = 0.0
            frame_sims.append(sim)
        avg_sim = float(np.mean(frame_sims))
        # Append the 6 new numbers to the cmp_entry
        cmp_entry = dict(cmp_entry)  # copy to avoid mutating input
        for i, sim in enumerate(frame_sims):
            cmp_entry[f"base_vs_{args.mode}_frame_{i}"] = sim
        cmp_entry[f"base_vs_{args.mode}_avg"] = avg_sim
        results.append(cmp_entry)
    save_jsonl(results, output_file)
    print(f"Saved {len(results)} entries to {output_file}")

if __name__ == "__main__":
    main()
