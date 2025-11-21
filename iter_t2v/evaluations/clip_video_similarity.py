#!/usr/bin/env python3
"""
CLIP Video-Text Similarity Calculator
=====================================

This script computes CLIP similarity scores between video frames and text prompts.
It extracts frames from a video and calculates the average similarity score.

Usage:
    python clip_video_similarity.py --video path/to/video.mp4 --text "A man playing guitar"
    python clip_video_similarity.py --video path/to/video.mp4 --text_file prompts.txt
"""

import os
import cv2
import json
import torch
import argparse
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F


class CLIPVideoSimilarity:
    def __init__(
        self,
        model_path: str = "openai/clip-vit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize CLIP model for video-text similarity computation.
        
        Args:
            model_path: Path or name of the CLIP model
            device: Device to run the model on (cuda/cpu)
        """
        self.device = device
        self.model = CLIPModel.from_pretrained(model_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        print(f"CLIP model loaded on {self.device}")

    ### this should be deprecated, use [/workspace/t2v_self/iter_t2v/evaluations/extract_frames.py] instead
    def extract_frames(self, video_path: str, max_frames: int = 32) -> List[np.ndarray]:
        """
        Extract frames from video for processing.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frame arrays
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // max_frames)
        
        frames = []
        frame_count = 0
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(frames)} frames from video")
        return frames

    def compute_frame_similarity(self, frame: np.ndarray, text: str) -> float:
        """
        Compute CLIP similarity between a single frame and text.
        
        Args:
            frame: Frame as numpy array (RGB)
            text: Text prompt
            
        Returns:
            Similarity score
        """
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame)
        
        # Process inputs
        inputs = self.processor(
            text=[text], 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get normalized features
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            # Normalize features
            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            # Compute cosine similarity
            similarity = (image_features @ text_features.T).item()
            
        return similarity

    def get_video_text_similarity(self, video_path: str, text: str, max_frames: int = 32) -> List[float]:
        """
        Compute CLIP similarity for each frame and return a list of scores.
        
        Args:
            video_path: Path to video file
            text: Text prompt
            max_frames: Maximum frames to process
            
        Returns:
            List of similarity scores for each frame.
        """
        # Extract frames
        frames = self.extract_frames(video_path, max_frames)
        
        if not frames:
            print(f"Warning: No frames extracted from {video_path}")
            return []
        
        # Compute similarity for each frame
        frame_similarities = []
        for frame in frames:
            similarity = self.compute_frame_similarity(frame, text)
            frame_similarities.append(similarity)
        
        return frame_similarities

    def compute_video_similarity(self, video_path: str, text: str, max_frames: int = 32) -> dict:
        """
        Compute CLIP similarity between video and text.
        
        Args:
            video_path: Path to video file
            text: Text prompt
            max_frames: Maximum frames to process
            
        Returns:
            Dictionary with similarity metrics
        """
        # Extract frames
        frames = self.extract_frames(video_path, max_frames)
        
        if not frames:
            return {
                "error": "No frames extracted from video",
                "avg_similarity": 0.0,
                "frame_similarities": []
            }
        
        # Compute similarity for each frame
        frame_similarities = []
        for i, frame in enumerate(frames):
            similarity = self.compute_frame_similarity(frame, text)
            frame_similarities.append(similarity)
            print(f"Frame {i+1}/{len(frames)}: {similarity:.4f}")
        
        # Calculate statistics
        avg_similarity = np.mean(frame_similarities)
        max_similarity = np.max(frame_similarities)
        min_similarity = np.min(frame_similarities)
        std_similarity = np.std(frame_similarities)
        
        return {
            "video_path": video_path,
            "text": text,
            "num_frames": len(frames),
            "avg_similarity": float(avg_similarity),
            "max_similarity": float(max_similarity),
            "min_similarity": float(min_similarity),
            "std_similarity": float(std_similarity),
            "frame_similarities": frame_similarities
        }

    def batch_compute_similarities(self, video_path: str, texts: List[str], max_frames: int = 32) -> List[dict]:
        """
        Compute similarities for multiple text prompts.
        
        Args:
            video_path: Path to video file
            texts: List of text prompts
            max_frames: Maximum frames to process
            
        Returns:
            List of similarity results for each text
        """
        results = []
        for i, text in enumerate(texts):
            print(f"\nProcessing text {i+1}/{len(texts)}: {text}")
            result = self.compute_video_similarity(video_path, text, max_frames)
            results.append(result)
        
        return results


def load_texts_from_file(file_path: str) -> List[str]:
    """Load text prompts from a file (one per line)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts


def main():
    parser = argparse.ArgumentParser(description="Compute CLIP similarity between video and text")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--text", help="Text prompt")
    parser.add_argument("--text_file", help="File containing text prompts (one per line)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--max_frames", type=int, default=32, help="Maximum frames to process")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="CLIP model path")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.text and not args.text_file:
        parser.error("Either --text or --text_file must be provided")
    
    if not os.path.exists(args.video):
        parser.error(f"Video file not found: {args.video}")
    
    # Set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize CLIP model
    clip_similarity = CLIPVideoSimilarity(model_path=args.model, device=device)
    
    # Prepare texts
    texts = []
    if args.text:
        texts.append(args.text)
    if args.text_file:
        if not os.path.exists(args.text_file):
            parser.error(f"Text file not found: {args.text_file}")
        texts.extend(load_texts_from_file(args.text_file))
    
    print(f"Computing similarities for {len(texts)} text prompt(s)")
    
    # Compute similarities
    if len(texts) == 1:
        results = clip_similarity.compute_video_similarity(args.video, texts[0], args.max_frames)
    else:
        results = clip_similarity.batch_compute_similarities(args.video, texts, args.max_frames)
    
    # Print results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    
    if isinstance(results, dict):
        # Single text result
        print(f"Video: {results['video_path']}")
        print(f"Text: {results['text']}")
        print(f"Average Similarity: {results['avg_similarity']:.4f}")
        print(f"Max Similarity: {results['max_similarity']:.4f}")
        print(f"Min Similarity: {results['min_similarity']:.4f}")
        print(f"Std Similarity: {results['std_similarity']:.4f}")
    else:
        # Multiple texts results
        for i, result in enumerate(results):
            print(f"\nText {i+1}: {result['text']}")
            print(f"Average Similarity: {result['avg_similarity']:.4f}")
        
        # Find best matching text
        best_result = max(results, key=lambda x: x['avg_similarity'])
        print(f"\nBest Match: {best_result['text']}")
        print(f"Best Score: {best_result['avg_similarity']:.4f}")
    
    # Save results to file
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
