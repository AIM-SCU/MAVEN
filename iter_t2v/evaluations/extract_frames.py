#!/usr/bin/env python3
"""
Video Frame Extraction Script
============================

This script extracts frames from videos in the results folder based on the specified mode.

For each mode:
- Input: videos from results/{mode}/videos/ folder
- Output: frames saved to results/{mode}/frames/ folder
- Processes videos in alphabetical order from the videos folder

Features:
- Extracts specified number of frames from each video
- Supports all modes: base, single, parallel, sequential
- Organizes frames by video name in separate subdirectories
- Uses OpenCV for fast and precise frame extraction
- Maintains aspect ratio and quality

Usage:
    # Extract 5 frames from base mode videos
    python evaluations/extract_frames.py --mode base --num-frames 5
    python evaluations/extract_frames.py --mode single --num-frames 5
    
    # Extract 16 frames from sequential mode videos
    python evaluations/extract_frames.py --mode sequential --num-frames 16
    
    # Extract frames with custom output directory
    python evaluations/extract_frames.py --mode parallel --num-frames 10 --output custom_frames_dir
"""
import argparse
from pathlib import Path
import sys
import os
from typing import List
import cv2
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODES = ["base", "single", "parallel", "sequential"]


def get_video_files(videos_dir: Path) -> List[Path]:
    """Get list of video files in directory, sorted alphabetically."""
    if not videos_dir.exists():
        return []
    
    # Common video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    
    video_files = []
    for file_path in videos_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    # Sort alphabetically
    video_files.sort(key=lambda x: x.name)
    return video_files


def sample_n_frames(video_path: str, num_frames: int = 5):
    """
    Return `num_frames` RGB frames, evenly spaced across the whole video.
    Works for any FPS, any duration.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise ValueError("Video contains zero frames")

    # Choose `num_frames` integer indices from 0 … total-1
    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue                       # skip if read fails
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def extract_frames_from_video(video_path: Path, output_dir: Path, num_frames: int) -> bool:
    """Extract frames from a single video using OpenCV."""
    try:
        # Create output directory for this video
        video_name = video_path.stem  # filename without extension
        frames_output_dir = output_dir / video_name
        frames_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frames using OpenCV
        frames = sample_n_frames(str(video_path), num_frames)
        
        if not frames:
            print(f"  No frames extracted from {video_path.name}")
            return False
        
        # Save frames as JPEG
        for i, frame_rgb in enumerate(frames):
            output_frame = frames_output_dir / f"frame_{i:04d}.jpg"
            # Convert RGB to BGR for OpenCV saving
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_frame), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"  Extracted {len(frames)} frames to {frames_output_dir}")
        return True
        
    except Exception as e:
        print(f"  Error extracting frames: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("--mode", choices=MODES, required=True, help="Mode to process")
    parser.add_argument("--num-frames", type=int, required=True, help="Number of frames to extract from each video")
    parser.add_argument("--output", help="Custom output directory (optional, defaults to results/{mode}/frames)")
    args = parser.parse_args()
    
    if args.num_frames <= 0:
        print("Error: Number of frames must be positive")
        sys.exit(1)
    
    # Determine input and output directories
    videos_dir = RESULTS_DIR / args.mode / "videos"
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = RESULTS_DIR / args.mode / "frames"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if videos directory exists
    if not videos_dir.exists():
        print(f"Error: Videos directory not found: {videos_dir}")
        sys.exit(1)
    
    # Get video files
    video_files = get_video_files(videos_dir)
    
    if not video_files:
        print(f"Error: No video files found in {videos_dir}")
        sys.exit(1)
    
    print(f"Found {len(video_files)} video files in {videos_dir}")
    print(f"Extracting {args.num_frames} frames from each video")
    print(f"Output directory: {output_dir}")
    
    # Check if OpenCV can be imported (already imported above, but verify it works)
    try:
        cv2.VideoCapture(0)  # Test creation
    except Exception as e:
        print(f"Error: OpenCV not properly installed: {e}")
        sys.exit(1)
    
    # Process each video
    successful_extractions = 0
    failed_extractions = 0
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\nProcessing video {i}/{len(video_files)}: {video_path.name}")
        
        success = extract_frames_from_video(video_path, output_dir, args.num_frames)
        
        if success:
            successful_extractions += 1
        else:
            failed_extractions += 1
            print(f"  Failed to extract frames from {video_path.name}")
    
    # Summary
    print(f"\n" + "="*50)
    print(f"Frame extraction completed for mode '{args.mode}'")
    print(f"Total videos processed: {len(video_files)}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")
    print(f"Frames saved to: {output_dir}")
    
    if failed_extractions > 0:
        print(f"\nWarning: {failed_extractions} videos failed frame extraction")
        sys.exit(1)


if __name__ == "__main__":
    main()
