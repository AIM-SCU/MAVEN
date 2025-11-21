#!/usr/bin/env python3
"""
Test script for EvalCrafter partitioning
=========================================

This script demonstrates how to use the partitioning feature of the EvalCrafter runner.
"""

import subprocess
import sys
from pathlib import Path

def run_partition_test(mode="base", total_partitions=2):
    """Test partitioning functionality."""
    script_path = Path(__file__).parent / "run_evalcrafter.py"
    
    print(f"Testing partitioning with {total_partitions} partitions for mode '{mode}'")
    
    for partition in range(total_partitions):
        print(f"\n=== Running partition {partition} ===")
        
        cmd = [
            "python", str(script_path),
            "--mode", mode,
            "--partition", str(partition),
            "--total-partitions", str(total_partitions),
            "--gpu", str(partition % 4)  # Cycle through GPUs 0-3
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        # For demonstration, just show the command without actually running it
        # Uncomment the next lines to actually run the partitions
        # result = subprocess.run(cmd, capture_output=True, text=True)
        # print(f"Return code: {result.returncode}")
        # if result.stdout:
        #     print(f"STDOUT:\n{result.stdout}")
        # if result.stderr:
        #     print(f"STDERR:\n{result.stderr}")

def show_usage_examples():
    """Show usage examples for the EvalCrafter script."""
    print("EvalCrafter Runner Usage Examples:")
    print("=" * 40)
    
    print("\n1. Process all videos in base mode:")
    print("   python evaluations/ec_video/run_evalcrafter.py --mode base")
    
    print("\n2. Process all videos with specific GPU:")
    print("   python evaluations/ec_video/run_evalcrafter.py --mode base --gpu 2")
    
    print("\n3. Process partition 1 of 2 on GPU 0:")
    print("   python evaluations/ec_video/run_evalcrafter.py --mode base --partition 21 --gpu 0")
    
    print("\n4. Process partition 2 of 2 on GPU 1:")
    print("   python evaluations/ec_video/run_evalcrafter.py --mode base --partition 22 --gpu 1")
    
    print("\n5. Process partition 2 of 4 on GPU 2:")
    print("   python evaluations/ec_video/run_evalcrafter.py --mode base --partition 42 --gpu 2")
    
    print("\n6. Process sequential mode partition 1 of 8:")
    print("   python evaluations/ec_video/run_evalcrafter.py --mode sequential --partition 81 --gpu 0")
    
    print("\nPartition Format: XY where X=total partitions, Y=partition index (1-based)")
    print("Examples:")
    print("  21 = partition 1 of 2")
    print("  22 = partition 2 of 2") 
    print("  42 = partition 2 of 4")
    print("  81 = partition 1 of 8")
    
    print("\nOutputs:")
    print("- JSONL files: results/{mode}/video_*_with_ec[_partN].jsonl")
    print("- Log files: results/{mode}/evalcrafter_metrics_log[_partN].txt")
    
    print("\nParallel Processing Example (4 GPUs, 4 partitions):")
    print("# Terminal 1:")
    print("python evaluations/ec_video/run_evalcrafter.py --mode base --partition 41 --gpu 0")
    print("# Terminal 2:")
    print("python evaluations/ec_video/run_evalcrafter.py --mode base --partition 42 --gpu 1")
    print("# Terminal 3:")
    print("python evaluations/ec_video/run_evalcrafter.py --mode base --partition 43 --gpu 2")
    print("# Terminal 4:")
    print("python evaluations/ec_video/run_evalcrafter.py --mode base --partition 44 --gpu 3")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_partition_test()
    else:
        show_usage_examples()
