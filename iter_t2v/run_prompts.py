#!/usr/bin/env python
"""
python run_prompts.py --mode base                 # refine prompts only
python run_prompts.py --mode base --generate_video  # read JSONL → gen videos

# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=1 python run_prompts.py --mode parallel --generate_video --partition 51

# Terminal 2 (GPU 1) 
CUDA_VISIBLE_DEVICES=3 python run_prompts.py --mode parallel --generate_video --partition 53

# Terminal 1 (GPU 0) - refinement (refine prompts)
CUDA_VISIBLE_DEVICES=0 python run_prompts.py --mode parallel --partition 21

# Terminal 2 (GPU 1) - refinement (refine prompts)
CUDA_VISIBLE_DEVICES=5 python run_prompts.py --mode parallel --partition 22
"""
from __future__ import annotations
import asyncio, argparse, json, sys, pathlib
from pathlib import Path
from typing import Any
import datetime

# ----- project path -----
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
sys.path.append(str(project_root.parent))

from prompt_loader import iter_prompts
from pipeline.generator import PromptGenerator
from autogen_ext.models.openai import OpenAIChatCompletionClient
from llm_router import LLMRouter


# ----- JSON helper -----------------------------------------------
def to_json_safe(obj: Any):
    if obj is ...:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    # Handle message objects from LLM prompts
    if hasattr(obj, '__class__') and obj.__class__.__name__ in ['SystemMessage', 'UserMessage', 'AssistantMessage']:
        return {
            "type": obj.__class__.__name__,
            "content": getattr(obj, 'content', ''),
            "source": getattr(obj, 'source', '')
        }
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    return obj
# ------------------------------------------------------------------


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["base", "single", "parallel", "sequential"], default="base")
    p.add_argument("--template_jsonl", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--experiment_tag", default="pal_template_v1")
    p.add_argument("--generate_video", action="store_true")
    p.add_argument("--partition", type=str, default=None, 
                   help="Partition format: XY where X=total partitions, Y=current partition (1-indexed). E.g., '21' = 2 partitions, 1st partition")
    args = p.parse_args()

    # ----- paths ---------------------------------------------------
    out_dir = Path(args.out_dir or f"/workspace/t2v_self/iter_t2v/results/{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    refined_path = out_dir / f"prompts_{args.mode}.jsonl"
    video_dir = out_dir / "videos"
    video_backup_log = out_dir / "video_generation_backup.jsonl"  # backup log

    # ===== VIDEO-ONLY BRANCH =======================================
    if args.generate_video:
        if not refined_path.exists():
            raise FileNotFoundError(
                f"{refined_path} not found.  Run without --generate_video first."
            )

        from clients.t2v_generation import T2VGenerationClient
        t2v_client = T2VGenerationClient()

        # Load file once, keep it in memory
        refined_records = [json.loads(l) for l in refined_path.open("r", encoding="utf-8")]

        # Parse partition parameter
        total_partitions = 1
        current_partition = 1
        if args.partition:
            if len(args.partition) != 2 or not args.partition.isdigit():
                raise ValueError("Partition format should be XY (e.g., '21' for 2 partitions, 1st partition)")
            total_partitions = int(args.partition[0])
            current_partition = int(args.partition[1])
            if current_partition < 1 or current_partition > total_partitions:
                raise ValueError(f"Current partition {current_partition} must be between 1 and {total_partitions}")

        # Apply absolute partitioning to ALL records first
        if total_partitions > 1:
            total_records = len(refined_records)
            records_per_partition = total_records // total_partitions
            remainder = total_records % total_partitions
            
            # Calculate absolute start and end indices for current partition
            start_idx = (current_partition - 1) * records_per_partition
            if current_partition <= remainder:
                start_idx += (current_partition - 1)
                partition_size = records_per_partition + 1
            else:
                start_idx += remainder
                partition_size = records_per_partition
            
            end_idx = start_idx + partition_size
            # Apply absolute partitioning to all records
            partitioned_records = refined_records[start_idx:end_idx]
            
            print(f"Absolute Partition {current_partition}/{total_partitions}: Assigned records {start_idx}-{end_idx-1} ({len(partitioned_records)} total)")
        else:
            partitioned_records = refined_records
            start_idx = 0

        # Filter records that need video generation from the partitioned subset
        records_needing_video = []
        for local_idx, rec in enumerate(partitioned_records):
            vp = rec.get("video_path")
            if not (vp and Path(vp).exists()):
                records_needing_video.append((start_idx + local_idx, rec))

        print(f"Found {len(records_needing_video)} records needing video generation in this partition")
        
        if not records_needing_video:
            print("No videos to generate in this partition!")
            return

        for local_idx, (global_idx, rec) in enumerate(records_needing_video, 1):
            vp = rec.get("video_path")
            if vp and Path(vp).exists():
                continue   # clip already exists

            # ── Generate and record video path ───────────────────────
            vp_path, vid_id = await t2v_client.generate(
                rec["final_prompt"], save_dir=video_dir
            )
            # sanity-check: decode ID back to prompt
            from utils.codec_utils import id_matches_prompt
            assert id_matches_prompt(vid_id, rec["final_prompt"]), \
                   "ID does not match prompt hash!"

            # ── Save to backup log immediately ──────────────────────
            backup_entry = {
                "global_idx": global_idx,
                "original_prompt": rec.get("original_prompt", ""),
                "final_prompt": rec["final_prompt"],
                "video_path": str(vp_path),
                "video_id": vid_id,
                "timestamp": datetime.datetime.now().isoformat()
            }
            with video_backup_log.open("a", encoding="utf-8") as backup_f:
                backup_f.write(json.dumps(to_json_safe(backup_entry), ensure_ascii=False) + "\n")

            refined_records[global_idx]["video_path"] = str(vp_path)
            refined_records[global_idx]["video_id"] = vid_id
            print(f"[{local_idx}/{len(records_needing_video)}]  ✔  {vp_path}")

            # ── Update only the specific record in JSONL file ───────────
            try:
                # Read the current file content
                current_records = []
                if refined_path.exists():
                    with refined_path.open("r", encoding="utf-8") as fin:
                        current_records = [json.loads(line) for line in fin]
                
                # Update the specific record
                if global_idx < len(current_records):
                    current_records[global_idx]["video_path"] = str(vp_path)
                    current_records[global_idx]["video_id"] = vid_id
                    
                    # Write back the entire file with the updated record
                    with refined_path.open("w", encoding="utf-8") as fout:
                        for r in current_records:
                            fout.write(json.dumps(to_json_safe(r), ensure_ascii=False) + "\n")
                else:
                    print(f"WARNING: global_idx {global_idx} out of range for current records length {len(current_records)}")
                    
            except Exception as e:
                print(f"WARNING: Failed to update main JSONL file: {e}")
                print(f"Video info saved in backup log: {video_backup_log}")

        print(f"Video generation complete → {video_dir}")
        print(f"Backup log saved → {video_backup_log}")
        return  # EARLY EXIT
    # ===== END VIDEO-ONLY ==========================================

    # ----- load any existing refined prompts to skip duplicates ----
    existing_prompts: set[str] = set()
    if refined_path.exists():
        with refined_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_prompts.add(json.loads(line)["original_prompt"])
                except Exception:
                    pass  # tolerate malformed historical rows

    # ----- collect all prompts for potential partitioning ----------
    all_prompts = list(iter_prompts(args.template_jsonl, experiment_tag=args.experiment_tag))
    
    # ----- apply absolute partitioning for refinement if specified ----------
    if args.partition:
        if len(args.partition) != 2 or not args.partition.isdigit():
            raise ValueError("Partition format should be XY (e.g., '21' for 2 partitions, 1st partition)")
        total_partitions = int(args.partition[0])
        current_partition = int(args.partition[1])
        if current_partition < 1 or current_partition > total_partitions:
            raise ValueError(f"Current partition {current_partition} must be between 1 and {total_partitions}")
        
        # Apply absolute partitioning to ALL prompts first
        total_prompts = len(all_prompts)
        prompts_per_partition = total_prompts // total_partitions
        remainder = total_prompts % total_partitions
        
        # Calculate absolute start and end indices for current partition
        start_idx = (current_partition - 1) * prompts_per_partition
        if current_partition <= remainder:
            start_idx += (current_partition - 1)
            partition_size = prompts_per_partition + 1
        else:
            start_idx += remainder
            partition_size = prompts_per_partition
        
        end_idx = start_idx + partition_size
        # Apply absolute partitioning
        partitioned_prompts = all_prompts[start_idx:end_idx]
        
        print(f"Absolute Refinement Partition {current_partition}/{total_partitions}: Assigned prompts {start_idx}-{end_idx-1} ({len(partitioned_prompts)} total)")
        
        # Filter out already processed prompts from the partitioned subset
        prompts_to_process = [p for p in partitioned_prompts if p.text not in existing_prompts]
        print(f"Found {len(prompts_to_process)} new prompts to process in this partition")
    else:
        prompts_to_process = [p for p in all_prompts if p.text not in existing_prompts]

    # ----- shared LLM ----------------------------------------------
    # NOTE: To run on multiple GPUs with Ollama, you need to run separate services.
    # The '500 Internal Server Error' with 'NoneType' object has no attribute 'acompletion'
    # from litellm usually means it failed to route the request, often due to an issue
    # parsing the model name from the command line.
    #
    # ALTERNATIVE: Skip litellm and connect directly to Ollama:
    # 1. Run `ollama serve` on different GPUs and ports:
    #    Terminal 1 (GPU 0):
    #    > CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11434 ollama serve
    #
    #    Terminal 2 (GPU 5):
    #    > CUDA_VISIBLE_DEVICES=5 OLLAMA_HOST=127.0.0.1:11435 ollama serve
    #
    #    Then use ports [11434, 11435] in the router and model "llama3.1:70b-instruct-q8_0"
    #
    # CURRENT APPROACH: litellm proxies:
    # 2. Run `litellm` proxies pointing to each ollama instance.
    #    Using quotes around the model name is safer. Add --debug for verbose logs.
    #    Terminal 3 (proxy for GPU 0):
    #    > litellm --model ollama/llama3.1:70b-instruct-q8_0 --api_base http://127.0.0.1:11434 --port 4000 --debug
    #
    #    Terminal 4 (proxy for GPU 5):
    #    > litellm --model ollama/llama3.1:70b-instruct-q8_0 --api_base http://127.0.0.1:11435 --port 4001 --debug
    #
    # This script will then connect to the litellm proxies on ports 4000 and 4001.
    llm_client = LLMRouter(
        ports=[4000, 4001],
        model="ollama/llama3.1:70b-instruct-q8_0",
        seed=42,  # for reproducibility
        temperature=0.4,  # low temp for consistency
    )
    
    generator = PromptGenerator(llm_client, generation_mode=args.mode)

    # stream-append mode ('a')
    with refined_path.open("a", encoding="utf-8") as fout:
        for prompt in prompts_to_process:
            llm_client.reset_seed()  # Reset seed for each new prompt
            run_info = await generator(prompt)   # dict with original/ final / timeline
            fout.write(json.dumps(to_json_safe(run_info), ensure_ascii=False) + "\n")
            fout.flush()                         # persist each row immediately
            existing_prompts.add(prompt.text)

    print(f"Refinement complete – results appended to {refined_path}")
    if args.partition:
        print(f"Processed partition {current_partition}/{total_partitions}")


if __name__ == "__main__":
    asyncio.run(main())
