from __future__ import annotations
import os, uuid, gc
from pathlib import Path
from utils.codec_utils import prompt_to_id
from typing import Dict, Any

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from rich.console import Console

console = Console()


class T2VGenerationClient:
    """
    Thin wrapper around CogVideoX (or any Diffusers-compatible pipeline).
    """

    def __init__(
        self,
        model_id: str = "THUDM/CogVideoX-5b",
        dtype=torch.bfloat16,
        default_output_root: str = "/workspace/t2v_self/iter_t2v/results/videos",
        **generation_args: Dict[str, Any],
    ) -> None:
        # ---------- Defaults that callers may override ----------------
        default_generation_args = dict(
            num_videos_per_prompt=1,
            num_inference_steps=50,     # TODO: may need to change
            num_frames=41,  # 41 frames = 5 seconds at 8 FPS
            guidance_scale=6,           # TODO: may need to change
            generator=None,          # we’ll set seed lazily
        )
        for k, v in default_generation_args.items():
            generation_args.setdefault(k, v)
        # ----------------------------------------------------------------

        self.model_id = model_id
        self.dtype = dtype
        self.default_root = Path(default_output_root)
        self.generation_args = generation_args

        # ---- Load once; stays on CPU by default ------------------------
        console.print(f"[cyan]Loading CogVideoX model ({model_id}) …[/cyan]")
        self.pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.vae.enable_tiling()
        self.pipe.to("cpu")

    # ------------------------------------------------------------------
    async def generate(self, prompt: str, save_dir: Path | None = None):
        """
        Generate a single MP4 for the given prompt.
        Returns the path to the saved video file.
        """
        save_dir = save_dir or self.default_root
        save_dir.mkdir(parents=True, exist_ok=True)
        vid_id = prompt_to_id(prompt, length=16)   # ≤16 chars
        outfile = save_dir / f"{vid_id}.mp4"

        if outfile.exists():
            console.print(f"[yellow]Skip – already exists:[/yellow] {outfile}")
            return outfile, vid_id

        # ------- Move to GPU if available (else CPU) -------------------
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe.to(device_str)

        # ------- Prepare deterministic generator -----------------------
        if self.generation_args["generator"] is None:
            self.generation_args["generator"] = torch.Generator(device=device_str).manual_seed(42)

        console.print(f"[green]T2V:[/green] Generating video …")
        output = self.pipe(prompt=prompt, **self.generation_args)
        frames = output.frames[0]

        export_to_video(frames, outfile, fps=8)
        console.print(f"[bold green]Saved video →[/bold green] {outfile}")

        # ------- Clean up GPU memory -----------------------------------
        del output, frames
        self.generation_args["generator"] = None   # new seed next call
        self.pipe.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        return outfile, vid_id
