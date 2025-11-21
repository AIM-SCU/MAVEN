# utils/codec_utils.py
import base64, hashlib

def prompt_to_id(prompt: str, length: int = 16) -> str:
    """Return a short, deterministic ID (≤ length chars)."""
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()[:8]  # 64-bit
    b64 = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")  # 11 chars
    if length >= len(b64):
        return b64
    return b64[:length]                 # final safety cut-off

def id_matches_prompt(vid_id: str, prompt: str) -> bool:
    """True if vid_id equals the prompt's computed ID (same truncation)."""
    return prompt_to_id(prompt, len(vid_id)) == vid_id
