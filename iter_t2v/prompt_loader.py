# /workspace/t2v_self/iter_t2v/prompt_loader.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator, List

from messages.messages_types import Prompt

"""
python - <<'PY'
from prompt_loader import load_prompts
prompts = load_prompts()        # uses the default path above
print(prompts[0])               # sanity-check
PY
"""
# --------------------------------------------------------------------------- #
# Default template path (can be overridden in load/iter functions)
DEFAULT_TEMPLATE_PATH = (
    "/workspace/t2v_self/iter_t2v/templates/pal_prompts_v1.jsonl"
)
# --------------------------------------------------------------------------- #


def load_prompts(
    jsonl_path: str | None = None,
    experiment_tag: str = "pal_template_v1",
) -> List[Prompt]:
    """
    Load the entire JSONL file into memory and return a list of Prompt objects.
    """
    path = Path(jsonl_path or DEFAULT_TEMPLATE_PATH)
    prompts: List[Prompt] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["experiment_tag"] = experiment_tag
            prompts.append(Prompt.from_row(row))

    return prompts


def iter_prompts(
    jsonl_path: str | None = None,
    experiment_tag: str = "pal_template_v1",
) -> Iterator[Prompt]:
    """
    Memory-friendly generator yielding one Prompt at a time.
    """
    path = Path(jsonl_path or DEFAULT_TEMPLATE_PATH)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["experiment_tag"] = experiment_tag
            yield Prompt.from_row(row)
