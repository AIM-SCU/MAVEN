"""
Utility helpers for reliably extracting JSON from LLM replies.
"""

from __future__ import annotations
import json
import re
import logging
from typing import Any, Dict, Callable, Awaitable


JSON_RE = re.compile(r"\{.*\}", flags=re.S)


def extract_json(text: str) -> Dict[str, Any]:
    """
    Find the first {...} block in `text` and load it as JSON.
    Raises ValueError if no JSON found or decode fails.
    """
    match = JSON_RE.search(text)
    if not match:
        raise ValueError("No JSON object detected in LLM reply.")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {e}") from e


async def call_llm_until_json(
    llm_call: Callable[[], Awaitable[str]],
    on_max_attempts_reached: Callable[[], Awaitable[None] | None] = None,
    on_temperature_change: Callable[[float], Awaitable[None] | None] = None,
    on_seed_reset: Callable[[int], Awaitable[None] | None] = None,
    hard_limit: int = 21,
) -> Dict[str, Any]:
    """
    Repeatedly call an async `llm_call` until `extract_json` succeeds
    or `max_attempts` is reached (default 3).

    Temperature progression: Start at 0.4, increase by 0.1 every 3 seed attempts, up to 1.0.
    For each new temperature, reset to seed 42 first.
    Hard limit stops after 3 seeds at temperature 1.0 (total 21 attempts).

    Parameters
    ----------
    llm_call : async function returning LLM text
    max_attempts : how many tries before changing seed/temperature (default 3)
    on_max_attempts_reached : async callback to execute when max attempts reached (e.g., change seed)
    on_temperature_change : async callback to change temperature when moving to next temperature level
    on_seed_reset : async callback to reset seed to 42 when temperature changes
    hard_limit : absolute max attempts before raising error (default 21: 7 temps × 3 attempts)

    Returns
    -------
    dict
        Parsed JSON from the model.
    """
    temperatures = [round(0.4 + 0.1 * i, 1) for i in range(7)]  # [0.4, 0.5, ..., 1.0]
    total_attempts = 0
    for temp in temperatures:
        # Set temperature and reset seed to 42
        if on_seed_reset:
            result = on_seed_reset() if callable(on_seed_reset) else None
            if result is not None and hasattr(result, '__await__'):
                await result
        if on_temperature_change:
            result = on_temperature_change(temp) if callable(on_temperature_change) else None
            if result is not None and hasattr(result, '__await__'):
                await result
        # Try 3 seeds per temperature
        for seed_attempt in range(3):
            total_attempts += 1
            raw = await llm_call()
            try:
                return extract_json(raw)
            except ValueError as err:
                logging.warning(f"Attempt {seed_attempt+1} at temp {temp} (total {total_attempts}) failed: {err}. Retrying…")
                if seed_attempt < 2:
                    # Change seed for next attempt (except after 3rd)
                    if on_max_attempts_reached:
                        result = on_max_attempts_reached() if callable(on_max_attempts_reached) else None
                        if result is not None and hasattr(result, '__await__'):
                            await result
        # After 3 seeds, move to next temperature
    raise RuntimeError(f"call_llm_until_json: Tried all seeds and temperatures without valid JSON.")
