"""
Utilities for building concise persona strings that become part of an LLM
system prompt.  The persona sentence is short, culture-aware, and dimension-
specific (person / action / location).

Example usage
-------------
from utils.persona_utils import build_persona_prompt
persona = build_persona_prompt(prompt, slice_="person")
"""
from __future__ import annotations
from typing import Set

from messages.messages_types import Prompt


def _culture_list_to_str(cultures: Set[str]) -> str:
    """
    Helper: turn {'Chinese', 'Romanian'} into 'Chinese and Romanian',
    or a single element set into just that element.
    """
    if not cultures:
        return ""
    if len(cultures) == 1:
        return next(iter(cultures))
    *first, last = sorted(cultures)
    return f"{', '.join(first)} and {last}"


def build_persona_prompt(prompt: Prompt, slice_: str) -> str:
    """
    Return a *one-sentence* persona description tailored to the cultural
    dimension an agent is about to refine.

    Parameters
    ----------
    prompt : Prompt
        The current Prompt instance.
    slice_ : str
        Which dimension the agent will refine. One of
        {'person', 'action', 'location', 'all'}.

    Examples
    --------
    >>> build_persona_prompt(prompt, slice_="person")
    'You are someone steeped in Chinese culture and recognise its everyday looks.'
    """
    slice_ = slice_.lower()
    if slice_ not in {"person", "action", "location", "all"}:
        raise ValueError("slice_ must be person | action | location | all")

    if slice_ == "person":
        culture = prompt.person_culture
        return f"You are a {culture} individual who understands typical appearance traits from this culture."

    if slice_ == "action":
        culture = prompt.action_culture
        return f"You are a {culture} observer skilled at describing how people typically eat, play music, and dance."

    if slice_ == "location":
        culture = prompt.location_culture
        return f"You are a {culture} tour guide who knows how to visually describe iconic landmarks."

    # slice_ == "all"  (single-agent mode or generic persona)
    cultures = {
        prompt.person_culture,
        prompt.action_culture,
        prompt.location_culture,
    }
    cultures_str = _culture_list_to_str(cultures)
    # return (
    #     f"You are well-versed in {cultures_str} cultures and can enrich any "
    #     f"aspect of the scene when asked."
    # )
    return f"You are someone familiar with {cultures_str} cultural settings and can enrich any part of the scene."

