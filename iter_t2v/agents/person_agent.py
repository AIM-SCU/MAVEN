from __future__ import annotations

import copy
import json
import textwrap
from typing import Any, Dict

from agents.agent_base import AgentBase
from messages.messages_types import Prompt
from iter_t2v.utils.persona_utils import build_persona_prompt
from iter_t2v.utils.json_utils import call_llm_until_json
from messages.agent_result import AgentResult

from autogen_core.models import UserMessage, SystemMessage

class PersonAgent(AgentBase):
    """
    Refines ONLY the PERSON dimension of a prompt.
    Expects the LLM to return JSON:
        { "refined_prompt": "...", "justification": "..." }
    """

    # ------------------------------------------------------------------ #
    def __init__(self, llm_client, name: str = "PersonAgent") -> None:
        super().__init__(name)
        self.llm_client = llm_client
        self.last_llm_prompt = None

    # ------------------------------------------------------------------ #
    async def refine(self, prompt: Prompt) -> Prompt:  # noqa: D401
        """
        Call an LLM with a persona-specific system prompt and return a new
        Prompt whose `.text` comes from the LLM’s "refined_prompt" field.
        """
        persona_line = build_persona_prompt(prompt, slice_="person")

        system_prompt = textwrap.dedent(
            f"""
            {persona_line}

            Your task is to improve only the appearance of the person in the prompt.
            Focus on visual traits like facial features, hairstyle, or clothing.
            Do not change the action or the location.

            Return JSON with:
            "refined_prompt": the updated sentence
            "justification": a brief explanation of how your addition makes the person more visually distinctive or culturally aligned.
            """
        ).strip()

        user_msg = prompt.text

        # ─── Real LLM call wrapped with JSON extractor ───────────────────
        async def _ask_llm() -> str:
            # Build the message list exactly once
            messages = [
                SystemMessage(content=system_prompt, source="system"),
                UserMessage(content=user_msg,       source="user"),
            ]

            # Store the actual prompt sent to LLM
            self.last_llm_prompt = messages
            
            assistant_reply = await self.llm_client.create(messages)
            raw_text = assistant_reply.content.strip()
            return raw_text

        reply: Dict[str, Any] = await call_llm_until_json(
            _ask_llm, 
            on_max_attempts_reached=self.llm_client.change_seed,
            on_temperature_change=self.llm_client.change_temperature,
            on_seed_reset=self.llm_client.reset_seed_to_42
        )
        refined_text: str = reply["refined_prompt"]
        justification: str = reply.get("justification", "")

        # ─── STUB fallback (uncomment while offline testing) ─────────────
        # raw_reply = json.dumps(
        #     {
        #         "refined_prompt": (
        #             f"{prompt.person_segment} wearing traditional "
        #             f"{prompt.person_culture.lower()} attire "
        #             f"{prompt.action_segment} at {prompt.location_segment}"
        #         ),
        #         "justification": f"added {prompt.person_culture} attire",
        #     }
        # )
        # reply: Dict[str, Any] = json.loads(raw_reply)
        # refined_text = reply["refined_prompt"]
        # justification = reply["justification"]
        # ──────────────────────────────────────────────────────────────────

        # Optionally log justification
        # print(f"[{self.name}] {justification}")

        new_prompt = copy.deepcopy(prompt)
        new_prompt.text = refined_text
        return AgentResult(
            prompt=new_prompt,
            justification=justification,
            llm_prompt=self.last_llm_prompt
        )
