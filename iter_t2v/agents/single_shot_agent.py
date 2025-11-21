from __future__ import annotations

import copy
import textwrap
from typing import Any, Dict

from autogen_core.models import SystemMessage, UserMessage

from agents.agent_base import AgentBase
from iter_t2v.utils.persona_utils import build_persona_prompt
from iter_t2v.utils.json_utils import call_llm_until_json
from messages.messages_types import Prompt
from messages.agent_result import AgentResult

class SingleShotAgent(AgentBase):
    """
    One-pass refinement across PERSON, ACTION, and LOCATION.
    Expects JSON:
        { "refined_prompt": "...", "justification": "..." }
    """

    def __init__(self, llm_client, name: str = "SingleShotAgent") -> None:
        super().__init__(name)
        self.llm_client = llm_client
        self.last_llm_prompt = None

    # ------------------------------------------------------------------ #
    async def refine(self, prompt: Prompt) -> Prompt:  # noqa: D401
        persona_line = build_persona_prompt(prompt, slice_="all")

        system_prompt = textwrap.dedent(
            f"""
            {persona_line}

            Your task is to improve the full prompt by refining the person, action, and location.
            Focus on:
            • appearance details (e.g., facial features, hairstyle, clothing)
            • how the action is performed (e.g., body posture, hand movement, visible objects)
            • iconic aspects of the place (e.g., landscape, architecture, lighting, environmental elements)

            Do not introduce new people, actions, or locations.

            Return JSON with:
            "refined_prompt": the updated sentence
            "justification": a brief explanation of how your changes improve the visual clarity or cultural specificity of the scene.
            """
        ).strip()

        user_msg = prompt.text

        # ---------- LLM call wrapped with JSON extraction ----------
        async def _ask_llm() -> str:
            assistant_msg = await self.llm_client.create(
                [
                    SystemMessage(content=system_prompt, source="system"),
                    UserMessage(content=user_msg, source="user"),
                ]
            )
            return assistant_msg.content.strip()

        reply: Dict[str, Any] = await call_llm_until_json(
            _ask_llm, 
            on_max_attempts_reached=self.llm_client.change_seed,
            on_temperature_change=self.llm_client.change_temperature,
            on_seed_reset=self.llm_client.reset_seed_to_42
        )
        refined_text = reply["refined_prompt"]
        justification: str = reply.get("justification", "")

        # -----------------------------------------------------------
        new_prompt = copy.deepcopy(prompt)
        new_prompt.text = refined_text

        # Store the actual prompt sent to LLM
        self.last_llm_prompt = [
            SystemMessage(content=system_prompt, source="system"),
            UserMessage(content=user_msg, source="user"),
        ]

        response = await self.llm_client.create(self.last_llm_prompt)

        return AgentResult(
            prompt=new_prompt,
            justification=justification,
            llm_prompt=self.last_llm_prompt,
        )
