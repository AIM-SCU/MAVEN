from __future__ import annotations

import copy
import json
import textwrap
from typing import Any, Dict

from autogen_core.models import SystemMessage, UserMessage

from agents.agent_base import AgentBase
from iter_t2v.utils.persona_utils import build_persona_prompt
from iter_t2v.utils.json_utils import call_llm_until_json
from messages.messages_types import Prompt
from messages.agent_result import AgentResult

class ActionAgent(AgentBase):
    """
    Refines ONLY the ACTION portion of a prompt.
    Returns JSON: {"refined_prompt": "...", "justification": "..."}
    """

    def __init__(self, llm_client, name: str = "ActionAgent") -> None:
        super().__init__(name)
        self.llm_client = llm_client
        self.last_llm_prompt = None

    # ------------------------------------------------------------------ #
    async def refine(self, prompt: Prompt) -> Prompt:
        persona_line = build_persona_prompt(prompt, slice_="action")

        system_prompt = textwrap.dedent(
            f"""
            {persona_line}

            Your task is to improve only the action portion of the prompt.
            Focus on how the action is performed: body posture, hand movement, or any visible objects.
            Do not change the person or the location.

            Return JSON with:
            "refined_prompt": the updated sentence
            "justification": a brief explanation of why your added details enhance the clarity, vividness, or cultural alignment of the action.
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

        # Store the actual prompt sent to LLM
        messages = [
            SystemMessage(content=system_prompt, source="system"),
            UserMessage(content=user_msg, source="user"),
        ]
        self.last_llm_prompt = messages

        response = await self.llm_client.create(messages)

        reply: Dict[str, Any] = await call_llm_until_json(
            _ask_llm, 
            on_max_attempts_reached=self.llm_client.change_seed,
            on_temperature_change=self.llm_client.change_temperature,
            on_seed_reset=self.llm_client.reset_seed_to_42
        )
        refined_text = reply["refined_prompt"]
        justification: str = reply.get("justification", "")

        new_prompt = copy.deepcopy(prompt)
        new_prompt.text = refined_text
        return AgentResult(
            prompt=new_prompt,
            justification=justification,
            llm_prompt=self.last_llm_prompt
        )
