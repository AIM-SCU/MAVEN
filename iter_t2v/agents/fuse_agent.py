from __future__ import annotations

import copy
import textwrap
from typing import List, Dict, Any

from autogen_core.models import SystemMessage, UserMessage

from agents.agent_base import AgentBase
from iter_t2v.utils.persona_utils import build_persona_prompt
from iter_t2v.utils.json_utils import call_llm_until_json
from messages.messages_types import Prompt
from messages.agent_result import AgentResult


class FuseAgent(AgentBase):
    """
    Combines three independently-refined prompts (person, action, location)
    into ONE coherent sentence.  Expects the LLM to emit JSON:
        { "refined_prompt": "...", "justification": "..." }
    """

    def __init__(self, llm_client, name: str = "FuseAgent") -> None:
        super().__init__(name)
        self.llm_client = llm_client
        self.last_llm_prompt = None

    # ------------------------------------------------------------------ #
    async def refine(self, prompts: List[Prompt]) -> Prompt:  # type: ignore[override]
        if len(prompts) != 3:
            raise ValueError("FuseAgent expects exactly three Prompt objects")

        # Use culture union for persona
        base_prompt = prompts[0]              # metadata seed
        persona_line = build_persona_prompt(base_prompt, slice_="all")

        # Collect candidate sentences
        candidate_sents = [p.text for p in prompts]
        joined = "\n---\n".join(candidate_sents)

        # system_prompt = textwrap.dedent(
        #     f"""
        #     {persona_line}

        #     You will get three candidate sentences for the **same** scene,
        #     each emphasising either person, action, or location.  Merge them
        #     into ONE vivid sentence that retains all unique details and reads
        #     naturally.  Do NOT add new entities.

        #     Return JSON with keys:
        #       "refined_prompt": fused sentence
        #       "justification" : brief (<20 words) note on how you merged
        #     """
        # ).strip()
        system_prompt = textwrap.dedent(
            """
            Your task is to merge three versions of the same scene into one sentence.
            Keep all appearance, action, and location details. Make the result vivid, coherent, and natural-sounding.
            Do not add new people, actions, or places.

            Return JSON with:
            "refined_prompt": the fused sentence
            "justification": a brief explanation of how you combined the three inputs (e.g., merged phrasing, reordered elements, kept best details).
            """
        ).strip()

        # ---------- LLM call wrapped with JSON extraction ----------
        messages = [
            SystemMessage(content=system_prompt, source="system"),
            UserMessage(content=joined, source="user"),
        ]

        async def _ask_llm() -> str:
            assistant_msg = await self.llm_client.create(messages)
            return assistant_msg.content.strip()

        # Store the actual prompt sent to LLM
        self.last_llm_prompt = messages

        reply: Dict[str, Any] = await call_llm_until_json(
            _ask_llm, 
            on_max_attempts_reached=self.llm_client.change_seed,
            on_temperature_change=self.llm_client.change_temperature,
            on_seed_reset=self.llm_client.reset_seed_to_42
        )
        fused_text = reply["refined_prompt"]
        justification: str = reply.get("justification", "")

        # -----------------------------------------------------------
        new_prompt = copy.deepcopy(base_prompt)
        new_prompt.text = fused_text
        return AgentResult(
            prompt=new_prompt,
            justification=reply.get("justification", ""),
            llm_prompt=self.last_llm_prompt,
        )
