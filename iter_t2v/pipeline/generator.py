# /workspace/t2v_self/iter_t2v/pipeline/generator.py
from __future__ import annotations
from typing import List, Dict, Any, Callable

from messages.messages_types import Prompt
from messages.agent_result import AgentResult
from agents.person_agent import PersonAgent
from agents.action_agent import ActionAgent
from agents.location_agent import LocationAgent
from agents.single_shot_agent import SingleShotAgent
from agents.fuse_agent import FuseAgent


class PromptGenerator:
    """Return dict {final_prompt, timeline} given a Prompt."""

    def __init__(self, llm_client, generation_mode: str = "base") -> None:
        modes = {"base", "single", "parallel", "sequential"}
        if generation_mode not in modes:
            raise ValueError(f"generation_mode must be {modes}")

        self.mode = generation_mode
        self.person_agent = PersonAgent(llm_client)
        self.action_agent = ActionAgent(llm_client)
        self.location_agent = LocationAgent(llm_client)
        self.single_agent = SingleShotAgent(llm_client)
        self.fuse_agent = FuseAgent(llm_client)

    # ---------- helpers ------------------------------------------------
    async def _run_and_log(
        self,
        agent: Any,
        prompt_in: Prompt,
        timeline: List[Dict[str, str]],
    ) -> Prompt:
        before = prompt_in.text
        result: AgentResult = await agent.refine(prompt_in)

        # Capture the actual LLM prompt if the agent exposes it
        llm_prompt = getattr(result, "llm_prompt", None) or getattr(agent, "last_llm_prompt", "Not captured")
        
        # Capture the seed and temperature used for this LLM call
        seed_used = getattr(agent.llm_client, 'current_seed', None) if hasattr(agent, 'llm_client') else None
        temperature_used = getattr(agent.llm_client, 'current_temperature', None) if hasattr(agent, 'llm_client') else None
        
        # Ensure temperature is a simple number, not a function or complex object
        if temperature_used is not None and not isinstance(temperature_used, (int, float)):
            temperature_used = str(temperature_used) if not callable(temperature_used) else None

        timeline.append(
            {
                "agent_name": agent.name,
                "input_prompt": before,
                "refined_prompt": result.prompt.text,
                "justification": result.justification,
                "llm_prompt": llm_prompt,
                "seed_used": seed_used,
                "temperature_used": temperature_used,
            }
        )
        return result.prompt

    # ---------- main callable ------------------------------------------
    async def __call__(self, prompt: Prompt) -> Dict[str, Any]:
        orig = prompt.text
        timeline: List[Dict[str, str]] = []

        if self.mode == "base":
            timeline.append(
                {
                    "agent_name": "Base",
                    "input_prompt": orig,
                    "refined_prompt": orig,
                    "justification": "none",
                    "llm_prompt": "No LLM call (base mode)",
                    "seed_used": None,
                    "temperature_used": None,
                }
            )
            return {
                "original_prompt": orig,
                "final_prompt": orig,
                "timeline": timeline,
            }

        if self.mode == "single":
            refined = await self._run_and_log(self.single_agent, prompt, timeline)
            return {"original_prompt": orig,
                    "final_prompt": refined.text,
                    "timeline": timeline}

        if self.mode == "parallel":
            # run the three slice-specific agents on the ORIGINAL prompt
            p = await self._run_and_log(self.person_agent, prompt, timeline)
            a = await self._run_and_log(self.action_agent, prompt, timeline)
            l = await self._run_and_log(self.location_agent, prompt, timeline)

            # ---------- FuseAgent (manual logging because it needs a list) ----
            before_texts = " | ".join([p.text, a.text, l.text])

            fused_result = await self.fuse_agent.refine([p, a, l])  # returns AgentResult

            # Capture LLM prompt for fuse agent
            fuse_llm_prompt = getattr(fused_result, "llm_prompt", None) or getattr(self.fuse_agent, "last_llm_prompt", "Not captured")
            
            # Capture seed and temperature for fuse agent
            fuse_seed_used = getattr(self.fuse_agent.llm_client, 'current_seed', None) if hasattr(self.fuse_agent, 'llm_client') else None
            fuse_temperature_used = getattr(self.fuse_agent.llm_client, 'current_temperature', None) if hasattr(self.fuse_agent, 'llm_client') else None
            
            # Ensure temperature is a simple number, not a function or complex object
            if fuse_temperature_used is not None and not isinstance(fuse_temperature_used, (int, float)):
                fuse_temperature_used = str(fuse_temperature_used) if not callable(fuse_temperature_used) else None

            timeline.append(
                {
                    "agent_name": self.fuse_agent.name,
                    "input_prompt": before_texts,
                    "refined_prompt": fused_result.prompt.text,
                    "justification": fused_result.justification,
                    "llm_prompt": fuse_llm_prompt,
                    "seed_used": fuse_seed_used,
                    "temperature_used": fuse_temperature_used,
                }
            )

            return {
                "original_prompt": orig,
                "final_prompt": fused_result.prompt.text,
                "timeline": timeline,
            }

        # sequential
        p = await self._run_and_log(self.person_agent, prompt, timeline)
        a = await self._run_and_log(self.action_agent, p, timeline)
        l = await self._run_and_log(self.location_agent, a, timeline)
        return {"original_prompt": orig,
                "final_prompt": l.text,
                "timeline": timeline}

