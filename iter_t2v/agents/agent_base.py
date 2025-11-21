# /workspace/t2v_self/iter_t2v/agents/agent_base.py
from __future__ import annotations
from abc import ABC, abstractmethod

from messages.messages_types import Prompt


class AgentBase(ABC):
    """
    All specialised agents inherit from this.  Each agent gets a Prompt,
    tweaks *only* its own dimension, and returns the modified Prompt.
    """

    def __init__(self, name: str):
        self.name = name

    # ------------------------------------------------------------------ #
    # Agents can override this to inject an LLM call, but the signature
    # MUST stay (Prompt in → Prompt out).
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def refine(self, prompt: Prompt) -> Prompt:  # noqa: D401
        """
        Refine the prompt and return a *new* Prompt object.
        """
        ...

    # Optionally: shared helper to call an LLM (sync or async)
    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Stub — fill in with your OpenAI/Ollama call.
        """
        # e.g. `response = self.llm_client.chat(system_prompt, user_prompt)`
        raise NotImplementedError("Hook up your LLM client here.")
