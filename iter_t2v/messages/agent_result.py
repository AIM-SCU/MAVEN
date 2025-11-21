# iter_t2v/messages/agent_result.py
from dataclasses import dataclass
from messages.messages_types import Prompt
from typing import Any

@dataclass
class AgentResult:
    prompt: Prompt          # mutated copy
    justification: str
    llm_prompt: Any = None  # Store the actual LLM prompt/messages
