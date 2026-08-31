from dataclasses import dataclass, field
from typing import Any

from runtime.llm import LLMProvider
from runtime.tools import Tool


@dataclass
class Agent:
    name: str
    instructions: str
    llm: LLMProvider
    tools: list[Tool] = field(default_factory=list)
    description: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)