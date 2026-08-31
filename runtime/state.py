from dataclasses import dataclass, field
from typing import Any

from runtime.messages import Message


@dataclass
class AgentState:
    task: str

    execution_id: str = ""

    messages: list[Message] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)

    step: int = 0
    status: str = "running"
    result: Any = None