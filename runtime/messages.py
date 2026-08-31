from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str
    content: str | None = None

    name: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None