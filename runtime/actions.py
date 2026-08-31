from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    raw_message: dict[str, Any]
    tool_call_id: str | None = None


@dataclass
class FinalAnswer:
    content: str