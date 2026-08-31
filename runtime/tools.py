from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> Any:
        pass

class ToolRegistry:

    def __init__(self):
        self._tools = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())