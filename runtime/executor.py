from typing import Any

from runtime.actions import ToolCall
from runtime.errors import ToolExecutionError
from runtime.tools import ToolRegistry


class ToolExecutor:

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def execute(self, tool_call: ToolCall) -> Any:

        try:
            tool = self.tool_registry.get(
                tool_call.tool_name
            )

            return tool.execute(
                tool_call.arguments
            )

        except Exception as error:
            wrapped = ToolExecutionError(
                f"Tool '{tool_call.tool_name}' failed: {error}"
            )
            if getattr(error, "retryable", False):
                wrapped.retryable = True
            raise wrapped from error