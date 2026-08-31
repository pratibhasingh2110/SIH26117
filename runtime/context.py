from runtime.state import AgentState
from runtime.tools import ToolRegistry


class ContextBuilder:

    def build(
        self,
        state: AgentState,
        tool_registry: ToolRegistry
    ) -> list[dict]:

        messages = []

        for message in state.messages:

            data = {
                "role": message.role
            }

            if message.content is not None:
                data["content"] = message.content

            if message.name is not None:
                data["name"] = message.name

            if message.tool_calls:
                data["tool_calls"] = message.tool_calls

            if message.tool_call_id is not None:
                data["tool_call_id"] = message.tool_call_id

            messages.append(data)

        return messages