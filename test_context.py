from runtime.state import AgentState
from runtime.tools import Tool, ToolRegistry
from runtime.context import ContextBuilder


class Calculator(Tool):

    name = "calculator"
    description = "Adds two numbers"

    input_schema = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    }

    def execute(self, arguments):
        return arguments["a"] + arguments["b"]


state = AgentState(
    task="Calculate 10 + 20"
)

state.messages.append({
    "role": "user",
    "content": state.task
})

registry = ToolRegistry()
registry.register(Calculator())

builder = ContextBuilder()

context = builder.build(
    state,
    registry
)

for message in context:
    print(message)