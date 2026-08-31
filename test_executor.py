from runtime.actions import ToolCall
from runtime.executor import ToolExecutor
from runtime.tools import Tool, ToolRegistry


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


registry = ToolRegistry()
registry.register(Calculator())

executor = ToolExecutor(registry)

call = ToolCall(
    tool_name="calculator",
    arguments={
        "a": 10,
        "b": 20
    },
    raw_message={}
)

result = executor.execute(call)

print(result)