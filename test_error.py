from runtime.actions import ToolCall
from runtime.executor import ToolExecutor
from runtime.tools import Tool, ToolRegistry


class BrokenTool(Tool):

    name = "broken"
    description = "A tool that always fails"

    input_schema = {
        "type": "object"
    }

    def execute(self, arguments):
        raise ValueError("Something went wrong")


registry = ToolRegistry()
registry.register(BrokenTool())

executor = ToolExecutor(registry)

try:

    executor.execute(
        ToolCall(
            tool_name="broken",
            arguments={},
            raw_message={}
        )
    )

except Exception as error:

    print(type(error).__name__)
    print(error)