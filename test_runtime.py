from runtime.actions import ToolCall, FinalAnswer
from runtime.runtime import AgentRuntime
from runtime.tools import Tool, ToolRegistry
from runtime.llm import LLMProvider


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


class FakeLLM(LLMProvider):

    def __init__(self):
        self.called = False

    def generate(self, messages, tools=None):

        if not self.called:
            self.called = True

            return ToolCall(
                tool_name="calculator",
                arguments={
                    "a": 10,
                    "b": 20
                }
            )

        return FinalAnswer(
            content="The answer is 30."
        )


registry = ToolRegistry()
registry.register(Calculator())

llm = FakeLLM()

runtime = AgentRuntime(
    llm=llm,
    tool_registry=registry
)

state = runtime.run(
    "Calculate 10 + 20"
)

print("Status:", state.status)
print("Steps:", state.step)
print("Observations:", state.observations)
print("Result:", state.result)