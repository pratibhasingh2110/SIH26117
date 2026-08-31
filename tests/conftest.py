import pytest

from runtime.agent import Agent
from runtime.tools import Tool
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


@pytest.fixture
def calculator_tool():
    return Calculator()


@pytest.fixture
def math_agent(calculator_tool):
    return Agent(
        name="MathAgent",
        instructions="You are a math assistant.",
        llm=_FakeMathLLM(),
        tools=[calculator_tool]
    )


class _FakeMathLLM(LLMProvider):

    def __init__(self):
        self.called = False

    def generate(self, messages, tools=None):

        if not self.called:
            self.called = True

            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "calculator",
                                "arguments": {"a": 10, "b": 20}
                            }
                        }
                    ]
                }
            }

        return {
            "message": {
                "role": "assistant",
                "content": "The answer is 30."
            }
        }
