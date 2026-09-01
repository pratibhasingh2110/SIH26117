import pytest

from runtime.agent import Agent
from runtime.tools import Tool
from runtime.llm import LLMProvider


class Calculator(Tool):

    name = "calculator"
    description = (
        "Performs arithmetic operations on two numbers. "
        "Supported operations: add (+), subtract (-), multiply (*), divide (/). "
        "When 'operation' is omitted, defaults to addition."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
            "operation": {
                "type": "string",
                "enum": ["add", "+", "subtract", "-", "multiply", "*", "divide", "/"],
                "description": (
                    "The arithmetic operation to perform. "
                    "Defaults to 'add' when omitted."
                ),
            },
        },
        "required": ["a", "b"],
    }

    _OPERATIONS = {
        "add": lambda a, b: a + b,
        "+": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "-": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "*": lambda a, b: a * b,
        "divide": lambda a, b: a / b,
        "/": lambda a, b: a / b,
    }

    def execute(self, arguments):
        a = arguments["a"]
        b = arguments["b"]
        operation = arguments.get("operation") or "add"
        key = operation.lower() if isinstance(operation, str) else ""

        if key not in self._OPERATIONS:
            raise ValueError(
                f"Unsupported operation: {operation!r}. "
                "Supported operations: add, subtract, multiply, divide."
            )

        if key in ("divide", "/") and b == 0:
            raise ValueError("Division by zero is not allowed.")

        return self._OPERATIONS[key](a, b)


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
