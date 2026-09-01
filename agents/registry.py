from runtime.agent import Agent
from runtime.tools import Tool
from runtime.tools import ToolRegistry
from providers.ollama import OllamaProvider

_CALCULATOR_NAME = "calculator"
_CALCULATOR_DESCRIPTION = (
    "Performs arithmetic operations on two numbers. "
    "Supported operations: add (+), subtract (-), multiply (*), divide (/). "
    "When 'operation' is omitted, defaults to addition."
)
_CALCULATOR_SCHEMA = {
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

_CALCULATOR_OPERATIONS = {
    "add": lambda a, b: a + b,
    "+": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "-": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "*": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "/": lambda a, b: a / b,
}


class Calculator(Tool):
    name = _CALCULATOR_NAME
    description = _CALCULATOR_DESCRIPTION
    input_schema = _CALCULATOR_SCHEMA

    def execute(self, arguments):
        a = arguments["a"]
        b = arguments["b"]
        operation = arguments.get("operation") or "add"
        key = operation.lower() if isinstance(operation, str) else ""

        if key not in _CALCULATOR_OPERATIONS:
            raise ValueError(
                f"Unsupported operation: {operation!r}. "
                "Supported operations: add, subtract, multiply, divide."
            )

        if key in ("divide", "/") and b == 0:
            raise ValueError("Division by zero is not allowed.")

        return _CALCULATOR_OPERATIONS[key](a, b)


_DEFAULT_MODEL = "qwen2.5:7b"
_DEFAULT_BASE_URL = "http://localhost:11434"


def build_math_agent(
    model: str = _DEFAULT_MODEL,
    base_url: str = _DEFAULT_BASE_URL,
) -> Agent:
    return Agent(
        name="MathAgent",
        description="Handles arithmetic and mathematical calculation tasks.",
        instructions=(
            "You are a math assistant. Use the calculator tool "
            "to perform arithmetic calculations."
        ),
        llm=OllamaProvider(model=model, base_url=base_url),
        tools=[Calculator()],
    )


def build_research_agent(
    model: str = _DEFAULT_MODEL,
    base_url: str = _DEFAULT_BASE_URL,
) -> Agent:
    return Agent(
        name="ResearchAgent",
        description="Handles information lookup and research-oriented tasks.",
        instructions=(
            "You are a research assistant. Gather and summarize "
            "information for research-oriented tasks."
        ),
        llm=OllamaProvider(model=model, base_url=base_url),
        tools=[],
    )


def build_general_agent(
    model: str = _DEFAULT_MODEL,
    base_url: str = _DEFAULT_BASE_URL,
) -> Agent:
    return Agent(
        name="GeneralAgent",
        description=(
            "Handles general tasks that do not clearly belong "
            "to another specialist."
        ),
        instructions=(
            "You are a general assistant. Answer general-purpose "
            "questions directly."
        ),
        llm=OllamaProvider(model=model, base_url=base_url),
        tools=[],
    )


def build_agents(
    model: str = _DEFAULT_MODEL,
    base_url: str = _DEFAULT_BASE_URL,
) -> list[Agent]:
    return [
        build_math_agent(model=model, base_url=base_url),
        build_research_agent(model=model, base_url=base_url),
        build_general_agent(model=model, base_url=base_url),
    ]
