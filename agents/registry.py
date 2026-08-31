from runtime.agent import Agent
from runtime.tools import Tool
from runtime.tools import ToolRegistry
from providers.ollama import OllamaProvider

_CALCULATOR_NAME = "calculator"
_CALCULATOR_DESCRIPTION = "Adds two numbers"
_CALCULATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "a": {"type": "number"},
        "b": {"type": "number"},
    },
    "required": ["a", "b"],
}


class Calculator(Tool):
    name = _CALCULATOR_NAME
    description = _CALCULATOR_DESCRIPTION
    input_schema = _CALCULATOR_SCHEMA

    def execute(self, arguments):
        return arguments["a"] + arguments["b"]


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
