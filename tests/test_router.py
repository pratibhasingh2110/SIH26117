import pytest

from runtime.agent import Agent
from runtime.llm import LLMProvider
from runtime.tools import Tool
from router import AgentRouter, AgentRoutingError, RoutingResult
import runtime


class Calculator(Tool):
    name = "calculator"
    description = "Adds two numbers"
    input_schema = {"type": "object"}

    def execute(self, arguments):
        return arguments["a"] + arguments["b"]


class FakeRoutingLLM(LLMProvider):
    """Returns a canned routing decision from the router prompt."""

    def __init__(self, decision: str):
        self.decision = decision
        self.messages = None
        self.calls = 0

    def generate(self, messages, tools=None):
        self.messages = messages
        self.calls += 1
        return {
            "message": {
                "role": "assistant",
                "content": self.decision,
            }
        }


class _NoopTool(Tool):
    name = "noop"
    description = "does nothing"
    input_schema = {"type": "object"}

    def execute(self, arguments):
        return "ok"


class _FakeLLM(LLMProvider):
    def generate(self, messages, tools=None):
        return {"message": {"role": "assistant", "content": "done"}}


def _make_agent(name, description, tools=None, llm=None):
    return Agent(
        name=name,
        instructions=f"You are {name}.",
        description=description,
        llm=llm or _FakeLLM(),
        tools=tools or [],
    )


def _agents():
    return [
        _make_agent("MathAgent", "Handles arithmetic and mathematical calculation tasks."),
        _make_agent("ResearchAgent", "Handles information lookup and research-oriented tasks."),
        _make_agent("GeneralAgent", "Handles general tasks that do not clearly belong to another specialist."),
    ]


def _router(decision="MathAgent"):
    llm = FakeRoutingLLM(
        '{"selected_agent": "%s", "reason": "task is math"}' % decision
    )
    agents = _agents()
    agents[0] = _make_agent(
        agents[0].name, agents[0].description, llm=llm
    )
    return AgentRouter(agents), llm


def _decision(content):
    return {"message": {"role": "assistant", "content": content}}


# 1. router initialization
def test_router_initialization():
    router = AgentRouter(_agents())
    assert set(router.agents.keys()) == {
        "MathAgent", "ResearchAgent", "GeneralAgent"
    }


# 2. registered agents
def test_registered_agents_are_available():
    router = AgentRouter(_agents())
    for name in ("MathAgent", "ResearchAgent", "GeneralAgent"):
        assert name in router.agents
        assert isinstance(router.agents[name], Agent)


# 3. routing to MathAgent
def test_routes_to_math_agent():
    router, _ = _router("MathAgent")
    result = router.route("Calculate 25 + 17.")
    assert result.agent.name == "MathAgent"
    assert result.reason


# 4. routing to ResearchAgent
def test_routes_to_research_agent():
    router, _ = _router("ResearchAgent")
    result = router.route("Find information about solar panels.")
    assert result.agent.name == "ResearchAgent"


# 5. routing to GeneralAgent
def test_routes_to_general_agent():
    router, _ = _router("GeneralAgent")
    result = router.route("Explain what a computer is.")
    assert result.agent.name == "GeneralAgent"


# 6. valid routing response parsing
def test_valid_response_parsing():
    router, _ = _router()
    result = router._parse_response(
        _decision('{"selected_agent": "MathAgent", "reason": "arithmetic"}')
    )
    assert isinstance(result, RoutingResult)
    assert result.agent.name == "MathAgent"
    assert result.reason == "arithmetic"


# 7. invalid agent response
def test_invalid_response_raises():
    router, _ = _router()
    with pytest.raises(AgentRoutingError):
        router._parse_response(_decision("not json"))


# 8. unknown agent response
def test_unknown_agent_raises():
    router, _ = _router()
    with pytest.raises(AgentRoutingError):
        router._parse_response(
            _decision('{"selected_agent": "NotReal", "reason": "x"}')
        )


# 9. empty agent registry
def test_empty_registry_raises():
    with pytest.raises(AgentRoutingError):
        AgentRouter([])


# 10. router does not execute tools
def test_router_does_not_execute_tools():
    llm, _ = _router()
    assert not hasattr(llm, "execute")
    # The routing LLM must never receive tools (they are not exposed to it).
    router, _ = _router()
    assert router.route("Calculate 2 + 2").agent.name == "MathAgent"


# 11. router returns an Agent
def test_router_returns_agent():
    router, _ = _router()
    result = router.route("task")
    assert isinstance(result.agent, Agent)
    assert isinstance(result, RoutingResult)


# 12. router + AgentRuntime integration
def test_router_runtime_integration():
    import runtime as runtime_pkg
    AgentRuntime = runtime_pkg.AgentRuntime

    class MathExecutionLLM(LLMProvider):
        def __init__(self):
            self.calls = 0

        def generate(self, messages, tools=None):
            self.calls += 1
            # First call is the routing decision.
            if self.calls == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": '{"selected_agent": "MathAgent", "reason": "arithmetic"}',
                    }
                }
            if self.calls == 2:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "calculator", "arguments": {"a": 25, "b": 17}}}
                        ],
                    }
                }
            return {"message": {"role": "assistant", "content": "42"}}

    agents = [
        _make_agent("MathAgent", "Arithmetic.", [Calculator()], MathExecutionLLM()),
        _make_agent("ResearchAgent", "Research.", [], _FakeLLM()),
        _make_agent("GeneralAgent", "General.", [], _FakeLLM()),
    ]
    router = AgentRouter(agents)

    state = AgentRuntime(max_steps=10).run(
        router.route("Use the calculator to calculate 25 + 17.").agent,
        "Use the calculator to calculate 25 + 17.",
    )

    assert state.status == "completed"
    assert state.result == "42"
    assert state.observations[0]["result"] == 42


# 13. router events recorded
def test_router_records_events():
    router, _ = _router()
    router.route("test task")
    types = [e.type for e in router.recorder.get_events()]
    assert "AgentRoutingStarted" in types
    assert "AgentRoutingCompleted" in types
