"""Deterministic tests for the demo/API layer.

All tests use fake LLM providers, so no Ollama instance is required.
These tests verify the {task -> router -> runtime -> tool -> result} path
and the response/error serialization contract.
"""

import pytest

from runtime import Agent, LLMProvider, Tool, RuntimeConfig, AgentRuntime
from router import AgentRouter
from demo.runner import (
    DemoRuntime,
    RoutingError,
    ExecutionError,
    run_task,
    serialize_run_result,
    serialize_error,
)


class Calculator(Tool):
    name = "calculator"
    description = "Adds two numbers"
    input_schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }

    def execute(self, arguments):
        return arguments["a"] + arguments["b"]


class MathLLM(LLMProvider):
    """Routes to MathAgent on first call, then executes calculator + answers."""

    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"selected_agent": "MathAgent", '
                        '"reason": "task is arithmetic"}'
                    ),
                }
            }
        if self.calls == 2:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "calculator",
                                "arguments": {"a": 25, "b": 17},
                            }
                        }
                    ],
                }
            }
        return {"message": {"role": "assistant", "content": "42"}}


class FailingMathLLM(LLMProvider):
    """Routes to MathAgent, then produces a tool call, but never answers
    (simulating a runtime failure)."""

    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"selected_agent": "MathAgent", '
                        '"reason": "task is arithmetic"}'
                    ),
                }
            }
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculator",
                            "arguments": {"a": 1, "b": 2},
                        }
                    }
                ],
            }
        }


class BrokenRoutingLLM(LLMProvider):
    """Routing LLM that returns garbage so routing fails."""

    def generate(self, messages, tools=None):
        return {"message": {"role": "assistant", "content": "not json at all"}}


class DummyLLM(LLMProvider):
    """Concrete no-op LLM for non-selected agents."""

    def generate(self, messages, tools=None):
        return {"message": {"role": "assistant", "content": ""}}


class BrokenRuntimeLLM(LLMProvider):
    """Routes fine, then the model errors on execution (runtime failure)."""

    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"selected_agent": "MathAgent", '
                        '"reason": "task is arithmetic"}'
                    ),
                }
            }
        raise RuntimeError("model exploded")


def _make_agents(routing_llm, runtime_llm=None):
    return [
        Agent(
            name="MathAgent",
            description="Handles arithmetic and mathematical calculation tasks.",
            instructions="You are a math assistant. Use the calculator tool.",
            llm=runtime_llm or routing_llm,
            tools=[Calculator()],
        ),
        Agent(
            name="ResearchAgent",
            description="Handles research tasks.",
            instructions="You are a research assistant.",
            llm=DummyLLM(),
        ),
        Agent(
            name="GeneralAgent",
            description="Handles general tasks.",
            instructions="You are a general assistant.",
            llm=DummyLLM(),
        ),
    ]


def _math_demo():
    llm = MathLLM()
    agents = _make_agents(llm, runtime_llm=llm)
    return DemoRuntime(agents)


# --- 1. valid /run request (golden path) -----------------------------------

def test_valid_run_request_golden_path():
    demo = _math_demo()
    result = demo.run("Use the calculator to calculate 25 + 17.")

    assert result["status"] == "completed"
    assert result["result"] == "42"
    assert result["agent"] == "MathAgent"
    assert result["steps"] == 2
    assert result["execution_id"].startswith("exec_")
    assert isinstance(result["events"], list)


# --- 2. routing integration -------------------------------------------------

def test_routing_integration_selects_math_agent():
    demo = _math_demo()
    result = demo.run("Use the calculator to calculate 25 + 17.")
    assert result["agent"] == "MathAgent"
    assert result["routing_reason"] == "task is arithmetic"


def test_routing_records_routing_events():
    demo = _math_demo()
    demo.run("Use the calculator to calculate 25 + 17.")
    types = [e.type for e in demo.recorder.get_events()]
    assert "AgentRoutingStarted" in types
    assert "AgentRoutingCompleted" in types


def test_routing_failure_raises_routing_error():
    llm = BrokenRoutingLLM()
    agents = _make_agents(llm, runtime_llm=llm)
    demo = DemoRuntime(agents)

    with pytest.raises(RoutingError):
        demo.run("some task")


# --- 3. runtime integration -------------------------------------------------

def test_runtime_integration_executes_tool():
    demo = _math_demo()
    result = demo.run("Use the calculator to calculate 25 + 17.")

    types = [e["type"] for e in result["events"]]
    assert "ToolCall" in types
    assert "ToolResult" in types

    tool_call = next(e for e in result["events"] if e["type"] == "ToolCall")
    assert tool_call["tool"] == "calculator"
    assert tool_call["arguments"] == {"a": 25, "b": 17}


def test_runtime_integration_lifecycle_order():
    demo = _math_demo()
    result = demo.run("Use the calculator to calculate 25 + 17.")

    order = [e["type"] for e in result["events"]]
    assert (
        "AgentRoutingStarted" in order
        and "AgentRoutingCompleted" in order
        and "AgentStarted" in order
        and "StepStarted" in order
        and "LLMCallStarted" in order
        and "LLMCallCompleted" in order
        and "ToolCall" in order
        and "ToolResult" in order
        and "AgentCompleted" in order
    )
    assert order.index("AgentStarted") < order.index("AgentCompleted")


def test_runtime_budget_exceeded_is_graceful():
    llm = FailingMathLLM()
    agents = _make_agents(llm, runtime_llm=llm)

    config = RuntimeConfig(max_llm_calls=2, max_retries=0)
    demo = DemoRuntime(agents, config=config)

    result = demo.run("Use the calculator to calculate 25 + 17.")
    assert result["status"] == "max_llm_calls_exceeded"


def test_runtime_failure_raises_execution_error():
    llm = BrokenRuntimeLLM()
    agents = _make_agents(llm, runtime_llm=llm)

    demo = DemoRuntime(agents)

    with pytest.raises(ExecutionError) as excinfo:
        demo.run("Use the calculator to calculate 25 + 17.")

    assert "model exploded" in str(excinfo.value)


# --- 4. response serialization ----------------------------------------------

def _completed_state():
    demo = _math_demo()
    result = demo.run("Use the calculator to calculate 25 + 17.")
    return result


def test_response_serialization_contract():
    result = _completed_state()

    assert set(result) == {
        "execution_id",
        "agent",
        "status",
        "result",
        "steps",
        "routing_reason",
        "events",
    }


def test_response_events_are_serializable():
    import json

    result = _completed_state()
    json.dumps(result)  # must not raise


def test_execution_id_is_shared_not_new():
    demo = _math_demo()
    result = demo.run("Use the calculator to calculate 25 + 17.")

    # execution identity comes through from AgentState; events carry the same id.
    event_ids = {e.get("execution_id") for e in result["events"] if "execution_id" in e}
    assert result["execution_id"] in event_ids


# --- 5. runtime failure serialization ----------------------------------------

def test_serialize_error_has_no_traceback():
    error = ExecutionError("Agent execution failed.")
    serialized = serialize_error(error=error, agent="MathAgent")

    assert serialized["status"] == "error"
    assert serialized["agent"] == "MathAgent"
    assert serialized["error_type"] == "ExecutionError"
    assert "Traceback" not in serialized["result"]
    assert "execution_id" in serialized


def test_serialize_error_events():
    from runtime.events import RuntimeEvent

    error = ExecutionError("oops")
    serialized = serialize_error(
        error=error,
        events=[RuntimeEvent(type="AgentStarted", data={"agent": "A"})],
    )
    assert serialized["events"] == [{"type": "AgentStarted", "agent": "A"}]


# --- run_task convenience -----------------------------------------------------

def test_run_task_entry_point():
    llm = MathLLM()
    agents = _make_agents(llm, runtime_llm=llm)
    result = run_task("Use the calculator to calculate 25 + 17.", agents=agents)

    assert result["status"] == "completed"
    assert result["result"] == "42"
    assert result["agent"] == "MathAgent"


def test_run_task_empty_task_raises():
    with pytest.raises(ExecutionError):
        DemoRuntime(_make_agents(MathLLM())).run("   ")


def test_run_task_forwards_model_and_base_url(monkeypatch):
    captured = {}

    class _FakeDemoRuntime:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, task):
            return {"status": "completed"}

    monkeypatch.setattr("demo.runner.DemoRuntime", _FakeDemoRuntime)

    from demo.runner import run_task as _run_task

    _run_task(
        "some task",
        model="qwen3.5:0.8b",
        base_url="http://localhost:11434",
    )

    assert captured["model"] == "qwen3.5:0.8b"
    assert captured["base_url"] == "http://localhost:11434"


def test_demo_runtime_forwards_model_and_base_url_to_agents(monkeypatch):
    captured = {}

    class _FakeRouter:
        def __init__(self, agents, recorder=None):
            self.agents_received = agents

    def _fake_build_agents(model, base_url):
        captured["model"] = model
        captured["base_url"] = base_url
        return []

    monkeypatch.setattr("demo.runner.build_agents", _fake_build_agents)
    monkeypatch.setattr("demo.runner.AgentRouter", _FakeRouter)

    DemoRuntime(model="qwen3.5:0.8b", base_url="http://localhost:11434")

    assert captured["model"] == "qwen3.5:0.8b"
    assert captured["base_url"] == "http://localhost:11434"
