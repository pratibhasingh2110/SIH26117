import time

import pytest

from runtime.config import RuntimeConfig
from runtime.runtime import AgentRuntime
from runtime.agent import Agent
from runtime.llm import LLMProvider
from runtime.tools import Tool


class _AlwaysToolLLM(LLMProvider):
    """LLM that always requests a tool call, never a final answer."""

    def __init__(self, tool_name: str = "noop"):
        self.tool_name = tool_name
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": self.tool_name,
                            "arguments": {}
                        }
                    }
                ]
            }
        }


class _FinalAnswerLLM(LLMProvider):
    """LLM that returns a final answer on the first call."""

    def generate(self, messages, tools=None):
        return {
            "message": {
                "role": "assistant",
                "content": "Done."
            }
        }


class _Noop(Tool):
    name = "noop"
    description = "does nothing"
    input_schema = {"type": "object"}

    def execute(self, arguments):
        return "ok"


class _MultiCallLLM(LLMProvider):
    """LLM that returns N tool calls in a single response, then a final answer."""

    def __init__(self, n_tool_rounds: int):
        self.n_tool_rounds = n_tool_rounds
        self.rounds = 0

    def generate(self, messages, tools=None):
        self.rounds += 1
        if self.rounds <= self.n_tool_rounds:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "noop",
                                "arguments": {}
                            }
                        }
                    ]
                }
            }
        return {
            "message": {
                "role": "assistant",
                "content": "Done."
            }
        }


class _BatchToolLLM(LLMProvider):
    """LLM that returns multiple tool calls in one response, repeated."""

    def __init__(self, calls_per_round: int = 2):
        self.calls_per_round = calls_per_round
        self.rounds = 0

    def generate(self, messages, tools=None):
        self.rounds += 1
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "noop",
                            "arguments": {}
                        }
                    }
                    for _ in range(self.calls_per_round)
                ]
            }
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_agent(llm, tools=None):
    return Agent(
        name="TestAgent",
        instructions="You are a test agent.",
        llm=llm,
        tools=tools or [_Noop()],
    )


# ------------------------------------------------------------------
# 1. Default configuration
# ------------------------------------------------------------------

def test_default_config_max_steps():
    runtime = AgentRuntime()
    assert runtime.config.max_steps == 10
    assert runtime.config.max_tool_calls is None


def test_default_config_via_kwarg():
    runtime = AgentRuntime(max_steps=5)
    assert runtime.config.max_steps == 5
    assert runtime.config.max_tool_calls is None


def test_explicit_config_object():
    cfg = RuntimeConfig(max_steps=3, max_tool_calls=7)
    runtime = AgentRuntime(config=cfg)
    assert runtime.config.max_steps == 3
    assert runtime.config.max_tool_calls == 7


def test_config_overrides_kwarg():
    cfg = RuntimeConfig(max_steps=99)
    runtime = AgentRuntime(config=cfg, max_steps=1)
    assert runtime.config.max_steps == 99


# ------------------------------------------------------------------
# 2. Custom max_steps
# ------------------------------------------------------------------

def test_custom_max_steps_reduces_budget():
    agent = _make_agent(_AlwaysToolLLM())
    runtime = AgentRuntime(max_steps=3)
    state = runtime.run(agent, "go")

    assert state.status == "max_steps_exceeded"
    assert state.step == 4


# ------------------------------------------------------------------
# 3. max_steps enforcement
# ------------------------------------------------------------------

def test_max_steps_enforcement():
    agent = _make_agent(_AlwaysToolLLM())
    cfg = RuntimeConfig(max_steps=2)
    runtime = AgentRuntime(config=cfg)
    state = runtime.run(agent, "go")

    assert state.status == "max_steps_exceeded"
    assert state.result == "Agent exceeded maximum execution steps."
    assert state.step == 3


def test_max_steps_enforcement_event():
    agent = _make_agent(_AlwaysToolLLM())
    runtime = AgentRuntime(max_steps=2)
    state = runtime.run(agent, "go")

    events = runtime.recorder.get_events()
    stopped = [e for e in events if e.type == "AgentStopped"]
    assert len(stopped) == 1
    assert stopped[0].data["reason"] == "max_steps_exceeded"


def test_max_steps_not_exceeded_when_completed():
    agent = _make_agent(_FinalAnswerLLM())
    runtime = AgentRuntime(max_steps=5)
    state = runtime.run(agent, "go")

    assert state.status == "completed"
    assert state.result == "Done."


# ------------------------------------------------------------------
# 4. Custom max_tool_calls
# ------------------------------------------------------------------

def test_custom_max_tool_calls():
    cfg = RuntimeConfig(max_steps=100, max_tool_calls=3)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(_AlwaysToolLLM())
    state = runtime.run(agent, "go")

    assert state.status == "max_tool_calls_exceeded"


# ------------------------------------------------------------------
# 5. max_tool_calls enforcement
# ------------------------------------------------------------------

def test_max_tool_calls_enforcement():
    cfg = RuntimeConfig(max_steps=100, max_tool_calls=2)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(_AlwaysToolLLM())
    state = runtime.run(agent, "go")

    assert state.status == "max_tool_calls_exceeded"
    assert state.result == "Agent exceeded maximum tool calls."


def test_max_tool_calls_enforcement_event():
    cfg = RuntimeConfig(max_steps=100, max_tool_calls=2)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(_AlwaysToolLLM())
    state = runtime.run(agent, "go")

    events = runtime.recorder.get_events()
    stopped = [e for e in events if e.type == "AgentStopped"]
    assert len(stopped) == 1
    assert stopped[0].data["reason"] == "max_tool_calls_exceeded"
    assert stopped[0].data["tool_calls"] == 2


def test_max_tool_calls_counts_failed_executions():
    class _FailTool(Tool):
        name = "noop"
        description = "always fails"
        input_schema = {"type": "object"}

        def execute(self, arguments):
            raise ValueError("boom")

    cfg = RuntimeConfig(max_steps=100, max_tool_calls=1)
    agent = _make_agent(_AlwaysToolLLM(), tools=[_FailTool()])
    runtime = AgentRuntime(config=cfg)
    state = runtime.run(agent, "go")

    assert state.status == "max_tool_calls_exceeded"


def test_max_tool_calls_none_means_unlimited():
    cfg = RuntimeConfig(max_steps=100, max_tool_calls=None)
    agent = _make_agent(_MultiCallLLM(n_tool_rounds=5))
    runtime = AgentRuntime(config=cfg)
    state = runtime.run(agent, "go")

    assert state.status == "completed"
    assert state.result == "Done."


def test_max_tool_calls_batch_calls():
    cfg = RuntimeConfig(max_steps=100, max_tool_calls=3)
    agent = _make_agent(_BatchToolLLM(calls_per_round=2))
    runtime = AgentRuntime(config=cfg)
    state = runtime.run(agent, "go")

    assert state.status == "max_tool_calls_exceeded"
    assert runtime._tool_call_count == 3


# ------------------------------------------------------------------
# 6. Normal execution unchanged
# ------------------------------------------------------------------

def test_normal_execution_unchanged(math_agent):
    runtime = AgentRuntime(max_steps=10)

    state = runtime.run(
        agent=math_agent,
        task="Calculate 10 + 20"
    )

    assert state.status == "completed"
    assert state.result == "The answer is 30."
    assert state.step == 2
    assert len(state.actions) == 1
    assert state.actions[0]["tool"] == "calculator"
    assert len(state.observations) == 1
    assert state.observations[0]["result"] == 30


def test_normal_execution_with_config_object(math_agent):
    cfg = RuntimeConfig(max_steps=10)
    runtime = AgentRuntime(config=cfg)

    state = runtime.run(
        agent=math_agent,
        task="Calculate 10 + 20"
    )

    assert state.status == "completed"
    assert state.result == "The answer is 30."


# ------------------------------------------------------------------
# 7. Timeout configuration
# ------------------------------------------------------------------

def test_default_config_timeout_none():
    runtime = AgentRuntime()
    assert runtime.config.timeout_seconds is None


def test_config_timeout_seconds():
    cfg = RuntimeConfig(timeout_seconds=5.0)
    runtime = AgentRuntime(config=cfg)
    assert runtime.config.timeout_seconds == 5.0


# ------------------------------------------------------------------
# 8. Timeout disabled (None)
# ------------------------------------------------------------------

def test_timeout_none_means_no_timeout():
    cfg = RuntimeConfig(max_steps=100, timeout_seconds=None)
    agent = _make_agent(_AlwaysToolLLM())
    runtime = AgentRuntime(config=cfg)
    state = runtime.run(agent, "go")

    assert state.status == "max_steps_exceeded"
    events = runtime.recorder.get_events()
    stopped = [e for e in events if e.type == "AgentStopped"]
    assert stopped[0].data["reason"] == "max_steps_exceeded"


# ------------------------------------------------------------------
# 9. Normal execution before timeout
# ------------------------------------------------------------------

def test_normal_execution_before_timeout(math_agent):
    cfg = RuntimeConfig(max_steps=10, timeout_seconds=10.0)
    runtime = AgentRuntime(config=cfg)
    state = runtime.run(agent=math_agent, task="Calculate 10 + 20")

    assert state.status == "completed"
    assert state.result == "The answer is 30."


# ------------------------------------------------------------------
# 10. Timeout exceeded via SIGALRM (slow LLM)
# ------------------------------------------------------------------

class _SlowLLM(LLMProvider):
    """LLM that sleeps before responding."""

    def __init__(self, delay: float):
        self.delay = delay

    def generate(self, messages, tools=None):
        time.sleep(self.delay)
        return {
            "message": {
                "role": "assistant",
                "content": "Done."
            }
        }


class _AlwaysSlowToolLLM(LLMProvider):
    """LLM that requests a tool call and sleeps."""

    def __init__(self, delay: float = 0.5):
        self.delay = delay
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        time.sleep(self.delay)
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "noop",
                            "arguments": {}
                        }
                    }
                ]
            }
        }


class _SlowTool(Tool):
    """Tool that sleeps before returning."""

    name = "slow_tool"
    description = "sleeps"
    input_schema = {"type": "object"}

    def __init__(self, delay: float):
        self.delay = delay

    def execute(self, arguments):
        time.sleep(self.delay)
        return "ok"


class _SlowToolLLM(LLMProvider):
    """LLM that calls a slow tool, then finishes."""

    def __init__(self, tool_name: str = "slow_tool"):
        self.tool_name = tool_name
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": self.tool_name,
                                "arguments": {}
                            }
                        }
                    ]
                }
            }
        return {
            "message": {
                "role": "assistant",
                "content": "Done."
            }
        }


def test_timeout_exceeded_stops_execution():
    cfg = RuntimeConfig(max_steps=100, timeout_seconds=0.1)
    agent = _make_agent(_SlowLLM(delay=5.0))
    runtime = AgentRuntime(config=cfg)
    state = runtime.run(agent, "go")

    assert state.status == "timeout_exceeded"


def test_timeout_exceeded_result_message():
    cfg = RuntimeConfig(max_steps=100, timeout_seconds=0.1)
    agent = _make_agent(_SlowLLM(delay=5.0))
    runtime = AgentRuntime(config=cfg)
    state = runtime.run(agent, "go")

    assert "timed out" in state.result
    assert "0.1" in state.result


def test_timeout_exceeded_event():
    cfg = RuntimeConfig(max_steps=100, timeout_seconds=0.1)
    agent = _make_agent(_SlowLLM(delay=5.0))
    runtime = AgentRuntime(config=cfg)
    state = runtime.run(agent, "go")

    events = runtime.recorder.get_events()
    stopped = [e for e in events if e.type == "AgentStopped"]
    assert len(stopped) == 1
    assert stopped[0].data["reason"] == "timeout_exceeded"


def test_timeout_event_elapsed_and_timeout():
    cfg = RuntimeConfig(max_steps=100, timeout_seconds=0.2)
    agent = _make_agent(_SlowLLM(delay=5.0))
    runtime = AgentRuntime(config=cfg)
    state = runtime.run(agent, "go")

    events = runtime.recorder.get_events()
    stopped = [e for e in events if e.type == "AgentStopped"]
    assert stopped[0].data["timeout"] == 0.2
    assert isinstance(stopped[0].data["elapsed"], float)
    assert stopped[0].data["elapsed"] >= 0.1


def test_timeout_state_is_timeout_exceeded():
    cfg = RuntimeConfig(max_steps=100, timeout_seconds=0.1)
    agent = _make_agent(_SlowLLM(delay=5.0))
    runtime = AgentRuntime(config=cfg)
    state = runtime.run(agent, "go")

    assert state.status == "timeout_exceeded"
    assert state.result is not None


# ------------------------------------------------------------------
# 11. Timeout combined with tool execution
# ------------------------------------------------------------------

def test_timeout_during_tool_execution():
    cfg = RuntimeConfig(max_steps=100, timeout_seconds=0.1)
    slow_tool = _SlowTool(delay=5.0)
    agent = _make_agent(_SlowToolLLM("slow_tool"), tools=[slow_tool])
    runtime = AgentRuntime(config=cfg)
    state = runtime.run(agent, "go")

    assert state.status == "timeout_exceeded"


def test_timeout_cleanup_restores_signal():
    import signal

    cfg = RuntimeConfig(max_steps=100, timeout_seconds=0.1)
    agent = _make_agent(_SlowLLM(delay=5.0))
    runtime = AgentRuntime(config=cfg)
    runtime.run(agent, "go")

    handler = signal.getsignal(signal.SIGALRM)
    assert handler == signal.SIG_DFL


# ------------------------------------------------------------------
# 12. Existing max_steps + max_tool_calls unchanged
# ------------------------------------------------------------------

def test_max_steps_still_works_with_timeout_none():
    agent = _make_agent(_AlwaysToolLLM())
    runtime = AgentRuntime(max_steps=3)
    state = runtime.run(agent, "go")

    assert state.status == "max_steps_exceeded"
    assert state.step == 4


def test_max_tool_calls_still_works_with_timeout_none():
    cfg = RuntimeConfig(max_steps=100, max_tool_calls=2)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(_AlwaysToolLLM())
    state = runtime.run(agent, "go")

    assert state.status == "max_tool_calls_exceeded"


def test_timeout_interacts_with_max_steps():
    cfg = RuntimeConfig(max_steps=100, timeout_seconds=0.1)
    agent = _make_agent(_AlwaysSlowToolLLM(delay=0.05))
    runtime = AgentRuntime(config=cfg)
    state = runtime.run(agent, "go")

    assert state.status in ("timeout_exceeded", "max_steps_exceeded")
