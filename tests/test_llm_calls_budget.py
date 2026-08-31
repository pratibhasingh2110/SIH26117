import pytest

from runtime.agent import Agent
from runtime.config import RuntimeConfig
from runtime.errors import TransientLLMError
from runtime.llm import LLMProvider
from runtime.runtime import AgentRuntime
from runtime.tools import Tool


class _FinalAnswerLLM(LLMProvider):
    """LLM that returns a final answer on every call."""

    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        return {
            "message": {
                "role": "assistant",
                "content": f"Done {self.calls}."
            }
        }


class _AlwaysToolLLM(LLMProvider):
    """LLM that always requests a tool call, never a final answer."""

    def __init__(self):
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
                            "name": "noop",
                            "arguments": {}
                        }
                    }
                ]
            }
        }


class _ToolThenDoneLLM(LLMProvider):
    """LLM that calls a tool N rounds, then returns a final answer."""

    def __init__(self, n_tool_rounds: int):
        self.n_tool_rounds = n_tool_rounds
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        if self.calls <= self.n_tool_rounds:
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


class _FlakyLLM(LLMProvider):
    """Fails transiently n times, then returns a final answer."""

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise TransientLLMError("transient boom")
        return {
            "message": {
                "role": "assistant",
                "content": "Recovered."
            }
        }


class _Noop(Tool):
    name = "noop"
    description = "does nothing"
    input_schema = {"type": "object"}

    def execute(self, arguments):
        return "ok"


class _SlowLLM(LLMProvider):
    """LLM that sleeps before responding."""

    def __init__(self, delay: float):
        self.delay = delay

    def generate(self, messages, tools=None):
        import time
        time.sleep(self.delay)
        return {
            "message": {
                "role": "assistant",
                "content": "Done."
            }
        }


def _make_agent(llm, tools=None):
    return Agent(
        name="BudgetAgent",
        instructions="instructions",
        llm=llm,
        tools=tools or [_Noop()],
    )


# ------------------------------------------------------------------
# 1. Default max_llm_calls=None
# ------------------------------------------------------------------

def test_default_max_llm_calls_none():
    cfg = RuntimeConfig()
    assert cfg.max_llm_calls is None

    runtime = AgentRuntime()
    assert runtime.config.max_llm_calls is None

    runtime = AgentRuntime(max_steps=5)
    assert runtime.config.max_llm_calls is None


# ------------------------------------------------------------------
# 2. Custom max_llm_calls
# ------------------------------------------------------------------

def test_custom_max_llm_calls_config_object():
    cfg = RuntimeConfig(max_steps=10, max_llm_calls=3)
    runtime = AgentRuntime(config=cfg)
    assert runtime.config.max_llm_calls == 3


# ------------------------------------------------------------------
# 3. One LLM call
# ------------------------------------------------------------------

def test_one_llm_call_completes():
    llm = _FinalAnswerLLM()
    cfg = RuntimeConfig(max_steps=10, max_llm_calls=3)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    assert state.status == "completed"
    assert llm.calls == 1
    assert runtime._llm_call_count == 1


# ------------------------------------------------------------------
# 4. Multiple LLM calls
# ------------------------------------------------------------------

def test_multiple_llm_calls_completes():
    llm = _ToolThenDoneLLM(n_tool_rounds=2)
    cfg = RuntimeConfig(max_steps=10, max_llm_calls=5)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    assert state.status == "completed"
    assert state.result == "Done."
    assert llm.calls == 3
    assert runtime._llm_call_count == 3
    assert state.step == 3


# ------------------------------------------------------------------
# 5. Budget exceeded
# ------------------------------------------------------------------

def test_budget_exceeded_stops():
    llm = _ToolThenDoneLLM(n_tool_rounds=5)
    cfg = RuntimeConfig(max_steps=100, max_llm_calls=2)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    assert state.status == "max_llm_calls_exceeded"
    assert state.result == "Agent exceeded maximum LLM calls."
    assert runtime._llm_call_count == 2
    assert llm.calls == 2


def test_budget_exceeded_event():
    llm = _ToolThenDoneLLM(n_tool_rounds=5)
    cfg = RuntimeConfig(max_steps=100, max_llm_calls=2)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    events = runtime.recorder.get_events()
    stopped = [e for e in events if e.type == "AgentStopped"]
    assert len(stopped) == 1
    assert stopped[0].data["reason"] == "max_llm_calls_exceeded"
    assert stopped[0].data["max_llm_calls"] == 2
    assert stopped[0].data["llm_call_count"] == 2
    assert stopped[0].data["steps"] == state.step


def test_budget_exceeded_no_full_answer():
    llm = _ToolThenDoneLLM(n_tool_rounds=5)
    cfg = RuntimeConfig(max_steps=100, max_llm_calls=1)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    assert state.status == "max_llm_calls_exceeded"
    assert state.result != "Done."


# ------------------------------------------------------------------
# 6. Exact-boundary behavior
# ------------------------------------------------------------------

def test_exact_boundary_completes():
    llm = _ToolThenDoneLLM(n_tool_rounds=2)
    cfg = RuntimeConfig(max_steps=100, max_llm_calls=3)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    # 3 LLM calls is exactly the budget: completes.
    assert state.status == "completed"
    assert runtime._llm_call_count == 3


def test_exact_boundary_exceeds_when_one_over():
    llm = _ToolThenDoneLLM(n_tool_rounds=4)
    cfg = RuntimeConfig(max_steps=100, max_llm_calls=3)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    assert state.status == "max_llm_calls_exceeded"
    assert runtime._llm_call_count == 3


# ------------------------------------------------------------------
# 7. Retries do not increase logical LLM-call count
# ------------------------------------------------------------------

def test_retries_do_not_inflate_llm_call_count():
    llm = _FlakyLLM(failures=2)
    cfg = RuntimeConfig(max_steps=10, max_retries=5, max_llm_calls=3)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    assert state.status == "completed"
    assert llm.calls == 3  # 1 initial + 2 retries
    assert runtime._llm_call_count == 1  # one logical call
    assert state.step == 1


# ------------------------------------------------------------------
# 8. max_steps interaction
# ------------------------------------------------------------------

def test_max_steps_interaction():
    llm = _AlwaysToolLLM()
    cfg = RuntimeConfig(max_steps=3, max_llm_calls=100)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    assert state.status == "max_steps_exceeded"
    assert state.step == 4


def test_llm_budget_hits_before_max_steps():
    llm = _AlwaysToolLLM()
    cfg = RuntimeConfig(max_steps=100, max_llm_calls=2)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    assert state.status == "max_llm_calls_exceeded"
    assert state.step == 3


def test_max_steps_hits_before_llm_budget():
    llm = _AlwaysToolLLM()
    cfg = RuntimeConfig(max_steps=2, max_llm_calls=100)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    assert state.status == "max_steps_exceeded"


# ------------------------------------------------------------------
# 9. max_tool_calls interaction
# ------------------------------------------------------------------

def test_max_tool_calls_interaction():
    llm = _AlwaysToolLLM()
    cfg = RuntimeConfig(max_steps=100, max_tool_calls=1, max_llm_calls=100)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    assert state.status == "max_tool_calls_exceeded"
    assert runtime._tool_call_count == 1


def test_llm_budget_hits_before_max_tool_calls():
    llm = _AlwaysToolLLM()
    cfg = RuntimeConfig(max_steps=100, max_tool_calls=1, max_llm_calls=1)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    # First LLM call uses the tool; second would be blocked by LLM budget.
    assert state.status == "max_llm_calls_exceeded"
    assert runtime._tool_call_count == 1


# ------------------------------------------------------------------
# 10. Timeout interaction
# ------------------------------------------------------------------

def test_timeout_interaction():
    cfg = RuntimeConfig(max_steps=100, timeout_seconds=0.1, max_llm_calls=100)
    agent = _make_agent(_SlowLLM(delay=5.0))
    runtime = AgentRuntime(config=cfg)

    state = runtime.run(agent, "go")

    assert state.status == "timeout_exceeded"


# ------------------------------------------------------------------
# 11. Retry interaction with budget
# ------------------------------------------------------------------

def test_retry_consumes_less_than_budget():
    llm = _FlakyLLM(failures=1)
    cfg = RuntimeConfig(max_steps=10, max_retries=2, max_llm_calls=1)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    # One logical call (with internal retries) fits within budget of 1.
    assert state.status == "completed"
    assert runtime._llm_call_count == 1


def test_retry_exhaustion_respects_budget():
    llm = _FlakyLLM(failures=5)
    cfg = RuntimeConfig(max_steps=10, max_retries=1, max_llm_calls=3)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    # Retries exhausted within the single logical call → failed, not a budget stop.
    assert state.status == "failed"


# ------------------------------------------------------------------
# 12. Normal Ollama-parallel (fake) execution remains functional
# ------------------------------------------------------------------

def test_normal_execution_unchanged_with_budget(config_e2e=None):
    llm = _ToolThenDoneLLM(n_tool_rounds=1)
    cfg = RuntimeConfig(max_steps=10, max_llm_calls=10)
    runtime = AgentRuntime(config=cfg)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    assert state.status == "completed"
    assert state.result == "Done."
    assert len(state.actions) == 1
    assert state.actions[0]["tool"] == "noop"
    assert runtime._llm_call_count == 2
