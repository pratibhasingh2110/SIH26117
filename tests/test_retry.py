import pytest

from runtime.agent import Agent
from runtime.config import RuntimeConfig
from runtime.errors import (
    LLMError,
    ToolExecutionError,
    ToolNotFoundError,
    TransientLLMError,
    TransientToolExecutionError,
)
from runtime.llm import LLMProvider
from runtime.runtime import AgentRuntime
from runtime.tools import Tool


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


class _HardFailLLM(LLMProvider):
    """Always raises a non-retryable LLM error."""

    def generate(self, messages, tools=None):
        raise LLMError("permanent boom")


class _FlakyTool(Tool):
    name = "flaky"
    description = "fails transiently then succeeds"
    input_schema = {"type": "object"}

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def execute(self, arguments):
        self.calls += 1
        if self.calls <= self.failures:
            raise TransientToolExecutionError("tool transient")
        return "recovered"


class _AlwaysFailTransientTool(Tool):
    name = "flaky"
    description = "always fails transiently"
    input_schema = {"type": "object"}

    def execute(self, arguments):
        raise TransientToolExecutionError("always transient")


class _Noop(Tool):
    name = "noop"
    description = "does nothing"
    input_schema = {"type": "object"}

    def execute(self, arguments):
        return "ok"


class _UnknownToolLLM(LLMProvider):
    name = "unknown_tool"

    def generate(self, messages, tools=None):
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "does_not_exist",
                            "arguments": {}
                        }
                    }
                ]
            }
        }


class _InvalidArgsToolLLM(LLMProvider):
    def generate(self, messages, tools=None):
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "noop",
                            "arguments": {"bad": "type"}
                        }
                    }
                ]
            }
        }


class _FinalAnswerLLM(LLMProvider):
    def generate(self, messages, tools=None):
        return {
            "message": {
                "role": "assistant",
                "content": "Done."
            }
        }


def _make_agent(llm, tools=None):
    return Agent(
        name="RetryAgent",
        instructions="instructions",
        llm=llm,
        tools=tools or [_Noop()],
    )


# ------------------------------------------------------------------
# 1. Retries disabled by default
# ------------------------------------------------------------------

def test_max_retries_default_zero():
    runtime = AgentRuntime()
    assert runtime.config.max_retries == 0


def test_max_retries_config_default_object():
    cfg = RuntimeConfig()
    assert cfg.max_retries == 0


# ------------------------------------------------------------------
# 2. Transient LLM failure → retry → success
# ------------------------------------------------------------------

def test_transient_llm_failure_retries_then_success():
    llm = _FlakyLLM(failures=2)
    agent = _make_agent(llm)
    cfg = RuntimeConfig(max_retries=3)
    runtime = AgentRuntime(config=cfg)

    state = runtime.run(agent, "go")

    assert state.status == "completed"
    assert state.result == "Recovered."
    assert llm.calls == 3

    events = runtime.recorder.get_events()
    retries = [e for e in events if e.type == "LLMRetry"]
    assert len(retries) == 2
    assert retries[0].data["attempt"] == 1
    assert retries[0].data["max_retries"] == 3
    assert "transient" in retries[0].data["error"]


def test_retry_does_not_increment_steps_or_llm_success_count():
    llm = _FlakyLLM(failures=2)
    agent = _make_agent(llm)
    cfg = RuntimeConfig(max_retries=3)
    runtime = AgentRuntime(config=cfg)

    state = runtime.run(agent, "go")

    # Retries happen within step 1; final answer at step 1.
    assert state.step == 1


# ------------------------------------------------------------------
# 3. Transient tool failure → retry → success
# ------------------------------------------------------------------

class _FlakyToolLLM(LLMProvider):
    """Calls flaky tool, then returns a final answer."""

    def __init__(self):
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
                                "name": "flaky",
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


def test_transient_tool_failure_retries_then_success():
    tool = _FlakyTool(failures=2)
    llm = _FlakyToolLLM()
    agent = _make_agent(llm, tools=[tool])
    cfg = RuntimeConfig(max_retries=4)
    runtime = AgentRuntime(config=cfg)

    state = runtime.run(agent, "go")

    assert state.status == "completed"
    assert state.result == "Done."
    assert tool.calls == 3

    events = runtime.recorder.get_events()
    retries = [e for e in events if e.type == "ToolRetry"]
    assert len(retries) == 2
    assert retries[0].data["tool"] == "flaky"
    assert retries[0].data["max_retries"] == 4
    assert "transient" in retries[0].data["error"]


def test_tool_retry_does_not_inflate_tool_call_count():
    tool = _FlakyTool(failures=2)
    llm = _FlakyToolLLM()
    agent = _make_agent(llm, tools=[tool])
    cfg = RuntimeConfig(max_retries=4)
    runtime = AgentRuntime(config=cfg)

    state = runtime.run(agent, "go")

    # One logical tool call succeeded; retries shouldn't inflate the count.
    assert runtime._tool_call_count == 1


# ------------------------------------------------------------------
# 4. Retry limit exhausted
# ------------------------------------------------------------------

def test_llm_retry_limit_exhausted_fails():
    llm = _FlakyLLM(failures=5)
    agent = _make_agent(llm)
    cfg = RuntimeConfig(max_retries=2)
    runtime = AgentRuntime(config=cfg)

    state = runtime.run(agent, "go")

    assert state.status == "failed"
    assert llm.calls == 3  # 1 initial + 2 retries

    events = runtime.recorder.get_events()
    retries = [e for e in events if e.type == "LLMRetry"]
    failed = [e for e in events if e.type == "LLMRetryFailed"]
    assert len(retries) == 2
    assert len(failed) == 1
    assert failed[0].data["attempt"] == 3
    assert failed[0].data["max_retries"] == 2


def test_tool_retry_limit_exhausted_fails():
    tool = _AlwaysFailTransientTool()
    llm = _FlakyToolLLM()
    agent = _make_agent(llm, tools=[tool])
    cfg = RuntimeConfig(max_retries=2)
    runtime = AgentRuntime(config=cfg)

    state = runtime.run(agent, "go")

    assert state.status == "completed"  # tool failure recorded as observation; agent continues
    events = runtime.recorder.get_events()
    tool_retries = [e for e in events if e.type == "ToolRetry"]
    tool_failed = [e for e in events if e.type == "ToolExecutionFailed"]
    retry_failed = [e for e in events if e.type == "ToolRetryFailed"]
    assert len(tool_retries) == 2
    assert len(retry_failed) == 1
    assert len(tool_failed) == 1
    assert retry_failed[0].data["attempt"] == 3
    assert retry_failed[0].data["max_retries"] == 2


# ------------------------------------------------------------------
# 5. Non-retryable errors are not retried
# ------------------------------------------------------------------

def test_non_retryable_llm_error_not_retried():
    llm = _HardFailLLM()
    agent = _make_agent(llm)
    cfg = RuntimeConfig(max_retries=3)
    runtime = AgentRuntime(config=cfg)

    state = runtime.run(agent, "go")

    assert state.status == "failed"
    events = runtime.recorder.get_events()
    assert not [e for e in events if e.type == "LLMRetry"]
    failed = [e for e in events if e.type == "LLMCallFailed"]
    assert len(failed) == 1


def test_unknown_tool_not_retried():
    llm = _UnknownToolLLM()
    agent = _make_agent(llm, tools=[_Noop()])
    cfg = RuntimeConfig(max_steps=1, max_retries=3)
    runtime = AgentRuntime(config=cfg)

    state = runtime.run(agent, "go")

    events = runtime.recorder.get_events()
    retries = [e for e in events if e.type == "ToolRetry"]
    assert not retries
    failed = [e for e in events if e.type == "ToolExecutionFailed"]
    assert len(failed) == 1
    assert "does_not_exist" in failed[0].data["error"]


# ------------------------------------------------------------------
# 6. Timeout is not retried
# ------------------------------------------------------------------

class _SlowLLM(LLMProvider):
    def generate(self, messages, tools=None):
        raise TransientLLMError("slow transient not really")


def test_timeout_not_retried():
    import signal
    import time

    class _SlowBlockingLLM(LLMProvider):
        def generate(self, messages, tools=None):
            time.sleep(5.0)
            return {"message": {"role": "assistant", "content": "late"}}

    agent = _make_agent(_SlowBlockingLLM())
    cfg = RuntimeConfig(max_retries=3, timeout_seconds=0.1)
    runtime = AgentRuntime(config=cfg)

    state = runtime.run(agent, "go")

    assert state.status == "timeout_exceeded"
    events = runtime.recorder.get_events()
    assert not [e for e in events if e.type == "LLMRetry"]


# ------------------------------------------------------------------
# 7. max_steps is not retried
# ------------------------------------------------------------------

def test_max_steps_still_works_with_retries_enabled():
    class _AlwaysToolLLM(LLMProvider):
        def generate(self, messages, tools=None):
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "noop", "arguments": {}}}
                    ]
                }
            }

    agent = _make_agent(_AlwaysToolLLM())
    cfg = RuntimeConfig(max_steps=2, max_retries=3)
    runtime = AgentRuntime(config=cfg)

    state = runtime.run(agent, "go")

    assert state.status == "max_steps_exceeded"
    assert state.step == 3


# ------------------------------------------------------------------
# 8. max_tool_calls is not retried / not inflated
# ------------------------------------------------------------------

def test_max_tool_calls_still_works_with_retries_enabled():
    class _AlwaysToolLLM(LLMProvider):
        def generate(self, messages, tools=None):
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "noop", "arguments": {}}}
                    ]
                }
            }

    agent = _make_agent(_AlwaysToolLLM())
    cfg = RuntimeConfig(max_steps=100, max_tool_calls=2, max_retries=3)
    runtime = AgentRuntime(config=cfg)

    state = runtime.run(agent, "go")

    assert state.status == "max_tool_calls_exceeded"
    assert runtime._tool_call_count == 2


# ------------------------------------------------------------------
# 9. Retry events carry structured metadata
# ------------------------------------------------------------------

def test_llm_retry_event_metadata():
    llm = _FlakyLLM(failures=1)
    agent = _make_agent(llm)
    cfg = RuntimeConfig(max_retries=2)
    runtime = AgentRuntime(config=cfg)

    runtime.run(agent, "go")

    events = runtime.recorder.get_events()
    retries = [e for e in events if e.type == "LLMRetry"]
    assert len(retries) == 1
    assert set(["attempt", "max_retries", "error", "step"]).issubset(
        retries[0].data.keys()
    )
    assert retries[0].data["step"] == 1


# ------------------------------------------------------------------
# 10. Existing successful execution remains unchanged with retries
# ------------------------------------------------------------------

def test_normal_execution_unchanged_with_retries_enabled(math_agent):
    cfg = RuntimeConfig(max_steps=10, max_retries=5)
    runtime = AgentRuntime(config=cfg)
    state = runtime.run(
        agent=math_agent,
        task="Calculate 10 + 20"
    )

    assert state.status == "completed"
    assert state.result == "The answer is 30."
    assert state.step == 2
    assert runtime._tool_call_count == 1
