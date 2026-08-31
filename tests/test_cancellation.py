import signal
import threading
import time

import pytest

from runtime.agent import Agent
from runtime.cancellation import CancellationToken
from runtime.config import RuntimeConfig
from runtime.errors import TransientLLMError, TransientToolExecutionError
from runtime.llm import LLMProvider
from runtime.runtime import AgentRuntime
from runtime.tools import Tool


class _Noop(Tool):
    name = "noop"
    description = "does nothing"
    input_schema = {"type": "object"}

    def execute(self, arguments):
        return "ok"


class _FinalAnswerLLM(LLMProvider):
    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        return {"message": {"role": "assistant", "content": "Done."}}


class _AlwaysToolLLM(LLMProvider):
    def __init__(self, on_generate=None):
        self.calls = 0
        self._on_generate = on_generate

    def generate(self, messages, tools=None):
        self.calls += 1
        if self._on_generate is not None:
            self._on_generate(self.calls)
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "noop", "arguments": {}}}
                ],
            }
        }


class _CancelAfterLLM:
    """On the Nth LLM generate call, request cancellation on the runtime."""

    def __init__(self, runtime, after_calls=1):
        self.runtime = runtime
        self.after_calls = after_calls

    def __call__(self, call_number):
        if call_number >= self.after_calls:
            self.runtime.cancel()


class _TransientThenCancelLLM(LLMProvider):
    """Raises transient error and requests cancellation on the first call."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            self.runtime.cancel()
            raise TransientLLMError("transient")
        return {"message": {"role": "assistant", "content": "Done."}}


class _FlakyToolLLM(LLMProvider):
    """Requests a tool call on the first round, then a final answer."""

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
                        {"function": {"name": "flaky", "arguments": {}}}
                    ],
                }
            }
        return {"message": {"role": "assistant", "content": "Done."}}


def _make_agent(llm, tools=None):
    return Agent(
        name="CancelAgent",
        instructions="instructions",
        llm=llm,
        tools=tools or [_Noop()],
    )


# ------------------------------------------------------------------
# 1. CancellationToken initial state
# ------------------------------------------------------------------

def test_token_initial_state():
    token = CancellationToken(execution_id="exec_1", created_at=time.monotonic())
    assert token.is_cancelled is False


def test_request_cancellation():
    token = CancellationToken(execution_id="exec_1", created_at=time.monotonic())
    token.request_cancellation()
    assert token.is_cancelled is True


def test_token_execution_id():
    token = CancellationToken(execution_id="exec_7", created_at=time.monotonic())
    assert token.execution_id == "exec_7"


def test_token_created_at():
    before = time.monotonic()
    token = CancellationToken(execution_id="exec_1", created_at=time.monotonic())
    after = time.monotonic()
    assert before <= token.created_at <= after


# ------------------------------------------------------------------
# 2. cancel() semantics
# ------------------------------------------------------------------

def test_cancel_no_active_execution():
    runtime = AgentRuntime()
    assert runtime.cancel() is None


def test_cancel_returns_token():
    runtime = AgentRuntime()
    exec_id, token = runtime._register_execution()
    returned = runtime.cancel()
    assert isinstance(returned, CancellationToken)
    assert returned.execution_id == exec_id
    assert returned.is_cancelled is True
    runtime._unregister_execution(exec_id)


def test_cancel_by_execution_id():
    runtime = AgentRuntime()
    exec_id, token = runtime._register_execution()
    returned = runtime.cancel(execution_id=exec_id)
    assert returned is token
    assert token.is_cancelled is True
    runtime._unregister_execution(exec_id)


def test_cancel_wrong_execution_id():
    runtime = AgentRuntime()
    exec_id, token = runtime._register_execution()
    assert runtime.cancel(execution_id="nonexistent") is None
    assert token.is_cancelled is False
    runtime._unregister_execution(exec_id)


def test_cancel_default_targets_most_recent():
    runtime = AgentRuntime()
    first_id, first_token = runtime._register_execution()
    second_id, second_token = runtime._register_execution()
    returned = runtime.cancel()
    assert returned is second_token
    assert second_token.is_cancelled is True
    assert first_token.is_cancelled is False
    runtime._unregister_execution(first_id)
    runtime._unregister_execution(second_id)


def test_cancel_idempotent():
    runtime = AgentRuntime()
    exec_id, token = runtime._register_execution()
    returned1 = runtime.cancel()
    returned2 = runtime.cancel(execution_id=exec_id)
    assert returned1 is token
    assert returned2 is token
    assert token.is_cancelled is True
    runtime._unregister_execution(exec_id)


# ------------------------------------------------------------------
# 3. Cancellation during execution (cooperative)
# ------------------------------------------------------------------

def test_cancel_at_loop_boundary():
    cancel_after = _CancelAfterLLM(runtime=None)
    runtime = AgentRuntime(max_steps=5)
    cancel_after.runtime = runtime
    agent = _make_agent(_AlwaysToolLLM(on_generate=cancel_after))

    state = runtime.run(agent, "go")

    assert state.status == "cancelled"
    assert state.step >= 1


def test_cancelled_result_message():
    cancel_after = _CancelAfterLLM(runtime=None)
    runtime = AgentRuntime(max_steps=5)
    cancel_after.runtime = runtime
    agent = _make_agent(_AlwaysToolLLM(on_generate=cancel_after))

    state = runtime.run(agent, "go")

    assert state.status == "cancelled"
    assert state.result == "Agent execution was cancelled."


def test_cancellation_requested_event():
    cancel_after = _CancelAfterLLM(runtime=None)
    runtime = AgentRuntime(max_steps=5)
    cancel_after.runtime = runtime
    agent = _make_agent(_AlwaysToolLLM(on_generate=cancel_after))

    runtime.run(agent, "go")

    events = runtime.recorder.get_events()
    requested = [e for e in events if e.type == "CancellationRequested"]
    assert len(requested) == 1
    data = requested[0].data
    assert "execution_id" in data
    assert "step" in data
    assert "llm_call_count" in data
    assert "tool_call_count" in data


def test_agent_stopped_cancel_event():
    cancel_after = _CancelAfterLLM(runtime=None)
    runtime = AgentRuntime(max_steps=5)
    cancel_after.runtime = runtime
    agent = _make_agent(_AlwaysToolLLM(on_generate=cancel_after))

    runtime.run(agent, "go")

    events = runtime.recorder.get_events()
    stopped = [e for e in events if e.type == "AgentStopped"]
    assert len(stopped) == 1
    assert stopped[0].data["reason"] == "cancelled"


def test_cancel_returns_current_state():
    cancel_after = _CancelAfterLLM(runtime=None)
    runtime = AgentRuntime(max_steps=5)
    cancel_after.runtime = runtime
    agent = _make_agent(_AlwaysToolLLM(on_generate=cancel_after))

    state = runtime.run(agent, "go")
    # The returned state is an AgentState marked cancelled.
    assert state.status == "cancelled"


# ------------------------------------------------------------------
# 4. Execution identity
# ------------------------------------------------------------------

def test_agent_state_execution_id():
    runtime = AgentRuntime(max_steps=1)
    agent = _make_agent(_AlwaysToolLLM())
    state = runtime.run(agent, "go")
    assert isinstance(state.execution_id, str)
    assert state.execution_id.startswith("exec_")
    assert state.execution_id != ""


def test_execution_id_on_events():
    runtime = AgentRuntime(max_steps=1)
    agent = _make_agent(_AlwaysToolLLM())
    state = runtime.run(agent, "go")

    events = runtime.recorder.get_events()
    assert events
    for event in events:
        assert event.data.get("execution_id") == state.execution_id


def test_two_runs_get_distinct_execution_ids():
    runtime = AgentRuntime(max_steps=1)
    agent1 = _make_agent(_AlwaysToolLLM())
    agent2 = _make_agent(_AlwaysToolLLM())
    s1 = runtime.run(agent1, "a")
    s2 = runtime.run(agent2, "b")
    assert s1.execution_id != s2.execution_id


# ------------------------------------------------------------------
# 5. Cancellation between retries
# ------------------------------------------------------------------

def test_cancel_between_llm_retries():
    runtime = AgentRuntime(config=RuntimeConfig(max_retries=3))
    llm = _TransientThenCancelLLM(runtime)
    agent = _make_agent(llm)

    state = runtime.run(agent, "go")

    assert state.status == "cancelled"
    # Cancellation observed between retries prevents a second generate call.
    assert llm.calls == 1


def test_cancel_between_tool_retries():
    class _CancellingFlakyTool(_Noop):
        name = "flaky"

        def __init__(self, runtime):
            self.runtime = runtime
            self.executions = 0

        def execute(self, arguments):
            self.executions += 1
            self.runtime.cancel()
            raise TransientToolExecutionError("transient")

    flaky = _CancellingFlakyTool(runtime=None)
    runtime = AgentRuntime(config=RuntimeConfig(max_retries=3))
    flaky.runtime = runtime
    agent = _make_agent(_FlakyToolLLM(), tools=[flaky])

    state = runtime.run(agent, "go")

    assert state.status == "cancelled"
    # Cancellation observed between retries prevents a second tool execution.
    assert flaky.executions == 1


# ------------------------------------------------------------------
# 6. Token cleanup
# ------------------------------------------------------------------

def test_token_removed_after_completion():
    runtime = AgentRuntime()
    runtime.run(_make_agent(_FinalAnswerLLM()), "go")
    assert runtime._active_executions == {}


def test_token_removed_after_cancellation():
    cancel_after = _CancelAfterLLM(runtime=None)
    runtime = AgentRuntime(max_steps=5)
    cancel_after.runtime = runtime
    runtime.run(_make_agent(_AlwaysToolLLM(on_generate=cancel_after)), "go")
    assert runtime._active_executions == {}


def test_token_removed_after_timeout():
    class _SlowLLM(LLMProvider):
        def generate(self, messages, tools=None):
            time.sleep(5.0)
            return {"message": {"role": "assistant", "content": "late"}}

    runtime = AgentRuntime(config=RuntimeConfig(timeout_seconds=0.1))
    runtime.run(_make_agent(_SlowLLM()), "go")
    assert runtime._active_executions == {}


def test_token_removed_after_failure():
    class _HardFailLLM(LLMProvider):
        def generate(self, messages, tools=None):
            raise ValueError("boom")

    runtime = AgentRuntime()
    runtime.run(_make_agent(_HardFailLLM()), "go")
    assert runtime._active_executions == {}


# ------------------------------------------------------------------
# 7. Existing mechanisms unchanged
# ------------------------------------------------------------------

def test_timeout_still_works():
    class _SlowLLM(LLMProvider):
        def generate(self, messages, tools=None):
            time.sleep(5.0)
            return {"message": {"role": "assistant", "content": "late"}}

    runtime = AgentRuntime(config=RuntimeConfig(timeout_seconds=0.1))
    state = runtime.run(_make_agent(_SlowLLM()), "go")
    assert state.status == "timeout_exceeded"


def test_sigalrm_cleanup_still_correct():
    runtime = AgentRuntime(config=RuntimeConfig(timeout_seconds=0.1))
    runtime.run(_make_agent(_FinalAnswerLLM()), "go")
    assert signal.getsignal(signal.SIGALRM) == signal.SIG_DFL


def test_max_steps_still_works():
    runtime = AgentRuntime(max_steps=2)
    state = runtime.run(_make_agent(_AlwaysToolLLM()), "go")
    assert state.status == "max_steps_exceeded"


def test_max_tool_calls_still_works():
    runtime = AgentRuntime(config=RuntimeConfig(max_steps=100, max_tool_calls=1))
    state = runtime.run(_make_agent(_AlwaysToolLLM()), "go")
    assert state.status == "max_tool_calls_exceeded"


def test_max_llm_calls_still_works():
    runtime = AgentRuntime(config=RuntimeConfig(max_steps=100, max_llm_calls=1))
    state = runtime.run(_make_agent(_AlwaysToolLLM()), "go")
    assert state.status == "max_llm_calls_exceeded"


def test_retry_still_works():
    class _FlakyLLM(LLMProvider):
        def __init__(self):
            self.calls = 0

        def generate(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                raise TransientLLMError("transient")
            return {"message": {"role": "assistant", "content": "Recovered."}}

    runtime = AgentRuntime(config=RuntimeConfig(max_retries=3))
    state = runtime.run(_make_agent(_FlakyLLM()), "go")
    assert state.status == "completed"
    assert state.result == "Recovered."


def test_normal_execution_unchanged():
    runtime = AgentRuntime(max_steps=5)
    state = runtime.run(_make_agent(_FinalAnswerLLM()), "go")
    assert state.status == "completed"
    assert state.result == "Done."
    assert runtime._active_executions == {}


# ------------------------------------------------------------------
# 8. Caller-managed-thread integration (runtime does not spawn threads)
# ------------------------------------------------------------------

def test_caller_thread_can_cancel_running_execution():
    runtime = AgentRuntime(max_steps=100)
    result = {}

    class _PollingBlockingLLM(LLMProvider):
        """Simulates a long operation that cooperatively observes cancellation.

        Loops until the active execution token is marked cancelled. This is
        the cooperative analogue of a blocking call: the runtime does not
        interrupt it via signal; the operation must observe the token.
        """

        def generate(self, messages, tools=None):
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                token = runtime._active_executions.get(
                    next(iter(runtime._active_executions), "")
                )
                if token is not None and token.is_cancelled:
                    return {"message": {"role": "assistant", "content": "done"}}
                time.sleep(0.01)
            return {"message": {"role": "assistant", "content": "done"}}

    def run_target():
        result["state"] = runtime.run(_make_agent(_PollingBlockingLLM()), "go")

    t = threading.Thread(target=run_target)
    t.start()
    time.sleep(0.1)
    token = runtime.cancel()
    assert token is not None
    t.join(timeout=5)
    assert not t.is_alive()
    assert result["state"].status == "cancelled"
    assert runtime._active_executions == {}
