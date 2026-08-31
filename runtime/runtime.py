import signal
import time

from runtime.actions import ToolCall, FinalAnswer
from runtime.agent import Agent
from runtime.config import RuntimeConfig
from runtime.context import ContextBuilder
from runtime.executor import ToolExecutor
from runtime.messages import Message
from runtime.parser import ResponseParser
from runtime.state import AgentState
from runtime.errors import ToolExecutionError
from runtime.events import EventRecorder


class _TimeoutExceeded(BaseException):
    """Raised internally when the execution timeout is exceeded."""


class _RetryExhausted(Exception):
    """Raised internally when retries are exhausted and execution must stop."""


def _timeout_handler(signum, frame):
    raise _TimeoutExceeded()


class AgentRuntime:

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        max_steps: int = 10,
        recorder: EventRecorder | None = None,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = RuntimeConfig(max_steps=max_steps)

        self.recorder = recorder or EventRecorder()

        self.context_builder = ContextBuilder()
        self.parser = ResponseParser()

        self._tool_call_counter = 0
        self._tool_call_count = 0
        self._llm_call_count = 0
        self._last_llm_error = None

    def _next_tool_call_id(self):
        self._tool_call_counter += 1
        return f"tool_call_{self._tool_call_counter}"

    def _is_retryable(self, error) -> bool:
        return (
            self.config.max_retries > 0
            and getattr(error, "retryable", False)
        )

    def _generate_with_retry(self, agent, context, tool_registry, state):
        for attempt in range(self.config.max_retries + 1):
            if attempt > 0:
                self.recorder.record(
                    "LLMRetry",
                    attempt=attempt,
                    max_retries=self.config.max_retries,
                    error=self._last_llm_error,
                    step=state.step,
                )

            try:
                return agent.llm.generate(
                    context,
                    tools=tool_registry.list_tools()
                )
            except _TimeoutExceeded:
                raise
            except Exception as error:
                if not self._is_retryable(error):
                    state.status = "failed"
                    state.result = str(error)
                    self.recorder.record(
                        "LLMCallFailed",
                        step=state.step,
                        error=str(error),
                    )
                    raise _RetryExhausted() from error

                if attempt < self.config.max_retries:
                    self._last_llm_error = str(error)
                    continue

                state.status = "failed"
                state.result = str(error)
                self.recorder.record(
                    "LLMRetryFailed",
                    step=state.step,
                    error=str(error),
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries,
                )
                raise _RetryExhausted() from error

        raise _RetryExhausted()

    def _execute_tool_with_retry(self, executor, tool_call, state):
        for attempt in range(self.config.max_retries + 1):
            try:
                return executor.execute(tool_call)
            except _TimeoutExceeded:
                raise
            except ToolExecutionError as error:
                if not self._is_retryable(error):
                    raise

                if attempt < self.config.max_retries:
                    self.recorder.record(
                        "ToolRetry",
                        attempt=attempt + 1,
                        max_retries=self.config.max_retries,
                        error=str(error),
                        tool=tool_call.tool_name,
                        step=state.step,
                    )
                    continue

                self.recorder.record(
                    "ToolRetryFailed",
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries,
                    error=str(error),
                    tool=tool_call.tool_name,
                    step=state.step,
                )
                raise

        raise ToolExecutionError("Tool execution retries exhausted.")

    def run(
        self,
        agent: Agent,
        task: str
    ) -> AgentState:

        state = AgentState(task=task)
        start_time = time.monotonic()

        tool_registry = self._build_tool_registry(agent)

        executor = ToolExecutor(tool_registry)

        self.recorder.record(
            "AgentStarted",
            agent=agent.name,
            task=task
        )

        state.messages.append(Message(
            role="system",
            content=agent.instructions
        ))

        state.messages.append(Message(
            role="user",
            content=task
        ))

        try:
            if self.config.timeout_seconds is not None:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.setitimer(
                    signal.ITIMER_REAL,
                    self.config.timeout_seconds
                )

            while state.status == "running":

                state.step += 1

                self.recorder.record(
                    "StepStarted",
                    step=state.step
                )

                if state.step > self.config.max_steps:

                    state.status = "max_steps_exceeded"
                    state.result = (
                        "Agent exceeded maximum execution steps."
                    )

                    self.recorder.record(
                        "AgentStopped",
                        reason="max_steps_exceeded",
                        steps=state.step
                    )

                    return state

                if self.config.timeout_seconds is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= self.config.timeout_seconds:

                        state.status = "timeout_exceeded"
                        state.result = (
                            f"Agent execution timed out after "
                            f"{elapsed:.1f}s "
                            f"(limit: {self.config.timeout_seconds}s)."
                        )

                        self.recorder.record(
                            "AgentStopped",
                            reason="timeout_exceeded",
                            elapsed=elapsed,
                            timeout=self.config.timeout_seconds,
                            steps=state.step
                        )

                        return state

                context = self.context_builder.build(
                    state,
                    tool_registry
                )

                if (
                    self.config.max_llm_calls is not None
                    and self._llm_call_count >= self.config.max_llm_calls
                ):

                    state.status = "max_llm_calls_exceeded"
                    state.result = (
                        "Agent exceeded maximum LLM calls."
                    )

                    self.recorder.record(
                        "AgentStopped",
                        reason="max_llm_calls_exceeded",
                        max_llm_calls=self.config.max_llm_calls,
                        llm_call_count=self._llm_call_count,
                        steps=state.step
                    )

                    return state

                self.recorder.record(
                    "LLMCallStarted",
                    step=state.step
                )

                try:

                    response = self._generate_with_retry(
                        agent,
                        context,
                        tool_registry,
                        state
                    )

                except _RetryExhausted:
                    return state

                except _TimeoutExceeded:
                    raise

                self._llm_call_count += 1

                self.recorder.record(
                    "LLMCallCompleted",
                    step=state.step
                )

                action = self.parser.parse(response)

                if isinstance(action, FinalAnswer):

                    state.result = action.content
                    state.status = "completed"

                    self.recorder.record(
                        "AgentCompleted",
                        result=action.content,
                        steps=state.step
                    )

                    return state

                if isinstance(action, list):

                    for tool_call in action:
                        if tool_call.tool_call_id is None:
                            tool_call.tool_call_id = self._next_tool_call_id()

                    raw = action[0].raw_message

                    assistant_tool_calls = []
                    for tool_call in action:
                        entry = {
                            "id": tool_call.tool_call_id,
                            "function": {
                                "name": tool_call.tool_name,
                                "arguments": tool_call.arguments
                            }
                        }
                        assistant_tool_calls.append(entry)

                    state.messages.append(Message(
                        role=raw.get("role", "assistant"),
                        content=raw.get("content"),
                        tool_calls=assistant_tool_calls,
                    ))

                    for tool_call in action:

                        if (
                            self.config.max_tool_calls is not None
                            and self._tool_call_count >= self.config.max_tool_calls
                        ):

                            state.status = "max_tool_calls_exceeded"
                            state.result = (
                                "Agent exceeded maximum tool calls."
                            )

                            self.recorder.record(
                                "AgentStopped",
                                reason="max_tool_calls_exceeded",
                                tool_calls=self._tool_call_count,
                                steps=state.step
                            )

                            return state

                        state.actions.append({
                            "tool": tool_call.tool_name,
                            "arguments": tool_call.arguments,
                            "tool_call_id": tool_call.tool_call_id
                        })

                        self.recorder.record(
                            "ToolCall",
                            tool=tool_call.tool_name,
                            arguments=tool_call.arguments,
                            tool_call_id=tool_call.tool_call_id
                        )

                        try:

                            result = self._execute_tool_with_retry(
                                executor,
                                tool_call,
                                state
                            )

                        except ToolExecutionError as error:

                            self._tool_call_count += 1

                            state.observations.append({
                                "tool": tool_call.tool_name,
                                "error": str(error),
                                "tool_call_id": tool_call.tool_call_id
                            })

                            self.recorder.record(
                                "ToolExecutionFailed",
                                tool=tool_call.tool_name,
                                error=str(error),
                                tool_call_id=tool_call.tool_call_id
                            )

                            state.messages.append(Message(
                                role="tool",
                                content=f"Tool execution failed: {error}",
                                name=tool_call.tool_name,
                                tool_call_id=tool_call.tool_call_id
                            ))

                            continue

                        self._tool_call_count += 1

                        state.observations.append({
                            "tool": tool_call.tool_name,
                            "result": result,
                            "tool_call_id": tool_call.tool_call_id
                        })

                        self.recorder.record(
                            "ToolResult",
                            tool=tool_call.tool_name,
                            result=result,
                            tool_call_id=tool_call.tool_call_id
                        )

                        state.messages.append(Message(
                            role="tool",
                            content=str(result),
                            name=tool_call.tool_name,
                            tool_call_id=tool_call.tool_call_id
                        ))

        except _TimeoutExceeded:
            elapsed = time.monotonic() - start_time
            state.status = "timeout_exceeded"
            state.result = (
                f"Agent execution timed out after "
                f"{elapsed:.1f}s "
                f"(limit: {self.config.timeout_seconds}s)."
            )
            self.recorder.record(
                "AgentStopped",
                reason="timeout_exceeded",
                elapsed=elapsed,
                timeout=self.config.timeout_seconds,
                steps=state.step
            )
            return state

        finally:
            if self.config.timeout_seconds is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)

        return state

    def _build_tool_registry(self, agent: Agent):

        from runtime.tools import ToolRegistry

        registry = ToolRegistry()

        for tool in agent.tools:
            registry.register(tool)

        return registry
