"""Minimal demo/API layer for the agent runtime.

This module exposes the existing AgentRouter + AgentRuntime pipeline
through a single callable interface. It does NOT re-implement any
runtime logic; it only orchestrates the existing components and
serializes their output for presentation.

The runtime itself is provider-independent, so this layer can be
exercised with any LLMProvider (including fakes) and does not require
Ollama for the normal test suite.
"""

from __future__ import annotations

from typing import Any

from runtime import AgentRuntime, EventRecorder, RuntimeConfig
from router import AgentRouter, AgentRoutingError
from agents.registry import build_agents


class DemoError(Exception):
    """Base error for the demo/API layer."""


class RoutingError(DemoError):
    """Raised when agent routing fails."""


class ExecutionError(DemoError):
    """Raised when a runtime execution fails."""


#: Terminal statuses that represent a usable (non-error) outcome.
_OK_STATUSES = frozenset({
    "completed",
    "max_steps_exceeded",
    "max_tool_calls_exceeded",
    "max_llm_calls_exceeded",
    "timeout_exceeded",
    "cancelled",
})


class DemoRuntime:
    """Thin facade that wires the existing router + runtime together.

    A fresh EventRecorder is shared by the router and the runtime so
    the full lifecycle trace is captured in one place.
    """

    def __init__(
        self,
        agents=None,
        *,
        recorder: EventRecorder | None = None,
        config: RuntimeConfig | None = None,
        max_steps: int = 10,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.recorder = recorder or EventRecorder()
        kwargs = {}
        if model is not None:
            kwargs["model"] = model
        if base_url is not None:
            kwargs["base_url"] = base_url
        self.agents = agents if agents is not None else build_agents(**kwargs)
        self.router = AgentRouter(self.agents, recorder=self.recorder)
        self.runtime = AgentRuntime(
            config=config,
            max_steps=max_steps,
            recorder=self.recorder,
        )

    def run(self, task: str) -> dict[str, Any]:
        if not task or not task.strip():
            raise ExecutionError("Task must not be empty.")

        try:
            routing = self.router.route(task)
        except AgentRoutingError as error:
            raise RoutingError(str(error)) from error

        agent = routing.agent
        state = self.runtime.run(agent, task)

        if state.status == "failed":
            raise ExecutionError(
                state.result or "Agent execution failed."
            )

        return serialize_run_result(
            state=state,
            agent=agent.name,
            routing_reason=routing.reason,
            events=self.recorder.get_events(),
        )


def _event_data(event) -> dict[str, Any]:
    """Flatten a RuntimeEvent into a serializable dict.

    The event `type` is always present. Only render useful,
    presentation-safe data; arbitrary objects (e.g. raw message dicts)
    are intentionally omitted.
    """
    data: dict[str, Any] = {"type": event.type}
    for key, value in event.data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            data[key] = value
        elif isinstance(value, dict) and all(
            isinstance(v, (str, int, float, bool)) for v in value.values()
        ):
            data[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (str, int, float, bool)) for v in value
        ):
            data[key] = list(value)
    return data


def serialize_run_result(
    *,
    state,
    agent: str,
    routing_reason: str,
    events,
) -> dict[str, Any]:
    """Serialize a completed/failed-tolerable run into a response dict.

    Uses the existing AgentState.execution_id for execution identity.
    """
    return {
        "execution_id": state.execution_id,
        "agent": agent,
        "status": state.status,
        "result": state.result if state.result is not None else "",
        "steps": state.step,
        "routing_reason": routing_reason,
        "events": [_event_data(e) for e in events],
    }


def serialize_error(
    *,
    error: Exception,
    agent: str = "",
    events=None,
) -> dict[str, Any]:
    """Serialize a demo/API error without exposing a Python traceback."""
    events = events or []
    return {
        "execution_id": "",
        "agent": agent,
        "status": "error",
        "result": _safe_message(error),
        "steps": 0,
        "error_type": type(error).__name__,
        "events": [_event_data(e) for e in events],
    }


def _safe_message(error: Exception) -> str:
    """Return a user-presentable message, never a full traceback."""
    text = str(error).strip()
    return text if text else type(error).__name__


def run_task(
    task: str,
    *,
    agents=None,
    recorder: EventRecorder | None = None,
    config: RuntimeConfig | None = None,
    max_steps: int = 10,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Callable entry point for the presentation layer.

    Returns a serialized response dict, or raises DemoError on failure.
    """
    demo = DemoRuntime(
        agents=agents,
        recorder=recorder,
        config=config,
        max_steps=max_steps,
        model=model,
        base_url=base_url,
    )
    return demo.run(task)
