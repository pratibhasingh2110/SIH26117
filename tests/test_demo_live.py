"""Standalone live demo test (requires a running Ollama with a model).

This test exercises the real golden path against a live Ollama instance.
It is skipped (not failed) when Ollama is unreachable or no suitable
model is installed, so the normal `pytest tests/` suite never depends on
Ollama.

Set OLLAMA_MODEL (default: qwen2.5:7b) to point at a model already
pulled into Ollama.
"""

import os

import pytest

from demo.runner import DemoRuntime

_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")


def _ollama_available() -> str | None:
    import requests

    try:
        r = requests.get(f"{_BASE_URL}/api/tags", timeout=3)
        r.raise_for_status()
        names = [m.get("name") for m in r.json().get("models", [])]
    except Exception:
        return None

    if _MODEL in names:
        return _MODEL
    return None


@pytest.mark.skipif(
    _ollama_available() is None,
    reason=f"Ollama unreachable at {_BASE_URL} or model '{_MODEL}' not pulled",
)
def test_golden_path_live():
    demo = DemoRuntime(model=_MODEL, base_url=_BASE_URL)

    result = demo.run("Use the calculator to calculate 25 + 17.")

    assert result["status"] == "completed"
    assert result["agent"] == "MathAgent"
    assert result["steps"] >= 2

    order = [e["type"] for e in result["events"]]
    assert (
        "AgentRoutingStarted" in order
        and "AgentRoutingCompleted" in order
        and "AgentStarted" in order
        and "ToolCall" in order
        and "ToolResult" in order
        and "AgentCompleted" in order
    )

    # The answer must resolve to 42 (model may phrase it differently).
    assert "42" in str(result["result"])
