from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    """Configuration for AgentRuntime execution limits."""

    max_steps: int = 10
    max_tool_calls: int | None = None
    max_llm_calls: int | None = None
    timeout_seconds: float | None = None
    max_retries: int = 0
