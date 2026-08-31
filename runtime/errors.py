class AgentRuntimeError(Exception):
    """Base exception for agent runtime errors."""


class ToolExecutionError(AgentRuntimeError):
    """Raised when a tool fails during execution."""


class ToolNotFoundError(AgentRuntimeError):
    """Raised when a requested tool is not registered."""


class LLMError(AgentRuntimeError):
    """Raised when the LLM provider fails."""