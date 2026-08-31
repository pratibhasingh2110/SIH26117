class AgentRuntimeError(Exception):
    """Base exception for agent runtime errors."""

    retryable: bool = False


class ToolExecutionError(AgentRuntimeError):
    """Raised when a tool fails during execution."""


class TransientToolExecutionError(ToolExecutionError):
    """Raised for transient tool failures that may succeed on retry."""

    retryable: bool = True


class ToolNotFoundError(AgentRuntimeError):
    """Raised when a requested tool is not registered."""


class LLMError(AgentRuntimeError):
    """Raised when the LLM provider fails."""


class TransientLLMError(LLMError):
    """Raised for transient LLM/provider failures that may succeed on retry."""

    retryable: bool = True