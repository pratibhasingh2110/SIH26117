import time
from dataclasses import dataclass, field


@dataclass
class CancellationToken:
    """Execution-scoped cancellation state for a single AgentRuntime run."""

    execution_id: str
    created_at: float
    _cancelled: bool = field(default=False, repr=False)

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def request_cancellation(self) -> None:
        self._cancelled = True


def new_token(execution_id: str) -> CancellationToken:
    """Create a CancellationToken stamped with the current monotonic time."""
    return CancellationToken(
        execution_id=execution_id,
        created_at=time.monotonic(),
    )
