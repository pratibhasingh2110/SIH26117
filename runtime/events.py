from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeEvent:
    type: str
    data: dict[str, Any]

class EventRecorder:

    def __init__(self):
        self.events: list[RuntimeEvent] = []

    def record(self, event_type: str, **data):
        self.events.append(
            RuntimeEvent(
                type=event_type,
                data=data
            )
        )

    def get_events(self):
        return self.events