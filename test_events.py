from runtime.events import EventRecorder


recorder = EventRecorder()

recorder.record(
    "AgentStarted",
    task="Calculate 25 + 17"
)

recorder.record(
    "ToolCalled",
    tool="calculator"
)

for event in recorder.get_events():
    print(event)