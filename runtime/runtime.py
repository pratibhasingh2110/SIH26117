from runtime.actions import ToolCall, FinalAnswer
from runtime.agent import Agent
from runtime.context import ContextBuilder
from runtime.executor import ToolExecutor
from runtime.messages import Message
from runtime.parser import ResponseParser
from runtime.state import AgentState
from runtime.errors import ToolExecutionError
from runtime.events import EventRecorder


class AgentRuntime:

    def __init__(
        self,
        max_steps: int = 10,
        recorder: EventRecorder | None = None
    ):
        self.max_steps = max_steps
        self.recorder = recorder or EventRecorder()

        self.context_builder = ContextBuilder()
        self.parser = ResponseParser()

        self._tool_call_counter = 0

    def _next_tool_call_id(self):
        self._tool_call_counter += 1
        return f"tool_call_{self._tool_call_counter}"

    def run(
        self,
        agent: Agent,
        task: str
    ) -> AgentState:

        state = AgentState(task=task)

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

        while state.status == "running":

            state.step += 1

            self.recorder.record(
                "StepStarted",
                step=state.step
            )

            if state.step > self.max_steps:

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

            context = self.context_builder.build(
                state,
                tool_registry
            )

            self.recorder.record(
                "LLMCallStarted",
                step=state.step
            )

            try:

                response = agent.llm.generate(
                    context,
                    tools=tool_registry.list_tools()
                )

            except Exception as error:

                state.status = "failed"
                state.result = str(error)

                self.recorder.record(
                    "LLMCallFailed",
                    step=state.step,
                    error=str(error)
                )

                return state

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

                        result = executor.execute(tool_call)

                    except ToolExecutionError as error:

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

        return state

    def _build_tool_registry(self, agent: Agent):

        from runtime.tools import ToolRegistry

        registry = ToolRegistry()

        for tool in agent.tools:
            registry.register(tool)

        return registry