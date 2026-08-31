from runtime.actions import ToolCall, FinalAnswer
from runtime.agent import Agent
from runtime.context import ContextBuilder
from runtime.executor import ToolExecutor
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

        state.messages.append({
            "role": "system",
            "content": agent.instructions
        })

        state.messages.append({
            "role": "user",
            "content": task
        })

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

            if isinstance(action, ToolCall):

                state.actions.append({
                    "tool": action.tool_name,
                    "arguments": action.arguments
                })

                self.recorder.record(
                    "ToolCall",
                    tool=action.tool_name,
                    arguments=action.arguments
                )

                state.messages.append(
                    action.raw_message
                )

                try:

                    result = executor.execute(action)

                except ToolExecutionError as error:

                    state.observations.append({
                        "tool": action.tool_name,
                        "error": str(error)
                    })

                    self.recorder.record(
                        "ToolExecutionFailed",
                        tool=action.tool_name,
                        error=str(error)
                    )

                    state.messages.append({
                        "role": "tool",
                        "content": f"Tool execution failed: {error}",
                        "name": action.tool_name
                    })

                    continue

                state.observations.append({
                    "tool": action.tool_name,
                    "result": result
                })

                self.recorder.record(
                    "ToolResult",
                    tool=action.tool_name,
                    result=result
                )

                state.messages.append({
                    "role": "tool",
                    "content": str(result),
                    "name": action.tool_name
                })

        return state

    def _build_tool_registry(self, agent: Agent):

        from runtime.tools import ToolRegistry

        registry = ToolRegistry()

        for tool in agent.tools:
            registry.register(tool)

        return registry