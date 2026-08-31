from runtime.runtime import AgentRuntime
from runtime.messages import Message
from runtime.agent import Agent
from runtime.llm import LLMProvider
from runtime.tools import Tool


class _UnknownFirstLLM(LLMProvider):

    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "not_registered",
                                "arguments": {}
                            }
                        }
                    ]
                }
            }
        return {
            "message": {
                "role": "assistant",
                "content": "OK"
            }
        }


class _AlwaysFails(Tool):

    name = "boom"
    description = "always fails"
    input_schema = {"type": "object"}

    def execute(self, arguments):
        raise ValueError("bad")


def test_unknown_tool_request_does_not_crash_run():
    agent = Agent(
        name="UnknownToolAgent",
        instructions="instructions",
        llm=_UnknownFirstLLM(),
        tools=[_AlwaysFails()]
    )

    state = AgentRuntime(max_steps=5).run(agent, "task")

    assert state.status == "completed"
    assert state.result == "OK"
    assert state.step == 2


def test_unknown_tool_records_observation_and_event():
    agent = Agent(
        name="UnknownToolAgent",
        instructions="instructions",
        llm=_UnknownFirstLLM(),
        tools=[_AlwaysFails()]
    )

    runtime = AgentRuntime(max_steps=5)
    state = runtime.run(agent, "task")

    assert len(state.observations) == 1
    assert state.observations[0]["tool"] == "not_registered"
    assert "not_registered" in state.observations[0]["error"]

    event_types = [e.type for e in runtime.recorder.get_events()]
    assert "ToolExecutionFailed" in event_types
    failed = [e for e in runtime.recorder.get_events() if e.type == "ToolExecutionFailed"]
    assert failed[0].data["tool"] == "not_registered"


def test_unknown_tool_appends_tool_error_message():
    agent = Agent(
        name="UnknownToolAgent",
        instructions="instructions",
        llm=_UnknownFirstLLM(),
        tools=[_AlwaysFails()]
    )

    state = AgentRuntime(max_steps=5).run(agent, "task")

    tool_messages = [m for m in state.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "not_registered"
    assert "failed" in tool_messages[0].content

def test_fake_llm_runtime_completes(math_agent):
    runtime = AgentRuntime(max_steps=10)

    state = runtime.run(
        agent=math_agent,
        task="Calculate 10 + 20"
    )

    assert state.status == "completed"
    assert state.result == "The answer is 30."
    assert state.step == 2


def test_fake_llm_runtime_executes_tool(math_agent):
    runtime = AgentRuntime(max_steps=10)

    state = runtime.run(
        agent=math_agent,
        task="Calculate 10 + 20"
    )

    assert len(state.actions) == 1
    assert state.actions[0]["tool"] == "calculator"
    assert state.actions[0]["arguments"] == {"a": 10, "b": 20}


def test_fake_llm_runtime_records_observation(math_agent):
    runtime = AgentRuntime(max_steps=10)

    state = runtime.run(
        agent=math_agent,
        task="Calculate 10 + 20"
    )

    assert len(state.observations) == 1
    assert state.observations[0]["tool"] == "calculator"
    assert state.observations[0]["result"] == 30


def test_runtime_messages_are_message_objects(math_agent):
    runtime = AgentRuntime(max_steps=10)

    state = runtime.run(
        agent=math_agent,
        task="Calculate 10 + 20"
    )

    assert all(isinstance(msg, Message) for msg in state.messages)

    roles = [msg.role for msg in state.messages]
    assert roles == ["system", "user", "assistant", "tool"]


def test_runtime_serializes_to_dict_context(math_agent):
    runtime = AgentRuntime(max_steps=10)

    state = runtime.run(
        agent=math_agent,
        task="Calculate 10 + 20"
    )

    context = runtime.context_builder.build(state, runtime._build_tool_registry(math_agent))

    assert all(isinstance(msg, dict) for msg in context)
    assert context[0]["role"] == "system"
    assert context[0]["content"] == "You are a math assistant."
    assert context[2]["tool_calls"]
    assert context[3]["role"] == "tool"
