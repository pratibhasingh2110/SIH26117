from runtime.runtime import AgentRuntime
from runtime.agent import Agent
from runtime.llm import LLMProvider
from runtime.tools import Tool


class AddTool(Tool):
    name = "add"
    description = "adds"
    input_schema = {"type": "object"}

    def execute(self, arguments):
        return arguments["a"] + arguments["b"]


class MultiplyTool(Tool):
    name = "multiply"
    description = "multiplies"
    input_schema = {"type": "object"}

    def execute(self, arguments):
        return arguments["a"] * arguments["b"]


class _TwoToolBatchLLM(LLMProvider):

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
                                "name": "add",
                                "arguments": {"a": 1, "b": 2}
                            }
                        },
                        {
                            "function": {
                                "name": "multiply",
                                "arguments": {"a": 3, "b": 4}
                            }
                        }
                    ]
                }
            }
        return {
            "message": {
                "role": "assistant",
                "content": "done"
            }
        }


def _two_tool_agent(llm):
    return Agent(
        name="TwoToolAgent",
        instructions="instructions",
        llm=llm,
        tools=[AddTool(), MultiplyTool()]
    )


def test_tool_call_without_id_is_assigned_deterministic_id():
    llm = _TwoToolBatchLLM()
    state = AgentRuntime(max_steps=10).run(_two_tool_agent(llm), "task")

    assert state.actions[0]["tool_call_id"] == "tool_call_1"
    assert state.actions[1]["tool_call_id"] == "tool_call_2"


def test_provider_supplied_id_is_preserved_through_execution():
    class ExplicitLLM(LLMProvider):
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
                                "id": "opaque_provider_id_99",
                                "function": {"name": "add", "arguments": {"a": 5, "b": 6}}
                            }
                        ]
                    }
                }
            return {"message": {"role": "assistant", "content": "done"}}

    state = AgentRuntime(max_steps=10).run(_two_tool_agent(ExplicitLLM()), "task")

    assert state.actions[0]["tool_call_id"] == "opaque_provider_id_99"
    tool_message = next(m for m in state.messages if m.role == "tool")
    assert tool_message.tool_call_id == "opaque_provider_id_99"


def test_injected_ids_are_unique():
    llm = _TwoToolBatchLLM()
    state = AgentRuntime(max_steps=10).run(_two_tool_agent(llm), "task")

    ids = [a["tool_call_id"] for a in state.actions]
    assert len(ids) == len(set(ids))


def test_tool_result_messages_carry_corresponding_tool_call_id():
    llm = _TwoToolBatchLLM()
    state = AgentRuntime(max_steps=10).run(_two_tool_agent(llm), "task")

    tool_messages = [m for m in state.messages if m.role == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0].tool_call_id == "tool_call_1"
    assert tool_messages[1].tool_call_id == "tool_call_2"


def test_assistant_tool_call_message_preserves_ids():
    llm = _TwoToolBatchLLM()
    state = AgentRuntime(max_steps=10).run(_two_tool_agent(llm), "task")

    assistant = [m for m in state.messages if m.role == "assistant"]
    assert len(assistant) == 1
    assert [tc["id"] for tc in assistant[0].tool_calls] == ["tool_call_1", "tool_call_2"]


def test_multiple_tool_calls_execute_in_order():
    recorded = []

    class RecordingAdd(AddTool):
        def execute(self, arguments):
            recorded.append("add")
            return super().execute(arguments)

    class RecordingMultiply(MultiplyTool):
        def execute(self, arguments):
            recorded.append("multiply")
            return super().execute(arguments)

    class RecordingLLM(LLMProvider):
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
                            {"function": {"name": "add", "arguments": {"a": 1, "b": 2}}},
                            {"function": {"name": "multiply", "arguments": {"a": 3, "b": 4}}},
                        ]
                    }
                }
            return {"message": {"role": "assistant", "content": "done"}}

    agent = Agent(
        name="OrderAgent",
        instructions="instructions",
        llm=RecordingLLM(),
        tools=[RecordingAdd(), RecordingMultiply()]
    )

    state = AgentRuntime(max_steps=10).run(agent, "task")

    assert recorded == ["add", "multiply"]
    assert [a["tool"] for a in state.actions] == ["add", "multiply"]


def test_complete_multi_tool_runtime_execution():
    state = AgentRuntime(max_steps=10).run(_two_tool_agent(_TwoToolBatchLLM()), "task")

    assert state.status == "completed"
    assert state.result == "done"
    assert state.step == 2
    assert len(state.actions) == 2
    assert len(state.observations) == 2
    assert state.observations[0]["result"] == 3
    assert state.observations[1]["result"] == 12


def test_context_serializes_tool_call_ids():
    runtime = AgentRuntime(max_steps=10)
    state = runtime.run(_two_tool_agent(_TwoToolBatchLLM()), "task")

    context = runtime.context_builder.build(state, runtime._build_tool_registry(_two_tool_agent(_TwoToolBatchLLM())))

    assistant = context[2]
    assert assistant["tool_calls"][0]["id"] == "tool_call_1"
    assert assistant["tool_calls"][1]["id"] == "tool_call_2"

    assert context[3]["role"] == "tool"
    assert context[3]["tool_call_id"] == "tool_call_1"
    assert context[4]["role"] == "tool"
    assert context[4]["tool_call_id"] == "tool_call_2"
