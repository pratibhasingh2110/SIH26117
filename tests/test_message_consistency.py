from runtime.messages import Message
from runtime.context import ContextBuilder
from runtime.state import AgentState
from runtime.tools import ToolRegistry


def _registry():
    return ToolRegistry()


def _message(role, content=None):
    return Message(role=role, content=content)


def test_messages_have_required_attributes():
    message = _message("user", "hello")
    assert message.role == "user"
    assert message.content == "hello"
    assert message.name is None
    assert message.tool_calls == []
    assert message.tool_call_id is None


def test_context_builder_serializes_message_objects():
    state = AgentState(task="task")
    state.messages.append(Message(
        role="system",
        content="instructions"
    ))
    state.messages.append(Message(
        role="user",
        content="hello"
    ))

    context = ContextBuilder().build(state, _registry())

    assert isinstance(context[0], dict)
    assert context[0]["role"] == "system"
    assert context[0]["content"] == "instructions"
    assert context[1]["role"] == "user"
    assert context[1]["content"] == "hello"


def test_context_builder_preserves_tool_fields():
    state = AgentState(task="task")
    state.messages.append(Message(
        role="assistant",
        content="",
        tool_calls=[{"function": {"name": "calculator"}}]
    ))
    state.messages.append(Message(
        role="tool",
        content="30",
        name="calculator"
    ))

    context = ContextBuilder().build(state, _registry())

    assert context[0]["tool_calls"] == [{"function": {"name": "calculator"}}]
    assert context[1]["name"] == "calculator"


def test_no_implicit_system_prompt_injected():
    state = AgentState(task="task")
    state.messages.append(Message(role="user", content="hello"))

    context = ContextBuilder().build(state, _registry())

    assert len(context) == 1
    assert context[0]["role"] == "user"
