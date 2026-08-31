from runtime.parser import ResponseParser
from runtime.actions import ToolCall, FinalAnswer


def _response(message):
    return {"message": message}


def test_parses_final_answer():
    response = _response({"role": "assistant", "content": "done"})

    action = ResponseParser().parse(response)

    assert isinstance(action, FinalAnswer)
    assert action.content == "done"


def test_parses_no_content_as_empty_final_answer():
    response = _response({"role": "assistant"})

    action = ResponseParser().parse(response)

    assert isinstance(action, FinalAnswer)
    assert action.content == ""


def test_parses_single_tool_call():
    response = _response({
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "calculator",
                    "arguments": {"a": 10, "b": 20}
                }
            }
        ]
    })

    actions = ResponseParser().parse(response)

    assert isinstance(actions, list)
    assert len(actions) == 1
    assert isinstance(actions[0], ToolCall)
    assert actions[0].tool_name == "calculator"
    assert actions[0].arguments == {"a": 10, "b": 20}


def test_parses_string_json_arguments():
    response = _response({
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "calculator",
                    "arguments": '{"a": 10, "b": 20}'
                }
            }
        ]
    })

    actions = ResponseParser().parse(response)

    assert actions[0].arguments == {"a": 10, "b": 20}


def test_parses_multiple_tool_calls():
    response = _response({
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
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
            },
            {
                "id": "call_3",
                "function": {
                    "name": "subtract",
                    "arguments": {"a": 5, "b": 6}
                }
            }
        ]
    })

    actions = ResponseParser().parse(response)

    assert isinstance(actions, list)
    assert len(actions) == 3
    assert [a.tool_name for a in actions] == ["add", "multiply", "subtract"]
    assert actions[1].arguments == {"a": 3, "b": 4}
    assert actions[2].arguments == {"a": 5, "b": 6}


def test_parser_preserves_provider_supplied_id():
    response = _response({
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "provider_call_abc",
                "function": {
                    "name": "calculator",
                    "arguments": {"a": 1, "b": 2}
                }
            }
        ]
    })

    actions = ResponseParser().parse(response)

    assert actions[0].tool_call_id == "provider_call_abc"


def test_parser_leaves_id_none_when_not_supplied():
    response = _response({
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "calculator",
                    "arguments": {"a": 1, "b": 2}
                }
            }
        ]
    })

    actions = ResponseParser().parse(response)

    assert actions[0].tool_call_id is None


def test_parser_preserves_ordering_of_multiple_ids():
    response = _response({
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "a", "function": {"name": "x", "arguments": {}}},
            {"id": "b", "function": {"name": "y", "arguments": {}}},
        ]
    })

    actions = ResponseParser().parse(response)

    assert [a.tool_call_id for a in actions] == ["a", "b"]
