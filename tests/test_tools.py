import pytest

from runtime.tools import ToolRegistry, Tool
from runtime.errors import ToolNotFoundError, AgentRuntimeError


def test_get_registered_tool(calculator_tool):
    registry = ToolRegistry()
    registry.register(calculator_tool)

    tool = registry.get("calculator")

    assert tool is calculator_tool


def test_get_unknown_tool_raises_runtime_error():
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError) as excinfo:
        registry.get("does_not_exist")

    assert "does_not_exist" in str(excinfo.value)


def test_get_unknown_tool_is_not_keyerror():
    registry = ToolRegistry()

    with pytest.raises(Exception) as excinfo:
        registry.get("missing")

    assert not isinstance(excinfo.value, KeyError)


def test_unknown_error_subclass_of_base_error():
    registry = ToolRegistry()

    with pytest.raises(AgentRuntimeError):
        registry.get("missing")
