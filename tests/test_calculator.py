import pytest

from runtime.errors import ToolExecutionError


def test_addition_without_operation_defaults_to_add(calculator_tool):
    assert calculator_tool.execute({"a": 25, "b": 17}) == 42


def test_explicit_addition(calculator_tool):
    assert calculator_tool.execute({"a": 25, "b": 17, "operation": "add"}) == 42


def test_addition_with_plus_symbol(calculator_tool):
    assert calculator_tool.execute({"a": 25, "b": 17, "operation": "+"}) == 42


def test_subtraction(calculator_tool):
    assert calculator_tool.execute({"a": 25, "b": 17, "operation": "subtract"}) == 8
    assert calculator_tool.execute({"a": 25, "b": 17, "operation": "-"}) == 8


def test_multiplication(calculator_tool):
    assert calculator_tool.execute({"a": 25, "b": 17, "operation": "multiply"}) == 425
    assert calculator_tool.execute({"a": 25, "b": 17, "operation": "*"}) == 425


def test_division(calculator_tool):
    assert calculator_tool.execute({"a": 25, "b": 17, "operation": "divide"}) == 25 / 17
    assert calculator_tool.execute({"a": 25, "b": 17, "operation": "/"}) == 25 / 17


def test_division_by_zero_raises_error(calculator_tool):
    with pytest.raises(ValueError, match="Division by zero"):
        calculator_tool.execute({"a": 25, "b": 0, "operation": "divide"})


def test_division_by_zero_is_tool_execution_error(calculator_tool):
    from runtime.actions import ToolCall
    from runtime.executor import ToolExecutor
    from runtime.tools import ToolRegistry

    registry = ToolRegistry()
    registry.register(calculator_tool)
    executor = ToolExecutor(registry)

    with pytest.raises(ToolExecutionError, match="Division by zero"):
        executor.execute(
            ToolCall(
                tool_name="calculator",
                arguments={"a": 25, "b": 0, "operation": "divide"},
                raw_message={},
            )
        )


@pytest.mark.parametrize("operation", ["modulo", "%", "power", "sqrt", "addd", "  "])
def test_invalid_operation_raises_error(calculator_tool, operation):
    with pytest.raises(ValueError, match="Unsupported operation"):
        calculator_tool.execute({"a": 25, "b": 17, "operation": operation})


def test_operation_is_optional_in_schema(calculator_tool):
    properties = calculator_tool.input_schema["properties"]
    assert "operation" in properties
    assert "operation" not in calculator_tool.input_schema.get("required", [])
    assert "a" in calculator_tool.input_schema["required"]
    assert "b" in calculator_tool.input_schema["required"]


def test_backward_compatibility_schema_still_requires_a_and_b(calculator_tool):
    properties = calculator_tool.input_schema["properties"]
    assert properties["a"]["type"] == "number"
    assert properties["b"]["type"] == "number"
