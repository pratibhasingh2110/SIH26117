from runtime.tools import Tool, ToolRegistry


class Calculator(Tool):

    name = "calculator"
    description = "Adds two numbers"

    input_schema = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    }

    def execute(self, arguments):
        return arguments["a"] + arguments["b"]


registry = ToolRegistry()

calculator = Calculator()
registry.register(calculator)

tool = registry.get("calculator")

result = tool.execute({
    "a": 10,
    "b": 20
})

print(result)