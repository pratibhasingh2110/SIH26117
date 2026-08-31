from providers.ollama import OllamaProvider

from runtime.agent import Agent
from runtime.runtime import AgentRuntime
from runtime.tools import Tool


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


llm = OllamaProvider(
    model="qwen3.5:0.8b"
)


agent = Agent(
    name="MathAgent",
    instructions="You are a mathematical assistant.",
    llm=llm,
    tools=[
        Calculator()
    ]
)


runtime = AgentRuntime(
    max_steps=10
)


state = runtime.run(
    agent=agent,
    task="Use the calculator to calculate 25 + 17."
)


print("\nSTATUS:", state.status)
print("STEPS:", state.step)
print("ACTIONS:", state.actions)
print("OBSERVATIONS:", state.observations)
print("RESULT:", state.result)


print("\nTRACE:")

for event in runtime.recorder.get_events():
    print(event)