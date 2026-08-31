import runtime
from providers.ollama import OllamaProvider
from runtime.agent import Agent
from runtime.runtime import AgentRuntime
from runtime.tools import Tool
from router import AgentRouter


class Calculator(Tool):
    name = "calculator"
    description = "Adds two numbers"
    input_schema = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    }

    def execute(self, arguments):
        return arguments["a"] + arguments["b"]


math_agent = Agent(
    name="MathAgent",
    description="Handles arithmetic and mathematical calculation tasks.",
    instructions="You are a math assistant. Use the calculator tool for arithmetic.",
    llm=OllamaProvider(model="qwen3.5:0.8b"),
    tools=[Calculator()],
)

research_agent = Agent(
    name="ResearchAgent",
    description="Handles information lookup and research-oriented tasks.",
    instructions="You are a research assistant. Summarize information for research tasks.",
    llm=OllamaProvider(model="qwen3.5:0.8b"),
)

general_agent = Agent(
    name="GeneralAgent",
    description="Handles general tasks that do not clearly belong to another specialist.",
    instructions="You are a general assistant. Answer general-purpose questions directly.",
    llm=OllamaProvider(model="qwen3.5:0.8b"),
)


router = AgentRouter([math_agent, research_agent, general_agent])

task = "Use the calculator to calculate 25 + 17."

result = router.route(task)
print("\nROUTER SELECTED:", result.agent.name)
print("REASON:", result.reason)

state = AgentRuntime(max_steps=10).run(result.agent, task)

print("\nSTATUS:", state.status)
print("STEPS:", state.step)
print("ACTIONS:", state.actions)
print("OBSERVATIONS:", state.observations)
print("RESULT:", state.result)

print("\nTRACE:")
for event in router.recorder.get_events():
    print(event)
for event in AgentRuntime(max_steps=10).recorder.get_events():
    print(event)
