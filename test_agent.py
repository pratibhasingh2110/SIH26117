from runtime.agent import Agent
from runtime.llm import LLMProvider


class FakeLLM(LLMProvider):

    def generate(self, messages, tools=None):
        return "test"


agent = Agent(
    name="TestAgent",
    instructions="You are a test agent.",
    llm=FakeLLM()
)

print(agent)