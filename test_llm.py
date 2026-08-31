from runtime.llm import LLMProvider


class TestLLM(LLMProvider):

    def generate(self, messages, tools=None):
        return "Hello from LLM"


llm = TestLLM()

response = llm.generate(
    [{"role": "user", "content": "Hello"}]
)

print(response)