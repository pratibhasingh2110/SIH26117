from providers.ollama import OllamaProvider
from runtime.actions import ToolCall, FinalAnswer
from runtime.parser import ResponseParser


llm = OllamaProvider(
    model="qwen3.5:0.8b"
)

response = llm.generate([
    {
        "role": "user",
        "content": "What is 10 + 20?"
    }
])

parser = ResponseParser()

action = parser.parse(response)

print(action)
print(type(action))