from providers.ollama import OllamaProvider


llm = OllamaProvider(
    model="qwen3.5:0.8b"
)

response = llm.generate([
    {
        "role": "user",
        "content": "What is 10 + 20?"
    }
])

print(response)