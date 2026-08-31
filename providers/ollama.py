import requests

from runtime.llm import LLMProvider


class OllamaProvider(LLMProvider):

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    def generate(self, messages, tools=None):

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    }
                }
                for tool in tools
            ]

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload
        )

        response.raise_for_status()

        return response.json()