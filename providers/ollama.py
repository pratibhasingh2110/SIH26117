import requests

from runtime.errors import LLMError, TransientLLMError
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

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload
            )

            response.raise_for_status()

        except (requests.ConnectionError, requests.Timeout) as error:
            raise TransientLLMError(
                f"LLM provider connection failed: {error}"
            ) from error

        except requests.HTTPError as error:
            raise LLMError(
                f"LLM provider returned an error: {error}"
            ) from error

        return response.json()