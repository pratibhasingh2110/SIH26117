import json

from runtime.actions import ToolCall, FinalAnswer


class ResponseParser:

    def parse(self, response):

        message = response["message"]

        tool_calls = message.get("tool_calls")

        if tool_calls:

            calls = []

            for tc in tool_calls:

                arguments = tc["function"]["arguments"]

                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                calls.append(ToolCall(
                    tool_name=tc["function"]["name"],
                    arguments=arguments,
                    raw_message=message,
                    tool_call_id=tc.get("id")
                ))

            return calls

        return FinalAnswer(
            content=message.get("content", "")
        )