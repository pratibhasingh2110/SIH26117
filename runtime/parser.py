from runtime.actions import ToolCall, FinalAnswer


class ResponseParser:

    def parse(self, response):

        message = response["message"]

        tool_calls = message.get("tool_calls")

        if tool_calls:
            tool_call = tool_calls[0]

            return ToolCall(
                tool_name=tool_call["function"]["name"],
                arguments=tool_call["function"]["arguments"],
                raw_message=message
            )

        return FinalAnswer(
            content=message.get("content", "")
        )