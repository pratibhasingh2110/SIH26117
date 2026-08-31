from runtime.actions import ToolCall, FinalAnswer


tool_call = ToolCall(
    tool_name="calculator",
    arguments={
        "a": 10,
        "b": 20
    }
)

final_answer = FinalAnswer(
    content="The answer is 30."
)

print(tool_call)
print(final_answer)