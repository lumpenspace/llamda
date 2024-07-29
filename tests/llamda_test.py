import pytest
from typing import List, Dict, Any
from llamda_fn.functions import LlamdaFunctions
from llamda_fn.llms.api_types import LLMessage, LlToolCall, ToolResponse
from llamda_fn.llms.llm_manager import LLManager
from llamda_fn.functions import LlamdaFunctions


class MockLLManager(LLManager):
    def __init__(self, responses: List[Dict[str, Any]]):
        self.responses = responses
        self.call_count = 0

    def chat_completion(self, messages: List[LLMessage], **kwargs: Any):
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


@pytest.fixture
def llamda_functions():
    llamda = LlamdaFunctions()

    @llamda.llamdafy()
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    @llamda.llamdafy()
    def calculate(operation: str, x: int, y: int) -> int:
        if operation == "add":
            return x + y
        elif operation == "subtract":
            return x - y
        else:
            raise ValueError(f"Unknown operation: {operation}")

    return llamda


@pytest.fixture
def mock_ll_manager():
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "Sure, I can help you with that. Let's use the greet function.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "greet", "arguments": '{"name": "Alice"}'},
                    }
                ],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": "Now, let's perform a calculation.",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"operation": "add", "x": 5, "y": 3}',
                        },
                    }
                ],
            }
        },
        {
            "message": {
                "role": "assistant",
                "content": "Here's a response without any function calls.",
            }
        },
    ]
    return MockLLManager(responses)


def test_llamda_function_execution_with_tool_calls(llamda_functions, mock_ll_manager):
    # Simulate a conversation with tool calls
    messages = [
        LLMessage(
            role="user", content="Can you greet someone and then do a calculation?"
        )
    ]

    # First interaction: Greet function
    completion = mock_ll_manager.chat_completion(messages)
    assert completion["message"]["tool_calls"] is not None
    tool_call = LlToolCall(**completion["message"]["tool_calls"][0])
    result = llamda_functions.execute_function(tool_call)
    assert isinstance(result, ToolResponse)
    assert result.result == '"Hello, Alice!"'

    # Second interaction: Calculate function
    messages.append(
        LLMessage(role="assistant", content=completion["message"]["content"])
    )
    messages.append(LLMessage(role="tool", content=result.result))
    completion = mock_ll_manager.chat_completion(messages)
    assert completion["message"]["tool_calls"] is not None
    tool_call = LlToolCall(**completion["message"]["tool_calls"][0])
    result = llamda_functions.execute_function(tool_call)
    assert isinstance(result, ToolResponse)
    assert result.result == "8"


def test_llamda_function_execution_without_tool_calls(
    llamda_functions, mock_ll_manager
):
    # Simulate a conversation without tool calls
    messages = [LLMessage(role="user", content="Just give me a simple response.")]

    completion = mock_ll_manager.chat_completion(messages)
    assert "tool_calls" not in completion["message"]

    # Verify that no functions were executed
    assert (
        mock_ll_manager.call_count == 3
    )  # This should be the third response in our mock


if __name__ == "__main__":
    pytest.main([__file__])
