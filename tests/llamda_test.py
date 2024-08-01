from typing import Any
from llamda_fn.functions import LlamdaFunctions
from llamda_fn.llms.ll_tool import LLToolResponse, LLToolCall
from llamda_fn.llms.ll_message import LLMessage


def test_llamda_function_execution_with_tool_calls(
    llamda_functions: LlamdaFunctions, mock_ll_manager: Any
):
    messages: list[LLMessage] = [
        LLMessage(
            role="user", content="Can you greet someone and then do a calculation?"
        )
    ]

    # First interaction: Greet function
    completion = mock_ll_manager.chat_completion(messages)
    assert completion.tool_calls is not None
    tool_call = LLToolCall(**completion.tool_calls[0].model_dump())
    response: LLToolResponse = llamda_functions.execute_function(tool_call)
    assert isinstance(response, LLToolResponse)
    assert response.result == '"Hello, Alice!"'

    # Second interaction: Calculate function
    messages.append(completion)
    messages.append(LLMessage.from_tool_response(response))
    completion = mock_ll_manager.chat_completion(messages)
    assert completion.tool_calls is not None
    tool_call = LLToolCall(**completion.tool_calls[0].model_dump())
    result = llamda_functions.execute_function(tool_call)
    assert isinstance(result, LLToolResponse)
    assert result.result == "8"

    # Verify the call count
    assert mock_ll_manager.call_count == 2


def test_llamda_function_execution_without_tool_calls(
    llamda_functions: LlamdaFunctions, mock_ll_manager: Any
):
    messages = [LLMessage(role="user", content="Just give me a simple response.")]
    completion = mock_ll_manager.chat_completion(messages)
    assert completion.tool_calls is None or len(completion.tool_calls) == 0

    # Verify that the content is as expected
    assert completion.content == "Here's a simple response without using any functions."

    # Verify that this is the first call to the mock_ll_manager
    assert mock_ll_manager.call_count == 1

    # No need to reset call_count here, as it should be reset in the fixture for each test
