from typing import TypeVar
from unittest.mock import Mock, patch

import pytest

from llamda_fn.llamda import Llamda
from llamda_fn.llms.ll_exchange import LLExchange
from llamda_fn.llms.ll_message import LLMessage
from llamda_fn.llms.ll_tool import LLToolCall, LLToolResponse

T = TypeVar("T")


@pytest.fixture
def mock_ll_manager():
    with patch("llamda_fn.llamda.LLManager") as mock:
        yield mock.return_value


@pytest.fixture
def llamda_instance(mock_ll_manager):
    return Llamda(system="Test system", max_consecutive_tool_calls=3)


def test_llamda_initialization(llamda_instance):
    assert llamda_instance.max_consecutive_tool_calls == 3
    assert isinstance(llamda_instance.exchange, LLExchange)
    assert llamda_instance.exchange[0].content == "Test system"


def test_llamda_fy_decorator(llamda_instance):
    @llamda_instance.fy()
    def test_function() -> str:
        return "Hello, World!"

    assert any(
        tool["function"]["name"] == "test_function" for tool in llamda_instance.tools
    )


def test_llamda_tools_property(llamda_instance):
    assert llamda_instance.tools == llamda_instance.functions.spec


def test_llamda_run_without_tool_calls(llamda_instance, mock_ll_manager):
    mock_ll_manager.query.return_value = LLMessage(
        role="assistant", content="Test response"
    )

    result = llamda_instance.run()

    assert isinstance(result, LLMessage)
    assert result.content == "Test response"
    mock_ll_manager.query.assert_called_once()


def test_llamda_run_with_tool_calls(llamda_instance, mock_ll_manager):
    tool_call = LLToolCall(id="1", name="test_function", arguments="{}")
    mock_ll_manager.query.side_effect = [
        LLMessage(role="assistant", content="", tool_calls=[tool_call]),
        LLMessage(role="assistant", content="Final response"),
    ]

    with patch.object(
        llamda_instance.functions,
        "execute_function",
        return_value=LLToolResponse(id="1", result="Tool result"),
    ):
        result = llamda_instance.run()

    assert isinstance(result, LLMessage)
    assert result.content == "Final response"
    assert mock_ll_manager.query.call_count == 2


def test_llamda_call_method(llamda_instance, mock_ll_manager):
    mock_ll_manager.query.return_value = LLMessage(
        role="assistant", content="Response to user input"
    )

    result = llamda_instance("User input")

    assert isinstance(result, LLMessage)
    assert result.content == "Response to user input"
    assert llamda_instance.exchange[-2].content == "User input"


@pytest.mark.asyncio
async def test_llamda_handle_tool_calls(llamda_instance):
    tool_calls: list[LLToolCall] = [
        LLToolCall(id="1", name="func1", arguments="{}"),
        LLToolCall(id="2", name="func2", arguments="{}"),
    ]

    with patch.object(
        llamda_instance.functions,
        "execute_function",
        return_value=LLToolResponse(id="1", result="Tool result"),
    ), patch.object(llamda_instance, "run") as mock_run:
        llamda_instance._handle_tool_calls(tool_calls)

    assert len(llamda_instance.exchange) == 3  # Two tool responses
    assert llamda_instance.exchange[1].role == "tool"

    mock_run.assert_called_once()


def test_llamda_process_tool_call(llamda_instance):
    tool_call = LLToolCall(id="1", name="test_function", arguments="{}")
    mock_log = Mock()

    with patch.object(
        llamda_instance.functions,
        "execute_function",
        return_value=LLToolResponse(id="1", result="Tool result"),
    ) as mock_execute:
        result = llamda_instance._process_tool_call(tool_call, mock_log)

    assert isinstance(result, LLToolResponse)
    assert result.result == "Tool result"
    mock_execute.assert_called_once_with(tool_call=tool_call)
    mock_log.assert_called_once()


# Add more tests as needed for edge cases and specific scenarios
