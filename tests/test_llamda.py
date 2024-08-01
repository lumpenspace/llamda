import json
from typing import Generator, TypeVar
from unittest.mock import Mock, patch

from json import dumps
import pytest

from llamda_fn.llamda import Llamda
from llamda_fn.llms.ll_exchange import LLExchange
from llamda_fn.llms.ll_message import LLMessage
from llamda_fn.llms.ll_tool import LLToolCall, LLToolResponse
from llamda_fn.llms.oai_api_types import OaiRole

T = TypeVar("T")


@pytest.fixture
def mock_ll_manager() -> Generator[Mock, None, None]:
    with patch("llamda_fn.llms.LLManager") as mock:
        yield mock.return_value


@pytest.fixture
def llamda_instance(mock_ll_manager: Mock) -> Llamda:
    llamda = Llamda(system="Test system", max_consecutive_tool_calls=3)
    llamda.api = mock_ll_manager
    llamda.functions = Mock()
    llamda.console = Mock()
    llamda.__call__ = Mock(
        return_value=LLMessage(
            role="user",
            content=LLToolResponse(
                tool_call_id="1",
                result="Hello, World!",
                success=True,
            ).json(),
        )
    )
    return llamda


def test_llamda_initialization(llamda_instance: Llamda) -> None:
    assert llamda_instance.max_consecutive_tool_calls == 3
    assert isinstance(llamda_instance.exchange, LLExchange)
    assert llamda_instance.exchange[0].content == "Test system"


def test_llamda_fy_decorator(llamda_instance: Llamda) -> None:
    mock_llamdafy = Mock(return_value=lambda: "Hello, World!")
    llamda_instance.functions.llamdafy = mock_llamdafy
    llamda_instance.functions.spec = [
        {"type": "function", "function": {"name": "test_function"}}
    ]

    @llamda_instance.fy
    def test_function() -> str:
        return "Hello, World!"

    mock_llamdafy.assert_called_once()
    assert any(
        tool["function"]["name"] == "test_function" for tool in llamda_instance.tools
    )
    assert test_function() == "Hello, World!"


def test_llamda_tools_property(llamda_instance: Llamda) -> None:
    assert llamda_instance.tools == llamda_instance.functions.spec


def test_llamda_run_without_tool_calls(
    llamda_instance: Llamda, mock_ll_manager: Mock
) -> None:
    mock_ll_manager.query.return_value = LLMessage(
        role="assistant", content="Test response"
    )

    result = llamda_instance.run()

    assert isinstance(result, LLMessage)
    assert result.content == "Test response"
    mock_ll_manager.query.assert_called_once()


def test_llamda_run_with_tool_calls(
    llamda_instance: Llamda, mock_ll_manager: Mock
) -> None:
    tool_call = LLToolCall(tool_call_id="1", name="test_function", arguments="{}")
    mock_ll_manager.query.side_effect = [
        LLMessage(role="assistant", content="", tool_calls=[tool_call]),
        LLMessage(role="assistant", content="Final response"),
    ]

    with patch.object(
        llamda_instance.functions,
        "execute_function",
        return_value=LLToolResponse(
            tool_call_id="1",
            result="Tool result",
            success=True,
        ),
    ):
        result = llamda_instance.run()

    assert isinstance(result, LLMessage)
    assert result.content == "Final response"
    assert mock_ll_manager.query.call_count == 2


def test_llamda_call_method(llamda_instance: Llamda, mock_ll_manager: Mock) -> None:
    mock_ll_manager.query.return_value = LLMessage(
        role="assistant", content="Response to user input"
    )

    result = llamda_instance("User input")
    print(dumps(llamda_instance.exchange.to_dict(), indent=2))
    assert isinstance(result, LLMessage)
    assert result.content == "Response to user input"
    assert llamda_instance.exchange[-2].content == "User input"
    assert llamda_instance.exchange[-1].content == "Response to user input"


def test_llamda_handle_tool_calls(llamda_instance: Llamda) -> None:
    llamda_instance.exchange.clear()

    tool_calls = [
        LLToolCall(tool_call_id="1", name="func1", arguments="{}"),
        LLToolCall(tool_call_id="2", name="func2", arguments="{}"),
    ]

    with patch.object(
        llamda_instance.functions,
        "execute_function",
        return_value=LLToolResponse(
            tool_call_id="1",
            result="Tool result",
            success=True,
        ),
    ) as mock_execute_function, patch.object(
        llamda_instance.api, "query"
    ) as mock_query:
        mock_query.side_effect = [
            LLMessage(role="assistant", content="", tool_calls=tool_calls),
            LLMessage(role="assistant", content="Final response"),
        ]
        result = llamda_instance.run()

    assert isinstance(result, LLMessage)
    assert result.content == "Final response"
    assert len(llamda_instance.exchange) == 4
    assert llamda_instance.exchange[1].role == "tool"
    assert json.loads(llamda_instance.exchange[1].content) == {
        "tool_call_id": "1",
        "success": True,
        "result": "Tool result",
    }

    assert llamda_instance.exchange[3].content == "Final response"

    mock_execute_function.assert_called()


def test_handle_tool_calls(llamda_instance: Llamda) -> None:
    llamda_instance.console = Mock()
    llamda_instance.console.msg = Mock()
    llamda_instance.console.msg.update_tool = Mock()
    llamda_instance.exchange = Mock()
    llamda_instance.exchange.append = Mock()

    with patch.object(llamda_instance, "run") as mock_run, patch.object(
        llamda_instance, "_handle_tool_calls"
    ) as mock_handle_tool_calls:
        mock_run.return_value = LLMessage(role="assistant", content="Final response")
        result = llamda_instance("Test input")

    assert isinstance(result, LLMessage)
    assert result.content == "Final response"
    assert llamda_instance.exchange.append.call_count == 1
    assert llamda_instance.console.msg.update_tool.call_count == 0
    mock_handle_tool_calls.assert_not_called()


def test_llamda_run(llamda_instance: Llamda) -> None:
    with patch.object(
        llamda_instance,
        "run",
        return_value=LLMessage(
            role="assistant",
            content="Here's a simple response without using any functions.",
        ),
    ):
        result = llamda_instance("Just give me a simple response.")
    assert isinstance(result, LLMessage)
    assert result.content == "Here's a simple response without using any functions."


def test_llamda_tool_calls(llamda_instance: Llamda) -> None:
    mock_messages = [
        LLMessage(
            role="assistant",
            content="",
            id="1",
            tool_calls=[LLToolCall(tool_call_id="1", name="greet", arguments="{}")],
        ),
        LLMessage.from_tool_response(
            response=LLToolResponse(
                tool_call_id="1",
                result="Hello, World!",
                success=True,
            ),
        ),
    ]

    llamda_instance.api.query = Mock(side_effect=mock_messages)

    result: LLMessage = LLMessage(
        id="1",
        role="assistant",
        content="",
        tool_calls=[LLToolCall(tool_call_id="2", name="add", arguments="{}")],
    )
    assert isinstance(result, LLMessage)
    assert result.content == ""


def mock_get_role_emoji(role: OaiRole) -> str:
    role_emojis = {
        "system": "🖥️",
        "user": "👤",
        "assistant": "🤖",
        "tool": "🛠️",
    }
    return role_emojis.get(role, "❓")


def test_llamda_with_mocked_emoji(llamda_instance: Llamda) -> None:
    assert llamda_instance.max_consecutive_tool_calls == 3
    assert isinstance(llamda_instance.exchange, LLExchange)
    assert llamda_instance.exchange[0].content == "Test system"
