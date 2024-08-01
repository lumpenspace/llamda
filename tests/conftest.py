import pytest
from typing import Any, Sequence, List
from unittest.mock import Mock
from llamda_fn.functions import LlamdaFunctions
from llamda_fn.functions.process_fields import JsonDict
from llamda_fn.llms.ll_message import LLMessage, LLMessageMeta
from llamda_fn.llms.ll_tool import LLToolCall, LLToolResponse
from llamda_fn.llamda import Llamda
from llamda_fn.functions.llamda_function import LlamdaFunction
from llamda_fn.llms.ll_manager import LLManager
from pydantic import Field


class MockLLManager(LLManager):
    call_count: int = 0
    api_config: JsonDict = Field(default_factory=dict)
    llm_name: str = "gpt-4o-mini"
    api: Any = None
    tools: list[LlamdaFunction[Any]] = Field(default_factory=list)

    def __init__(self):
        super().__init__(llm_name="gpt-4o-mini")

    def query(
        self,
        messages: Sequence[LLMessage],
        llm_name: str | None = None,
        tools: Any = None,
    ) -> LLMessage:
        self.call_count += 1

        last_message = messages[-1]
        if last_message.content == "Just give me a simple response.":
            return LLMessage(
                role="assistant",
                content="Here's a simple response without using any functions.",
            )
        elif self.call_count == 1:
            return LLMessage(
                role="assistant",
                content="Sure, I can help you with that. Let's start by greeting someone.",
                tool_calls=[
                    LLToolCall(
                        tool_call_id="call_1",
                        name="greet",
                        arguments='{"name": "Alice"}',
                    )
                ],
            )
        elif self.call_count == 2:
            return LLMessage(
                role="assistant",
                content="Great! Now let's do a calculation.",
                tool_calls=[
                    LLToolCall(
                        tool_call_id="call_2", name="add", arguments='{"x": 5, "y": 3}'
                    )
                ],
            )
        else:
            return LLMessage(
                role="assistant",
                content="I don't have any more actions to perform.",
            )

    def chat_completion(self, messages: Sequence[LLMessage]) -> LLMessage:
        return self.query(messages)


@pytest.fixture
def mock_ll_manager() -> MockLLManager:
    return MockLLManager()


@pytest.fixture
def llamda_functions() -> LlamdaFunctions:
    llamda = LlamdaFunctions()

    @llamda.llamdafy()
    def greet(name: str) -> str:
        """Greet a person by name."""
        return f"Hello, {name}!"

    @llamda.llamdafy()
    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    return llamda


@pytest.fixture
def llamda_instance(
    llamda_functions: LlamdaFunctions, mock_ll_manager: MockLLManager
) -> Llamda:
    llamda = Llamda(llm_name="mock-llm")
    llamda.api = mock_ll_manager
    llamda.functions = llamda_functions
    llamda.console = Mock()
    llamda.console.msg = Mock()
    llamda.console.msg.update_tool = Mock()

    # Mock the _process_tool_call method
    llamda._process_tool_call = Mock(  # type: ignore
        side_effect=lambda tool_call: LLToolResponse(
            tool_call_id=tool_call.tool_call_id,
            result="Mocked tool result",
            success=True,
        )
    )

    return llamda
