import pytest
from typing import List
from llamda_fn.functions import LlamdaFunctions
from llamda_fn.llms.api_types import LLMessage, LLCompletion, LLMessageMeta, LlToolCall


class MockLLManager:
    def __init__(self):
        self.tools = []
        self.call_count = 0

    def chat_completion(self, messages):
        self.call_count += 1  # Correctly increment the call count

        last_message = messages[-1]
        if last_message.content == "Just give me a simple response.":
            return LLCompletion(
                message=LLMessage(
                    id="chatcmpl-simple",
                    role="assistant",
                    content="Here's a simple response without using any functions.",
                    tool_calls=None,
                    meta=LLMessageMeta(
                        choice={"finish_reason": "stop", "index": 0, "logprobs": None},
                        completion={
                            "id": "chatcmpl-simple",
                            "created": 1677652290,
                            "model": "gpt-3.5-turbo-0613",
                            "object": "chat.completion",
                            "system_fingerprint": None,
                            "usage": None,
                        },
                    ),
                )
            )
        elif self.call_count == 1:
            return LLCompletion(
                message=LLMessage(
                    id="chatcmpl-1",
                    role="assistant",
                    content="Sure, I can help you with that. Let's start by greeting someone.",
                    tool_calls=[
                        LlToolCall(
                            id="call_1", name="greet", arguments='{"name": "Alice"}'
                        )
                    ],
                    meta=LLMessageMeta(
                        choice={
                            "finish_reason": "tool_calls",
                            "index": 0,
                            "logprobs": None,
                        },
                        completion={
                            "id": "chatcmpl-1",
                            "created": 1677652288,
                            "model": "gpt-3.5-turbo-0613",
                            "object": "chat.completion",
                            "system_fingerprint": None,
                            "usage": None,
                        },
                    ),
                )
            )
        elif self.call_count == 2:
            return LLCompletion(
                message=LLMessage(
                    id="chatcmpl-2",
                    role="assistant",
                    content="Great! Now let's do a calculation.",
                    tool_calls=[
                        LlToolCall(
                            id="call_2", name="add", arguments='{"x": 5, "y": 3}'
                        )
                    ],
                    meta=LLMessageMeta(
                        choice={
                            "finish_reason": "tool_calls",
                            "index": 0,
                            "logprobs": None,
                        },
                        completion={
                            "id": "chatcmpl-2",
                            "created": 1677652289,
                            "model": "gpt-3.5-turbo-0613",
                            "object": "chat.completion",
                            "system_fingerprint": None,
                            "usage": None,
                        },
                    ),
                )
            )
        else:
            raise ValueError(
                f"Unexpected call to chat_completion (call_count: {self.call_count})"
            )


@pytest.fixture
def mock_ll_manager():
    return MockLLManager()


@pytest.fixture
def llamda_functions():
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
def sample_messages():
    return [
        LLMessage(role="system", content="You are a helpful assistant."),
        LLMessage(role="user", content="Hello, how are you?"),
    ]
