import uuid
from functools import cached_property
from typing import Any, Literal, Self, List

from openai.types.chat import ChatCompletion as OaiCompletion
from openai.types.chat import ChatCompletionToolParam as OaiToolParam
from openai.types.chat import ChatCompletionMessage as OaiResponseMessage
from openai.types.chat import ChatCompletionMessageParam as OaiRequestMessage
from openai.types.chat import ChatCompletionMessageToolCall as OaiToolCall
from openai.types.chat import ChatCompletionFunctionCallOptionParam as OaiToolFunction
from pydantic import BaseModel, Field

Role = Literal["user", "system", "assistant", "tool"]


class LlToolCall(BaseModel):
    id: str
    name: str
    arguments: str

    @classmethod
    def from_oai_tool_call(cls, call: OaiToolCall) -> Self:
        return cls(
            id=call.id,
            name=call.function.name,
            arguments=call.function.arguments,
        )


class ToolResponse(BaseModel):
    id: str
    name: str
    arguments: str
    _result: str

    def __init__(self, result: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._result = result

    @cached_property
    def result(self) -> str:
        if isinstance(self._result, BaseModel):
            return self._result.model_dump_json()
        else:
            return self._result


class LLMessageMeta(BaseModel):
    choice: dict[str, Any] | None = Field(exclude=True)
    completion: dict[str, Any] | None = Field(exclude=True)


class LLMessage(BaseModel):
    id: str = Field(default_factory=uuid.uuid4)
    role: Role
    content: str
    name: str | None = None
    tool_calls: List[LlToolCall] | None = None
    meta: LLMessageMeta | None = None

    @classmethod
    def from_execution(cls, execution: ToolResponse) -> Self:
        return cls(
            role="tool",
            id=execution.id,
            name=execution.name,
            content=execution.result,
        )


class LLUserMessage(LLMessage):
    role: Role = "user"


class LLSystemMessage(LLMessage):
    role: Role = "system"


class LLAssistantMessage(LLMessage):
    role: Role = "assistant"


class LLToolMessage(LLMessage):
    role: Role = "tool"


class LLCompletion(BaseModel):
    message: LLMessage
    meta: LLMessageMeta | None = None

    @classmethod
    def from_completion(cls, completion: OaiCompletion) -> Self:
        choice = completion.choices[0]
        message = choice.message
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                LlToolCall.from_oai_tool_call(tc) for tc in message.tool_calls
            ]

        return cls(
            message=LLMessage(
                id=completion.id,
                meta=LLMessageMeta(
                    choice=choice.model_dump(exclude={"message"}),
                    completion=completion.model_dump(exclude={"choices"}),
                ),
                role=message.role,
                content=message.content or "",
                tool_calls=tool_calls,
            )
        )


class OaiRequest(BaseModel):
    messages: list[OaiRequestMessage]
    tools: list[OaiToolParam]


__all__ = [
    "LLMessage",
    "LLToolMessage",
    "LLUserMessage",
    "LLSystemMessage",
    "LLAssistantMessage",
    "LLCompletion",
    "OaiCompletion",
    "OaiToolParam",
    "OaiToolFunction",
    "OaiResponseMessage",
    "OaiToolCall",
]
