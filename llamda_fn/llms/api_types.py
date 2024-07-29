import uuid
from functools import cached_property
from typing import Any, Literal, Self

from openai.types.chat import ChatCompletionMessage as OaiMessage
from openai.types.chat import ChatCompletionToolParam as OaiToolParam
from openai.types.chat import ChatCompletion as OaiCompletion
from openai.types.chat import ChatCompletionMessageToolCall as OaiToolCall
from pydantic import BaseModel, Field, field_validator

type Role = Literal["user"] | Literal["system"] | Literal["assistant"] | Literal["tool"]


OaiChoice = OaiMessage | OaiToolCall


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str


class ToolResponse(BaseModel):
    """Tool response"""

    id: str
    name: str
    arguments: str
    _result: str

    def __init__(self, result: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._result = result

    # memoised getter for result json
    @cached_property
    def result(self) -> str:
        """Memoised getter for result json"""
        if isinstance(self._result, BaseModel):
            return self._result.model_dump_json()
        else:
            return self._result


class LLMessageMeta(BaseModel):
    """Meta data for the message"""

    choice: dict[str, Any] | None = Field(exclude=True)
    completion: dict[str, Any] | None = Field(exclude=True)


class LLMessage(BaseModel):
    """Message. Contains a meta field for any additional data
    if the message comes from the api"""

    id: str | None = Field(..., exclude=True)
    role: Role
    content: str
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    meta: LLMessageMeta | None = None

    # add the id if none is set
    @field_validator("id")
    @classmethod
    def add_id(cls, v: str | None) -> str:
        """Add a uuid if none is set"""
        return v or str(uuid.uuid4())


class LLChoice(BaseModel):
    """Choice - one item in the choices list in the completion"""

    message: OaiMessage
    meta: dict[str, Any] | None = Field(exclude=True)

    def __init__(self, message: LLMessage, **kwargs: Any) -> None:
        super().__init__(
            meta=kwargs,
            message=message,
        )


class LLCompletion(BaseModel):
    """Completion - the response from the api"""

    message: LLMessage
    meta: LLMessageMeta | None = None

    @classmethod
    def from_completion(cls, completion: OaiCompletion) -> Self:
        """Create a Message from an OaiCompletion"""
        completion_dict: dict[str, Any] = completion.model_dump()
        choice: LLChoice = LLChoice(**completion_dict.pop("choices")[0])
        if not choice:
            raise ValueError("No choices in completion")

        # check tool calls
        tool_calls: list[ToolCall] = []
        message: OaiMessage = choice.message
        if not message:
            raise ValueError("No message in completion")
        if message.tool_calls:
            tool_calls: list[ToolCall] = [
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
                for tool_call in message.tool_calls
            ]
        return cls(
            message=LLMessage(
                id=completion.id,
                meta=LLMessageMeta(
                    choice=choice.meta,
                    completion=completion_dict,
                ),
                role=message.role,
                content=message.content or "",
                tool_calls=tool_calls,
            )
        )


__all__ = [
    "ToolCall",
    "OaiToolParam",
    "OaiMessage",
    "OaiToolCall",
    "OaiChoice",
    "ToolResponse",
    "LLMessage",
    "LLChoice",
    "LLCompletion",
]
