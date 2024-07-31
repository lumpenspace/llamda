"""Classes related to Messages"""

import uuid
from functools import cached_property
from typing import Any, Self, Sequence
from pydantic import BaseModel, Field

from llamda_fn.llms.ll_tool import LLToolCall, LLToolResponse
from .oai_api_types import (
    OaiAssistantMessage,
    OaiMessage,
    OaiRole,
    OaiSystemMessage,
    OaiToolMessage,
    OaiUserMessage,
    OaiCompletion,
    OaiResponseMessage,
)

__all__: list[str] = ["LLMessageMeta", "LLMessage"]


class LLMessageMeta(BaseModel):
    """Metadata for messages"""

    choice: dict[str, Any] | None = Field(exclude=True)
    completion: dict[str, Any] | None = Field(exclude=True)


class LLMessage(BaseModel):
    """Represents a message in an Exchange"""

    id: str = Field(default_factory=uuid.uuid4)
    role: OaiRole = "user"
    content: str = ""
    name: str | None = None
    tool_calls: Sequence[LLToolCall] | None = None
    meta: LLMessageMeta | None = None

    @cached_property
    def oai_props(self) -> OaiMessage:
        """Rerutns the OpenAI API verison of a message"""
        kwargs: dict[Any, Any] = {}
        print(f"kwargs: {kwargs}", self.role, self.model_dump())
        role = self.role
        if self.name:
            kwargs["name"] = self.name
        match role:
            case "user":
                return OaiUserMessage(
                    role="user",
                    content=self.content,
                    **kwargs,
                )
            case "system":
                return OaiSystemMessage(
                    role="system",
                    content=self.content,
                    **kwargs,
                )
            case "assistant":
                if self.tool_calls:
                    kwargs["tool_calls"] = [
                        tool_call.model_dump() for tool_call in self.tool_calls or []
                    ]
                return OaiAssistantMessage(
                    role="assistant",
                    content=self.content,
                    **kwargs,
                )
            case "tool":
                return OaiToolMessage(**kwargs, role="tool")

    @classmethod
    def make_oai_message(cls, **kwargs: Any) -> OaiMessage:
        """
        Creates an OpenAI-compatible message from the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments to create the LLMessage.

        Returns:
            OaiMessage: An OpenAI-compatible message object.
        """
        return cls(**kwargs).oai_props

    @classmethod
    def from_tool_response(cls, response: LLToolResponse) -> Self:
        """A message containing the result"""
        return cls(id=response.id, role="tool", content=response.result)

    @classmethod
    def from_completion(cls, completion: OaiCompletion) -> Self:
        """Creates a Message from the first choice of an OpenAI-type completion request"""

        choice = completion.choices[0]
        message: OaiResponseMessage = choice.message

        tool_calls: list[LLToolCall] = (
            [LLToolCall.from_call(tc) for tc in message.tool_calls]
            if message.tool_calls
            else []
        )

        return cls(
            id=completion.id,
            meta=LLMessageMeta(
                choice=choice.model_dump(exclude={"message", "tool_call"}),
                completion=completion.model_dump(exclude={"choices"}),
            ),
            role=message.role,
            content=message.content or "",
            tool_calls=tool_calls,
        )
