"""Classes related to Messages"""

import json
from re import S
import uuid
from functools import cached_property
from typing import Any, Self, Sequence
from pydantic import BaseModel, Field
from rich.console import Console

from llamda_fn.llms.ll_tool import LLToolCall, LLToolResponse
from .oai_api_types import (
    OaiMessage,
    OaiRole,
    OaiCompletion,
    OaiResponseMessage,
)


__all__: list[str] = ["LLMessageMeta", "LLMessage"]

console = Console()


class LLMessageMeta(BaseModel):
    """Metadata for messages"""

    choice: dict[str, Any] | None = Field(exclude=True)
    completion: dict[str, Any] | None = Field(exclude=True)


class LLMessage(BaseModel):
    """Represents a message in an Exchange"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: OaiRole = "user"
    content: str = ""
    name: str | None = None
    tool_calls: Sequence[LLToolCall] | None = None
    meta: LLMessageMeta | None = None

    @cached_property
    def oai_props(self) -> dict[str, Any]:
        """Rerutns the OpenAI API verison of a message"""
        kwargs: dict[Any, Any] = {}
        role = self.role
        if self.name:
            kwargs["name"] = self.name
        match role:
            case "user":
                return {
                    "role": "user",
                    "content": self.content,
                    **kwargs,
                }
            case "system":
                return {
                    "role": "system",
                    "content": self.content,
                    **kwargs,
                }
            case "assistant":
                if self.tool_calls:
                    console.log(f"tool_calls: {self.tool_calls}")
                    kwargs["tool_calls"] = [
                        tool_call.oai for tool_call in self.tool_calls or []
                    ]
                return {
                    "role": "assistant",
                    "content": self.content,
                    **kwargs,
                }
            case "tool":
                return {
                    "tool_call_id": self.id,
                    "role": "tool",
                    "content": self.content,
                    **kwargs,
                }
            case _:
                raise ValueError(f"Invalid role: {role}")

    @classmethod
    def make_oai_message(cls, **kwargs: Any) -> dict[str, Any]:
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
        return cls(
            id=response.tool_call_id,
            content=response.result,
            role="tool",
        )

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
