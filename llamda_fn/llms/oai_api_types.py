"""Typer for communicating with the openai api"""

from typing import Literal
from openai.types.chat import (
    ChatCompletionMessageToolCall as OaiToolCall,
    ChatCompletionMessageParam as OaiMessage,
    ChatCompletionAssistantMessageParam as OaiAssistantMessage,
    ChatCompletionUserMessageParam as OaiUserMessage,
    ChatCompletionSystemMessageParam as OaiSystemMessage,
    ChatCompletionToolMessageParam as OaiToolMessage,
    ChatCompletionToolParam as OaiToolSpec,
    ChatCompletion as OaiCompletion,
    ChatCompletionMessage as OaiResponseMessage,
)

from openai import OpenAI as OaiClient


__all__: list[str] = [
    "OaiCompletion",
    "OaiSystemMessage",
    "OaiToolCall",
    "OaiToolMessage",
    "OaiToolSpec",
    "OaiAssistantMessage",
    "OaiUserMessage",
    "OaiResponseMessage",
    "OaiMessage",
    "OaiRole",
    "OaiRoleMessageMap",
    "OaiException",
    "OaiClient",
]

type OaiRole = Literal["user"] | Literal["system"] | Literal["assistant"] | Literal[
    "tool"
]


class OaiException(BaseException):
    """An exception type for LLM API Responses."""


OaiRoleMessageMap: dict[
    OaiRole, type[OaiUserMessage] | type[OaiSystemMessage] | type[OaiAssistantMessage]
] = {
    "user": OaiUserMessage,
    "system": OaiSystemMessage,
    "assistant": OaiAssistantMessage,
}
