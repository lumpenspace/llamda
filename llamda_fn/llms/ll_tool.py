"""Classes which transforms LLamdafunctions into OAI compatible tools"""

from functools import cached_property

from pydantic import BaseModel


from .oai_api_types import OaiToolCall, OaiToolMessage

__all__: list[str] = ["LLToolCall", "LLToolResponse"]


class LLToolCall(BaseModel):
    """
    Describes a function call from the LLM
    """

    tool_call_id: str
    name: str
    arguments: str

    @classmethod
    def from_call(cls, call: OaiToolCall) -> "LLToolCall":
        """Gets data from the Openai Tool Call"""
        return cls(
            tool_call_id=call.id,
            name=call.function.name,
            arguments=call.function.arguments,
        )


class LLToolResponse(BaseModel):
    """Describes the result of executing a LLamda Function
    and provides the means of turning it into an API message"""

    tool_call_id: str
    success: bool
    result: str = ""

    @cached_property
    def oai(self) -> OaiToolMessage:
        """The OpenAI-ready version of tool message."""
        return OaiToolMessage(
            tool_call_id=self.tool_call_id, role="tool", content=self.result
        )
