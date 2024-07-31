"""Classes which transforms LLamdafunctions into OAI compatible tools"""

from functools import cached_property
from typing import Any

from pydantic import BaseModel

from .oai_api_types import OaiToolCall, OaiToolMessage

__all__: list[str] = ["LLToolCall", "LLToolResponse"]


class LLToolCall(BaseModel):
    """
    Describes a function call from the LLM
    """

    id: str
    name: str
    arguments: str

    @classmethod
    def from_call(cls, call: OaiToolCall) -> "LLToolCall":
        """Gets data from the Openai Tool Call"""
        return cls(
            id=call.id,
            name=call.function.name,
            arguments=call.function.arguments,
        )


class LLToolResponse(BaseModel):
    """Describes the result of executing a LLamda Function
    and provides the means of turning it into an API message"""

    id: str
    _result: str

    def __init__(self, result: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._result = result

    @cached_property
    def oai(self) -> OaiToolMessage:
        """The OpenAI-ready version of tool message."""
        return OaiToolMessage(tool_call_id=self.id, role="tool", content=self.result)

    @cached_property
    def result(self) -> str:
        """Returns the jsonified result"""
        if isinstance(self._result, BaseModel):
            return self._result.model_dump_json()
        else:
            return self._result
