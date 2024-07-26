"""
Response types for the Llamda library.

The response types are defined here to avoid circular imports.
"""

import json
from typing import Any, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ParameterError(BaseModel):
    """
    A tool call that failed because of a parameter error.
    """

    name: str
    description: str

    def __str__(self) -> str:
        return self.description

    def __repr__(self) -> str:
        return repr(self.description)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParameterError):
            return self.description == other
        return self.description == other.description


class ToolCallResult(BaseModel, Generic[T]):
    """
    A tool call result.
    """

    result: Optional[T] = Field(...)
    success: bool = Field(...)
    parameter_errors: Optional[list[ParameterError]] = Field(default=[])
    exception: Optional[str] = Field(default=None)

    def __str__(self) -> str:
        return str(self.result)

    def __repr__(self) -> str:
        return repr(self.result)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCallResult):
            return self.result == other
        return self.result == other.result


class ExecutionResponseItem(BaseModel, Generic[T]):
    """
    An execution response item.
    it
    """

    function_name: str
    result: ToolCallResult[T]

    def __str__(self) -> str:
        return str(self.result)

    def __repr__(self) -> str:
        return repr(self.result)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExecutionResponseItem):
            return self.result == other
        return self.result == other.result


class ExecutionResponse(BaseModel, Generic[T]):
    """
    An execution response.
    """

    results: Dict[str, ExecutionResponseItem[T]] = Field(default={})

    def to_tool_response(self) -> str:
        """
        Convert the execution response to a tool response.
        """
        tool_responses: list[dict[str, str | ToolCallResult[T]]] = []
        for call_id, result in self.__dict__["results"]:
            content: ToolCallResult[T] = result.result
            tool_response: dict[str, str | ToolCallResult[T] | Any] = {
                "role": "tool",
                "content": content,
                "tool_call_id": call_id,
            }
            tool_responses.append(tool_response)
        return json.dumps(tool_responses)
