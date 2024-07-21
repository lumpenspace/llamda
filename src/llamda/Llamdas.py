"""
# Llamdas

Collection of LlamdaFunction instances.
"""

import json
from typing import Any, Dict, Generic, List, Mapping, TypeVar

from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel, Field, field_validator
from pydantic.errors import PydanticInvalidForJsonSchema, PydanticUserError

from .llamda_function import LlamdaFunction
from .response_types import ExecutionResponse, ToolCallResult

T = TypeVar("T")


class Llamdas(BaseModel, Generic[T]):
    """
    A collection of LlamdaFunction instances.
    """

    class Config:
        """
        Config for the Llamdas class.
        """

        arbitrary_types_allowed = True

    llamdas: Mapping[str, LlamdaFunction[Any]] = Field(
        ...,
        json_schema_extra={
            "description": "Mapping of function names to LlamdaFunction instances"
        },
        alias="functions",
    )
    handle_exceptions: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Flag to determine if exceptions should be automatically handled"
        },
    )

    @property
    def functions(self) -> Dict[str, LlamdaFunction[Any]]:
        """
        Get the functions in the collection.
        """
        return dict(self.llamdas)

    @field_validator("llamdas", mode="before")
    @classmethod
    def validate_functions(
        cls, v: Dict[str, LlamdaFunction[Any]] | List[LlamdaFunction[Any]]
    ) -> Dict[str, LlamdaFunction[Any]]:
        """
        Transforms a list of LlamdaFunction instances or a dictionary of function names
        to LlamdaFunction instances.
        """

        if isinstance(v, List):
            func_names: list[str] = [func.__name__ for func in v]
            if len(func_names) != len(set(func_names)):
                raise ValueError("Function names must be unique")

        return {func.__name__: func for func in v} if isinstance(v, List) else v

    def to_openai_tools(self) -> list[dict[str, str | Any]]:
        """
        Transforms the LlamdaFunction instances in the collection into OpenAI tool
        definitions.
        """
        return [
            {"type": "function", "function": func.to_schema()}
            for func in self.functions.values()
        ]

    def execute(self, message: ChatCompletionMessage) -> ExecutionResponse[T]:
        """
        Executes the tool calls in the message, if any.

        Returns:
            ExecutionResponse[T]: The execution response.
        """
        tool_calls: Any | None = getattr(message, "tool_calls", None)
        if tool_calls is None:
            return ExecutionResponse(results={})

        results: Dict[str, Any] = {}
        for call in tool_calls:
            func_name: str = call["function"]["name"]

            if func_name in self.functions:
                func: LlamdaFunction[Any] = self.functions[func_name]
                try:
                    args: Dict[str, Any] = json.loads(call["function"]["arguments"])
                    result: ToolCallResult[T] = func(
                        **args, handle_exceptions=self.handle_exceptions
                    )

                    results[call["id"]] = {
                        "function_name": func_name,
                        "result": result,
                    }
                except (
                    PydanticUserError,
                    PydanticInvalidForJsonSchema,
                    ValueError,
                    TypeError,
                ) as e:
                    if self.handle_exceptions:
                        results[call["id"]] = {
                            "function_name": func_name,
                            "result": ToolCallResult(
                                success=False,
                                exception=str(e),
                                result=None,
                            ),
                        }
            else:
                raise ValueError(f"Function {func_name} not found")
        return ExecutionResponse(results=results)
