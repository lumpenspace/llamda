import json

from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Optional, TypeVar, ParamSpec, Union
from pydantic import BaseModel, ValidationError
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from .llamda_classes import LlamdaFunction, LlamdaPydantic

R = TypeVar("R")
P = ParamSpec("P")


class LlamdaFunctions:
    """
    Main class, produces a decorator for creating Llamda functions and manages their execution.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Union[LlamdaFunction[Any], LlamdaPydantic[Any]]] = {}

    def llamdafy(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[[Callable[P, R]], Union[LlamdaFunction[R], LlamdaPydantic[R]]]:
        """
        Decorator for creating Llamda functions.
        """

        def decorator(
            func: Callable[P, R]
        ) -> Union[LlamdaFunction[R], LlamdaPydantic[R]]:
            func_name: str = name or func.__name__
            func_description: str = description or func.__doc__ or ""

            # Check if the function is expecting a Pydantic model
            sig = signature(func)
            if len(sig.parameters) == 1:
                param = next(iter(sig.parameters.values()))
                if issubclass(param.annotation, BaseModel):
                    llamda_func = LlamdaPydantic.create(
                        func_name, param.annotation, func_description, func
                    )
                    self._tools[func_name] = llamda_func
                    return llamda_func

            # If not a Pydantic model, treat it as a regular function
            fields: Dict[str, tuple[type, Any]] = {}
            for param_name, param in sig.parameters.items():
                field_type = (
                    param.annotation if param.annotation != Parameter.empty else Any
                )
                field_default = (
                    param.default if param.default != Parameter.empty else ...
                )
                fields[param_name] = (field_type, field_default)

            llamda_func = LlamdaFunction.create(
                func_name, fields, func_description, func
            )
            self._tools[func_name] = llamda_func
            return llamda_func

        return decorator

    @property
    def tools(self) -> Dict[str, Union[LlamdaFunction[Any], LlamdaPydantic[Any]]]:
        """
        Get the tools.
        """
        return self._tools

    def prepare_tools(
        self, tool_names: Optional[List[str]] = None
    ) -> List[ChatCompletionToolParam]:
        """
        Prepare the tools for the OpenAI API.
        """
        tools = []
        if tool_names is None:
            tool_names = list(self._tools.keys())

        for name in tool_names:
            if name in self._tools:
                tool_schema = self._tools[name].to_schema()
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_schema["title"],
                            "description": tool_schema["description"],
                            "parameters": {
                                "type": "object",
                                "properties": tool_schema["properties"],
                                "required": tool_schema.get("required", []),
                            },
                        },
                    }
                )
        return tools

    def execute_function(
        self, function_name: str, function_args: str
    ) -> Dict[str, str]:
        """
        Execute a function and return the result or error message.
        """
        if function_name in self._tools:
            try:
                parsed_args = json.loads(function_args)
                result = self._tools[function_name].run(**parsed_args)
                return {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(result),
                }
            except ValidationError as e:
                return {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(
                        {"error": f"Error: Validation failed - {str(e)}"}
                    ),
                }
            except Exception as e:
                return {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps({"error": f"Error: {str(e)}"}),
                }
        else:
            return {
                "role": "function",
                "name": function_name,
                "content": json.dumps({"error": "Error: Function not found"}),
            }


def create_llamda_function(
    func: Callable[P, R],
    name: str | None = None,
    description: str | None = None,
) -> Union[LlamdaFunction[R], LlamdaPydantic[R]]:
    """
    Create a Llamda function.
    """
    func_name: str = name or func.__name__
    func_description: str = description or func.__doc__ or ""

    sig = signature(func)
    if len(sig.parameters) == 1:
        param = next(iter(sig.parameters.values()))
        if issubclass(param.annotation, BaseModel):
            return LlamdaPydantic.create(
                func_name, param.annotation, func_description, func
            )

    fields: Dict[str, tuple[type, Any]] = {}
    for param_name, param in sig.parameters.items():
        field_type = param.annotation if param.annotation != Parameter.empty else Any
        field_default = param.default if param.default != Parameter.empty else ...
        fields[param_name] = (field_type, field_default)

    return LlamdaFunction.create(func_name, fields, func_description, func)
