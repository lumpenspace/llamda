"""
# LlamdaFunction

The `LlamdaFunction` class represents a callable function with additional metadata.
"""

import json
from typing import (
    get_args,
    Callable,
    Dict,
    TypeVar,
    Generic,
    Any,
)
from inspect import Signature
from llamda.response_types import ToolCallResult, ParameterError
from llamda.introspection_tools import get_type_str, is_argument_required

T = TypeVar("T")


class LlamdaFunction(Generic[T]):
    """Represents a callable function with additional metadata for use with the LLamda."""

    def __init__(
        self,
        func: Callable[..., T],
        description: str,
        signature: Signature,
        param_descriptions: Dict[str, str],
    ) -> None:
        self.func: Callable[..., T] = func
        self.description: str = description
        self.signature: Signature = signature
        self.param_descriptions: Dict[str, str] = param_descriptions

        # Set the __doc__ attribute to the docstring of the wrapped function
        self.__doc__: str | None = func.__doc__
        self.__annotations__: dict[str, Any] = func.__annotations__
        self.__module__: str = func.__module__
        self.__name__: str = func.__name__

    def __repr__(self):
        return f"<LlamdaFunction {self.func.__name__}>"

    def __str__(self):
        return f"{repr(self)} \n{self.func.__doc__} \n{json.dumps(self.to_schema(), indent=2)}"

    def __call__(
        self, handle_exceptions: bool = False, **kwargs: Any
    ) -> ToolCallResult[T]:
        """Calls the wrapped function with the provided keyword arguments.

        Args:
            handle_exceptions (bool, optional): whether to automatically handle exceptions.
            Defaults to False.

        Raises:
            ValueError: If a required parameter is missing or of the wrong type
            and handle_exceptions is False.

        Returns:
            ToolCallResult: The result of the function call, including success status
            and any errors or exceptions.
        """
        parameter_error = None
        for name in self.signature.parameters.keys():
            if is_argument_required(self.func, name):
                if name not in kwargs:
                    parameter_error = ParameterError(
                        name=name, description=f"Parameter '{name}' is required"
                    )
                    break
                elif not isinstance(
                    kwargs[name], self.signature.parameters[name].annotation
                ):
                    parameter_error = ParameterError(
                        name=name,
                        description=f"Parameter '{name}' is not of the correct type",
                    )
                    break
            else:
                # if the parameter is present and of the wrong type,
                # add it to the parameter_errors
                if name in kwargs and not type(kwargs[name]) in get_args(
                    self.signature.parameters[name].annotation
                ):
                    parameter_error = ParameterError(
                        name=name,
                        description=f"Parameter '{name}' is not of the correct type",
                    )

        if parameter_error:
            if handle_exceptions:
                return ToolCallResult(
                    success=False, parameter_error=parameter_error, result=None
                )
            else:
                raise ValueError(parameter_error.description)

        try:
            result = self.func(**kwargs)
            return ToolCallResult(result=result, success=True)
        except (TypeError, ValueError) as e:
            if handle_exceptions:
                return ToolCallResult(result=None, success=False, exception=str(e))
            else:
                raise e

    def to_schema(
        self,
    ) -> dict[str, str | dict[str, str | dict[str, dict[str, str | None]] | list[str]]]:
        """Converts the LlamdaFunction instance to a JSON schema
        representation.

        Returns:

        Dict: The JSON schema representation of the LlamdaFunction.
        """
        return {
            "name": self.func.__name__,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: {
                        "type": print(name, parameter)
                        and get_type_str(parameter.annotation),
                        "description": self.param_descriptions.get(name, ""),
                    }
                    for name, parameter in self.signature.parameters.items()
                },
                "required": [
                    name
                    for name in self.signature.parameters.keys()
                    # unless the parameter is optional or has a default value
                    if is_argument_required(self.func, name)
                ],
            },
        }


__all__: list[str] = [
    "LlamdaFunction",
]
