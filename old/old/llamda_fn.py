"""
# LlamdaFunction

The `LlamdaFunction` class represents a callable function with additional metadata.
"""

from dataclasses import _DataclassT
from inspect import Signature
import json
from signal import Sigmasks
from typing import (
    get_args,
    Callable,
    Dict,
    TypeVar,
    Generic,
    Any,
    Type,
)
from llamda.response_types import ToolCallResult, ParameterError
from llamda.introspection_tools import get_type_str, is_argument_required
from pydantic import BaseConfig, BaseModel, ConfigDict, Field, RootModel, fun
from pydantic_settings import BaseSettings, SettingsConfigDict

T = TypeVar("T", bound=Callable[..., Any])

FunctionTypes = Callable[..., T]


class LlamdaFunction(BaseModel, Generic[T]):
    """Represents a callable function with additional metadata for use with the LLamda."""

    annotations: dict[str, Any] = Field(
        ...,
    )
    param_descriptions: dict[str, str] = Field(...)
    return_type: TypeVar = Field(...)
    signature: Signature = Field(...,)
    description: str = Field()
    doc: str = Field(...)
    name: str = Field()
    fn: T = Field(...)

    class Config:
        arbitrary_types_allowed = True

    def __str__(self):
        return f"""{repr(self)} \n{self.__doc__}\n
               {json.dumps(self.model_json_schema(), indent=2)}"""

    def __call__(
        self, *args: Any, handle_exceptions: bool = False, **kwargs: Any
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
        parameter_errors: list[ParameterError] = []

        for name in self.signature.parameters.keys():
            parameter_error = None
            if is_argument_required(self.fn, name):
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
                parameter_errors.append(parameter_error)

        if len(parameter_errors) > 0:
            if handle_exceptions:
                return ToolCallResult(
                    success=False, parameter_errors=parameter_errors, result=None
                )
            else:
                raise ValueError(parameter_errors[0].description)

        try:
            result: T = self.fn(*args, **kwargs)
            return ToolCallResult(result=result, success=True)
        except (TypeError, ValueError) as e:

            if handle_exceptions:
                return ToolCallResult(result=None, success=False, exception=str(e))
            else:
                raise e

    # def to_schema(
    #     self,
    # ) -> dict[str, str | dict[str, str | dict[str, dict[str, str | None]] | list[str]]]:
    #     """Converts the LlamdaFunction instance to a JSON schema
    #     representation.

    #     Returns:

    #     Dict: The JSON schema representation of the LlamdaFunction.
    #     """
    #     if isinstance(self.fn, BaseModel):
    #         return self.fn.model_json_schema()
    #     return {
    #         "name": self.name,
    #         "description": self.description,
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 name: {
    #                     "type": print(name, parameter)
    #                     and get_type_str(parameter.annotation),
    #                     "description": self.param_descriptions.get(name, ""),
    #                 }
    #                 for name, parameter in self.signature.parameters.items()
    #             },
    #             "required": [
    #                 name
    #                 for name in self.signature.parameters.keys()
    #                 # unless the parameter is optional or has a default value
    #                 if is_argument_required(self.fn, name)
    #             ],
    #         },
    #     }


__all__: list[str] = [
    "LlamdaFunction",
]
