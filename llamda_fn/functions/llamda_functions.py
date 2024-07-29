import json
from inspect import Parameter, isclass, signature
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    ParamSpec,
    Union,
    Sequence,
    Iterator,
)

from pydantic import BaseModel, ValidationError
from llamda_fn.llms.api_types import ToolCall, ToolResponse, OaiToolParam
from .llamda_classes import LlamdaFunction, LlamdaPydantic

R = TypeVar("R")
P = ParamSpec("P")


class LlamdaFunctions:
    """Creation and management of LLM tools"""

    def __init__(self) -> None:
        self._tools: Dict[str, Union[LlamdaFunction[Any], LlamdaPydantic[Any]]] = {}

    @property
    def tools(self) -> Dict[str, Union[LlamdaFunction[Any], LlamdaPydantic[Any]]]:
        """Returns the tools registered with the registry"""
        return self._tools

    def llamdafy(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable[[Callable[P, R]], Union[LlamdaFunction[R], LlamdaPydantic[R]]]:
        """Decorator to mark a function as a tool and register it with the registry"""

        def decorator(
            func: Callable[P, R]
        ) -> Union[LlamdaFunction[R], LlamdaPydantic[R]]:
            func_name: str = name or func.__name__
            func_description: str = description or func.__doc__ or ""

            sig = signature(func)
            if len(sig.parameters) == 1:
                param = next(iter(sig.parameters.values()))
                if isclass(param.annotation) and issubclass(
                    param.annotation, BaseModel
                ):
                    llamda_func = LlamdaPydantic.create(
                        func_name, param.annotation, func_description, func
                    )
                    self._tools[func_name] = llamda_func
                    return llamda_func

            fields: Dict[str, tuple[type, Any]] = {
                param_name: (
                    param.annotation if param.annotation != Parameter.empty else Any,
                    param.default if param.default != Parameter.empty else ...,
                )
                for param_name, param in sig.parameters.items()
            }

            llamda_func: LlamdaFunction[R] = LlamdaFunction.create(
                func_name, fields, func_description, func
            )
            self._tools[func_name] = llamda_func
            return llamda_func

        return decorator

    def get(self, names: Optional[List[str]] = None) -> Sequence[OaiToolParam]:
        """Returns the tool spec for some or all of the functions in the registry"""
        if names is None:
            names = list(self._tools.keys())

        return [
            self._tools[name].to_tool_schema() for name in names if name in self._tools
        ]

    def execute_function(self, tool_call: ToolCall) -> ToolResponse:
        """Executes the function specified in the tool call with the required arguments"""
        try:
            if tool_call.name not in self._tools:
                raise KeyError(f"Function '{tool_call.name}' not found")

            parsed_args = json.loads(tool_call.arguments)
            result = self._tools[tool_call.name].run(**parsed_args)
        except KeyError as e:
            result: dict[str, str] = {"error": f"Error: {str(e)}"}
        except ValidationError as e:
            result = {"error": f"Error: Validation failed - {str(e)}"}
        except Exception as e:
            result = {"error": f"Error: {str(e)}"}

        return ToolResponse(
            result,
            **tool_call.model_dump(),
        )

    def __getitem__(self, key: str) -> Union[LlamdaFunction[Any], LlamdaPydantic[Any]]:
        return self._tools[key]

    def __contains__(self, key: str) -> bool:
        return key in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)
