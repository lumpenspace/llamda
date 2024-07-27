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
from llamda_fn.utils.api import ToolCall, ToolParam
from .llamda_classes import LlamdaFunction, LlamdaPydantic

R = TypeVar("R")
P = ParamSpec("P")


class LlamdaFunctions:
    def __init__(self) -> None:
        self._tools: Dict[str, Union[LlamdaFunction[Any], LlamdaPydantic[Any]]] = {}

    @property
    def tools(self) -> Dict[str, Union[LlamdaFunction[Any], LlamdaPydantic[Any]]]:
        return self._tools

    def llamdafy(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable[[Callable[P, R]], Union[LlamdaFunction[R], LlamdaPydantic[R]]]:
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

            llamda_func = LlamdaFunction.create(
                func_name, fields, func_description, func
            )
            self._tools[func_name] = llamda_func
            return llamda_func

        return decorator

    def get(self, names: Optional[List[str]] = None) -> Sequence[ToolParam]:
        if names is None:
            names = list(self._tools.keys())

        return [
            self._tools[name].to_tool_schema() for name in names if name in self._tools
        ]

    def execute_function(self, tool_call: ToolCall) -> Dict[str, Any]:
        try:
            if tool_call.function.name not in self._tools:
                raise KeyError(f"Function '{tool_call.function.name}' not found")

            parsed_args = json.loads(tool_call.function.arguments)
            result = self._tools[tool_call.function.name].run(**parsed_args)
        except KeyError as e:
            result = {"error": f"Error: {str(e)}"}
        except ValidationError as e:
            result = {"error": f"Error: Validation failed - {str(e)}"}
        except Exception as e:
            result = {"error": f"Error: {str(e)}"}

        return {
            "content": json.dumps(result),
            "role": "tool",
            "tool_call_id": tool_call.id,
        }

    def __getitem__(self, key: str) -> Union[LlamdaFunction[Any], LlamdaPydantic[Any]]:
        return self._tools[key]

    def __contains__(self, key: str) -> bool:
        return key in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)
