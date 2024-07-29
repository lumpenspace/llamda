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
    Sequence,
    Iterator,
)

from pydantic import BaseModel, ValidationError
from llamda_fn.llms.api_types import LlToolCall, ToolResponse, OaiToolParam
from .llamda_classes import LlamdaFunction, LlamdaPydantic, LlamdaCallable

R = TypeVar("R")
P = ParamSpec("P")


class LlamdaFunctions:
    def __init__(self) -> None:
        self._tools: Dict[str, LlamdaCallable[Any]] = {}

    @property
    def tools(self) -> Dict[str, LlamdaCallable[Any]]:
        return self._tools

    def llamdafy(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable[[Callable[P, R]], LlamdaCallable[R]]:
        def decorator(func: Callable[P, R]) -> LlamdaCallable[R]:
            func_name: str = name or func.__name__
            func_description: str = description or func.__doc__ or ""

            sig = signature(func)
            if len(sig.parameters) == 1:
                param = next(iter(sig.parameters.values()))
                if isclass(param.annotation) and issubclass(
                    param.annotation, BaseModel
                ):
                    llamda_func = LlamdaPydantic.create(
                        call_func=func,
                        name=func_name,
                        description=func_description,
                        model=param.annotation,
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

            llamda_func: LlamdaCallable[R] = LlamdaFunction.create(
                call_func=func,
                fields=fields,
                name=func_name,
                description=func_description,
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

    def execute_function(self, tool_call: LlToolCall) -> ToolResponse:
        """Executes the function specified in the tool call with the required arguments"""
        try:
            if tool_call.name not in self._tools:
                raise KeyError(f"Function '{tool_call.name}' not found")

            parsed_args = json.loads(tool_call.arguments)
            result = self._tools[tool_call.name].run(**parsed_args)
        except KeyError as e:
            result = {"error": f"Error: {str(e)}"}
        except ValidationError as e:
            result = {"error": f"Error: Validation failed - {str(e)}"}
        except Exception as e:
            result = {"error": f"Error: {str(e)}"}

        return ToolResponse(
            id=tool_call.id,
            name=tool_call.name,
            arguments=tool_call.arguments,
            result=json.dumps(result),
        )

    def __getitem__(self, key: str) -> LlamdaCallable[Any]:
        return self._tools[key]

    def __contains__(self, key: str) -> bool:
        return key in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)
