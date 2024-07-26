from typing import Dict, Any, Callable, TypeVar, ParamSpec, cast
from inspect import signature, Parameter
from .llamda_function import LlamdaFunction
from pydantic import Field

R = TypeVar("R")
P = ParamSpec("P")


class Llamda:
    """
    Main class, produces a decorator for creating Llamda functions.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, LlamdaFunction[Any]] = {}

    def llamdafy(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[[Callable[P, R]], LlamdaFunction[R]]:
        """
        Decorator for creating Llamda functions.
        """

        def decorator(func: Callable[P, R]) -> LlamdaFunction[R]:
            func_name: str = name or func.__name__
            func_description: str = description or func.__doc__ or ""

            # Extract fields from function signature
            fields: Dict[str, Any] = {}
            for param_name, param in signature(func).parameters.items():
                if param.annotation != Parameter.empty:
                    field_info = {}
                    if param.default != Parameter.empty:
                        field_info["default"] = param.default
                    fields[param_name] = (param.annotation, Field(**field_info))

            llamda_func: LlamdaFunction[R] = LlamdaFunction.create(
                func_name, fields, func_description, func
            )

            self._tools[func_name] = llamda_func
            # This cast is necessary to ensure that the return type is LlamdaFunction[R]
            # pylance disable=unnecessary-cast
            return cast(LlamdaFunction[R], llamda_func)

        return decorator

    @property
    def tools(self) -> Dict[str, LlamdaFunction[Any]]:
        """
        Get the tools.
        """
        return self._tools


def create_llamda_function(
    func: Callable[P, R],
    name: str | None = None,
    description: str | None = None,
) -> LlamdaFunction[R]:
    """
    Create a Llamda function.
    """
    func_name: str = name or func.__name__
    func_description: str = description or func.__doc__ or ""

    # Extract fields from function signature
    fields: Dict[str, Any] = {}
    for param_name, param in signature(func).parameters.items():
        if param.annotation != Parameter.empty:
            field_info = {}
            if param.default != Parameter.empty:
                field_info["default"] = param.default
            fields[param_name] = (param.annotation, Field(**field_info))

    return LlamdaFunction.create(func_name, fields, func_description, func)
