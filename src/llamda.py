"""
Llambda is a library for creating Llamda functions.
"""

from typing import Dict, Any, Callable, Type
from .llamda_function import LlamdaFunction, R


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
        fields: Dict[str, Any] | None = None,
    ) -> Callable[..., Type[LlamdaFunction[Any]]]:
        """
        Decorator for creating Llamda functions.
        """

        def decorator(func: Callable[..., R]) -> Type[LlamdaFunction[R]]:
            func_name: str = name or func.__name__
            llamda_func: Type[LlamdaFunction[R]] = LlamdaFunction.create(
                func_name, fields or {}, description or func.__doc__ or "", func
            )

            self._tools[func_name] = llamda_func
            return llamda_func

        return decorator

    @property
    def tools(self) -> Dict[str, Type[LlamdaFunction[Any]]]:
        """
        Get the tools.
        """
        return self._tools


def create_llamda_function(
    func: Callable[..., R],
    name: str | None = None,
    description: str | None = None,
    fields: Dict[str, Any] | None = None,
) -> Type[LlamdaFunction[R]]:
    """
    Create a Llamda function.
    """
    return LlamdaFunction.create(
        name or func.__name__, fields or {}, description or func.__doc__ or "", func
    )
