"""
@llamdafy decorator for transforming Python functions into LLM function calls.

The decorator will read:

- The main description from the decorator arguments.
- The parameter descriptions from the function docstring.
- The type annotations from the function arguments.
- The return type annotation from the function.

The decorator will return a @LlamdaFunction object.
"""

from inspect import Parameter, signature
from types import MappingProxyType
from typing import Any, Callable, Type, TypeVar, get_type_hints

from docstring_parser import Docstring
from docstring_parser import parse as parse_docstring

from .llamda_function import LlamdaFunction
from .utils import console

T = TypeVar("T")


def llamdafy(**descriptions: str):
    """
    Decorator to transform a Python function into an LLM function.
    """

    def decorator(func: Callable[..., T]) -> LlamdaFunction[T]:
        # Extract function name
        name = descriptions.get("n  ame", func.__name__)

        # Extract and merge function description
        main_description: str | None = (
            descriptions.get("main")
            or parse_docstring(func.__doc__ or "").short_description
            or ""
        )

        if not main_description:
            console.log(
                f"No main description found for function {name}", style="warning"
            )

        # Extract parameter information
        params: MappingProxyType[str, Parameter] = signature(func).parameters
        type_hints: dict[str, Type[Any]] = get_type_hints(func)
        docstring: Docstring = parse_docstring(func.__doc__ or "")

        parameters: dict[str, Any] = {}
        for param_name, _ in params.items():
            # Get description from docstring or descriptions dict
            param_desc: str | None = next(
                (p.description for p in docstring.params if p.arg_name == param_name),
                None,
            )
            param_desc = descriptions.get(param_name, param_desc)
            parameters[param_name] = param_desc or ""

        # Get return type
        return_type: type | None = type_hints.get("return")

        # Create LlamdaFunction instance
        llamda_func = LlamdaFunction(
            name=name,
            description=main_description,
            parameters=parameters,
            return_type=return_type,
            fn=func,
        )

        return llamda_func

    return decorator


__all__: list[str] = ["llamdafy"]
