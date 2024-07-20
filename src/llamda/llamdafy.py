"""
@llamdafy decorator for transforming Python functions into LLM function calls.

The decorator will read:

- The main description from the decorator arguments.
- The parameter descriptions from the function docstring.
- The type annotations from the function arguments.
- The return type annotation from the function.

The decorator will return a @LlamdaFunction object.
"""

import inspect
from typing import Callable, TypeVar

from llamda.llamda_fn import LlamdaFunction
from llamda.introspection_tools import (
    get_docstring_descriptions,
    strip_meta_from_docstring,
)

T = TypeVar("T")


def llamdafy(func: Callable[..., T], /, **descriptions: str) -> LlamdaFunction[T]:
    """
    Decorator to make a function into a LlamdaFunction.
    """

    def decorator(func: Callable[..., T]) -> LlamdaFunction[T]:
        signature: inspect.Signature = inspect.signature(func)

        # Extract main description from decorator params or docstring
        description: str | None = descriptions.get(
            "main", strip_meta_from_docstring(func.__doc__) if func.__doc__ else None
        )

        if not description:
            raise ValueError(f"Description missing for function '{func.__name__}'")

        # Extract parameter descriptions from the docstring
        docstring_description: dict[str, str] = get_docstring_descriptions(
            func.__doc__ or ""
        )

        # Merge descriptions from arguments and docstring
        merged_descriptions: dict[str, str] = {**descriptions, **docstring_description}

        # Check if all parameters have descriptions
        for name in signature.parameters:
            if name not in merged_descriptions:
                raise ValueError(
                    f"Description missing for parameter '{name}' in function '{func.__name__}'"
                )

        # Check if all parameters have type annotations
        for name, parameter in signature.parameters.items():
            if parameter.annotation == inspect.Parameter.empty:
                raise ValueError(
                    f"Type annotation missing for parameter '{name}'\
                          in function '{func.__name__}'"
                )

        return LlamdaFunction(
            func=func,
            description=description,
            signature=signature,
            param_descriptions=merged_descriptions,
        )

    return decorator(func)
