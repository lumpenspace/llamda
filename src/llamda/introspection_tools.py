"""
Introspection tools for the Llamda library.
"""

import inspect
import re
from typing import Type, Union, Optional, get_origin, get_args, Any, Callable


def strip_meta_from_docstring(docstring: str) -> str:
    """
    Strip meta information from the docstring.
    """

    # Remove lines starting with @param or @return
    result = re.sub(r"^[ \t]*@\w+.*\n?", "", docstring, flags=re.MULTILINE)

    # Remove leading/trailing whitespace and empty lines
    result = re.sub(r"^\s+|\s+$", "", result, flags=re.MULTILINE)

    return result


def get_docstring_descriptions(docstring: str | None) -> dict[str, str]:
    """
    Get the descriptions from the docstring.
    """
    docstring_descriptions: dict[str, str] = {}
    if docstring is not None:
        pattern = r"@param\s+(\w+):\s*(.*)"
        matches: list[Any] = re.findall(pattern, docstring)
        docstring_descriptions = {name: description for name, description in matches}
    return docstring_descriptions


def get_type_str(annotation: Type[Any]) -> str:
    """
    Get the type string from the annotation.
    """
    origin: Any | None = get_origin(annotation)
    if origin is Union:
        args: tuple[Any, ...] = get_args(annotation)
        types: list[str] = [get_type_str(arg) for arg in args if arg is not type(None)]
        if len(types) == 1:
            return types[0]
        return ", ".join(types)
    elif origin is Optional:
        args: tuple[Any, ...] = get_args(annotation)
        return get_type_str(args[0])
    return annotation.__name__.lower()


def is_argument_required(func: Callable[..., Any], argument_name: str) -> bool:
    """
    Check if the argument is required.
    """
    signature: inspect.Signature = inspect.signature(func)
    argument: inspect.Parameter = signature.parameters[argument_name]
    origin: Any | None = get_origin(argument.annotation)
    args: tuple[Any, ...] = get_args(argument.annotation)
    optional: bool = origin is Union and type(None) in args
    has_default: bool = argument.default is not inspect.Parameter.empty
    return not optional and not has_default


__all__: list[str] = [
    "get_docstring_descriptions",
    "get_type_str",
    "is_argument_required",
]
