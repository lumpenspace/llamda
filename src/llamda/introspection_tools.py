import inspect
import re
from typing import Type, Union, Optional, get_origin, get_args


def strip_meta_from_docstring(docstring: str) -> str:

    # Remove lines starting with @param or @return
    result = re.sub(r'^[ \t]*@\w+.*\n?', '', docstring, flags=re.MULTILINE)

    # Remove leading/trailing whitespace and empty lines
    result = re.sub(r'^\s+|\s+$', '', result, flags=re.MULTILINE)

    return result

def get_docstring_descriptions(docstring: str) -> dict:
    docstring_descriptions = {}
    if docstring is not None:
        pattern = r"@param\s+(\w+):\s*(.*)"
        matches = re.findall(pattern, docstring)
        docstring_descriptions = {name: description for name, description in matches}
    return docstring_descriptions

def get_type_str(annotation: Type):
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        types = [get_type_str(arg) for arg in args if arg is not type(None)]
        if len(types) == 1:
            return types[0]
        return types
    elif origin is Optional:
        args = get_args(annotation)
        return get_type_str(args[0])
    return annotation.__name__.lower()

def is_argument_required(argument: inspect.Parameter) -> bool:
    origin = get_origin(argument.annotation)
    args = get_args(argument.annotation)
    optional = origin is Union and type(None) in args
    has_default = argument.default is not inspect.Parameter.empty
    return not optional and not has_default
