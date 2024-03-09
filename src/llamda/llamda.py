import inspect
import re
from typing import Callable, Dict, Optional, Union, get_args, get_origin, Type
from llamda.response_types import ResultObject, ExecReturnType

def strip_params_from_docstring(docstring: str) -> str:
    return re.sub(r"@param\s+\w+:.*\n?", "", docstring).strip()

def llamda(func: Optional[Callable[..., ExecReturnType]] = None, /, *, handle_exceptions: bool = False, **descriptions: str):
    def decorator(func: Callable[..., ExecReturnType]) -> Callable[..., ResultObject[ExecReturnType]]:
        signature = inspect.signature(func)

        # Extract main description from decorator params or docstring
        description = descriptions.get(
            "main",
            strip_params_from_docstring(func.__doc__) if func.__doc__ else None)
    
        if not description:
            raise ValueError(f"Description missing for function '{func.__name__}'")
       
        # Extract parameter descriptions from the docstring
        docstring_descriptions = {}
        if func.__doc__:
            pattern = r"@param\s+(\w+):\s*(.*)"
            matches = re.findall(pattern, func.__doc__)
            docstring_descriptions = {name: description for name, description in matches}
        
        # Merge descriptions from arguments and docstring
        merged_descriptions = {**descriptions, **docstring_descriptions}
        
        # Check if all parameters have descriptions
        for name in signature.parameters:
            if name not in merged_descriptions:
                raise ValueError(f"Description missing for parameter '{name}' in function '{func.__name__}'")
        
        # Check if all parameters have type annotations
        for name, parameter in signature.parameters.items():
            if parameter.annotation == inspect.Parameter.empty:
                raise ValueError(f"Type annotation missing for parameter '{name}' in function '{func.__name__}'")
        
        func.descriptions = merged_descriptions
        
        def wrapper(*args, **kwargs) -> ResultObject:
            # Check if all required parameters are present and of the correct type
            parameter_errors = {}
            for name, parameter in signature.parameters.items():
                if get_origin(parameter.annotation) is not Optional and name not in kwargs:
                    parameter_errors[name] = f"Parameter '{name}' is required"
                    next
                if name in kwargs and parameter.annotation is not inspect.Parameter.empty:
                    expected_types = get_args(parameter.annotation) if get_origin(parameter.annotation) is Union else (parameter.annotation,)
                    if not isinstance(kwargs[name], tuple(expected_types)):
                        parameter_errors[name] = f"Parameter '{name}' is not of type '{parameter.annotation}'"
            if parameter_errors:
                if handle_exceptions:
                    return ResultObject(success=False, parameter_errors=parameter_errors)
                else:
                    raise ValueError(parameter_errors)

            try:
                result = func(*args, **kwargs)
                return ResultObject(result=result, success=True)
            except Exception as e:
                if handle_exceptions:
                    return ResultObject(success=False, exception=str(e))
                else:
                    raise e
                
        wrapper.to_schema = lambda: {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: {
                        "type": get_type_str(parameter.annotation),
                        "description": merged_descriptions.get(name, "")
                    }
                    for name, parameter in signature.parameters.items()
                },
                "required": [
                    name
                    for name, parameter in signature.parameters.items()
                    # unless the parameter is optional or has a default value
                    if is_argument_required(parameter)
                ]
            }
        }

        return wrapper
    return decorator if func is None else decorator(func)

def get_type_str(annotation: Type):
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        return [get_type_str(arg) for arg in args if arg is not type(None)]
    elif origin is Optional:
        args = get_args(annotation)
        return get_type_str(args[0])
    return annotation.__name__.lower()

def is_argument_required(argument: inspect.Parameter) -> bool:
    return argument.default is inspect.Parameter.empty and (\
        (get_origin(argument.annotation) is not Optional) or\
        (get_origin(argument.annotation) is Union and type(None) not in get_args(argument.annotation)))