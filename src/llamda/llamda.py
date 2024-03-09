import inspect
from typing import Callable, Dict, Optional, Union, get_args, get_origin, Type
from llamda.response_types import ResultObject, ExecReturnType
from llamda.introspection_tools import get_type_str, is_argument_required, get_docstring_descriptions, strip_meta_from_docstring

def llamda(func: Optional[Callable[..., ExecReturnType]] = None, /, *, handle_exceptions: bool = False, **descriptions: str):
    def decorator(func: Callable[..., ExecReturnType]) -> Callable[..., ResultObject[ExecReturnType]]:
        signature = inspect.signature(func)

        # Extract main description from decorator params or docstring
        description = descriptions.get(
            "main",
            strip_meta_from_docstring(func.__doc__) if func.__doc__ else None)
    
        if not description:
            raise ValueError(f"Description missing for function '{func.__name__}'")

        # Extract parameter descriptions from the docstring
        docstring_descriptions = get_docstring_descriptions(func.__doc__)
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
            # Construct a new arguments dictionary that aligns with the signature's expectations
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Check if all required parameters are present and of the correct type
            parameter_errors = {}
            for name, parameter in signature.parameters.items():
                if is_argument_required(parameter) and name not in bound_args.arguments:
                    parameter_errors[name] = f"Parameter '{name}' is required"
                    continue
                if name in bound_args.arguments and parameter.annotation is not inspect.Parameter.empty:
                    expected_types = get_args(parameter.annotation) if get_origin(parameter.annotation) is Union else (parameter.annotation,)
                    if not isinstance(bound_args.arguments[name], expected_types):
                        parameter_errors[name] = f"Parameter '{name}' is not of type '{get_type_str(parameter.annotation)}'"
            if parameter_errors:
                if handle_exceptions:
                    return ResultObject(success=False, parameter_errors=parameter_errors)
                else:
                    raise ValueError(parameter_errors)

            try:
                result = func(**bound_args.arguments)
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