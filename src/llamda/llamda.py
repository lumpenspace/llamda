import inspect
from typing import Callable, Dict, Optional, Union, get_args, get_origin, Type
from llamda.response_types import ResultObject, ExecReturnType

def llamda(handle_exceptions: bool = False, **descriptions: Dict[str, str]):
    def decorator(func: Callable[..., ExecReturnType]) -> Callable[..., ResultObject[ExecReturnType]]:
        signature = inspect.signature(func)
       
        # Check if all parameters have descriptions
        for name in signature.parameters:
            if name not in descriptions:
                raise ValueError(f"Description missing for parameter '{name}' in function '{func.__name__}'")
        
        # Check if all parameters have type annotations
        for name, parameter in signature.parameters.items():
            if parameter.annotation == inspect.Parameter.empty:
                raise ValueError(f"Type annotation missing for parameter '{name}' in function '{func.__name__}'")
        
        # Check if the function has a docstring
        if not func.__doc__:
            raise ValueError(f"Docstring missing for function '{func.__name__}'")
        
        func.descriptions = descriptions
        func.to_schema = lambda: {
            "name": func.__name__,
            "description": inspect.cleandoc(func.__doc__),
            "parameters": {
                "type": "object",
                "properties": {
                    name: {
                        "type": get_type_str(parameter.annotation),
                        "description": descriptions.get(name, "")
                    }
                    for name, parameter in signature.parameters.items()
                },
                "required": [
                    name
                    for name, parameter in signature.parameters.items()
                    if get_origin(parameter.annotation) is not Optional
                ]
            }
        }
        
        def wrapper(*args, **kwargs) -> ResultObject:
            # Check if all required parameters are present and of the correct type
            parameter_errors = {}
            for name, parameter in signature.parameters.items():
                if get_origin(parameter.annotation) is not Optional and name not in kwargs:
                    parameter_errors[name] = f"Parameter '{name}' is required"
                    next
                if not isinstance(kwargs[name], parameter.annotation):
                    parameter_errors[name] = f"Parameter '{name}' is not of type '{parameter.annotation}'"

            if parameter_errors:
                if handle_exceptions:
                    return ResultObject( success=False, parameter_errors=parameter_errors)
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

        return wrapper
    return decorator

def get_type_str(annotation: Type):
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        return [str(arg) for arg in args if arg is not type(None)]
    elif origin is Optional:
        args = get_args(annotation)
        return str(args[0])
    return str(annotation)