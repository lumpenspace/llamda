import inspect
from typing import Callable, Optional

from .response_types import ExecReturnType
from .LlamdaFunction import LlamdaFunction
from .introspection_tools import get_docstring_descriptions, strip_meta_from_docstring

def llamdafy(func: Optional[Callable[..., ExecReturnType]] = None, /, **descriptions: str):
    def decorator(func: Callable[..., ExecReturnType]) -> LlamdaFunction:
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
                
        return LlamdaFunction(func=func, description=description, signature=signature, param_descriptions=merged_descriptions)
    
    return decorator if func is None else decorator(func)