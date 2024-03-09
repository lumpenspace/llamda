from typing import  get_args, get_origin, Union, Callable, Optional, Dict
import inspect
from .response_types import ToolCallResult, ParameterError
from .introspection_tools import get_type_str, is_argument_required

class LlamdaFunction:
    def __init__(self, func, description, signature, param_descriptions):
        self.func: Callable = func
        self.description: str = description
        self.signature: inspect.Signature = signature
        self.param_descriptions: Dict[str, str] = param_descriptions
        
    @property
    def __name__(self):
        return self.func.__name__
      
    @property
    def __doc__(self):
        return self.func.__doc__
      
    @property
    def __annotations__(self):
        return self.func.__annotations__
    
    @property
    def __module__(self):
        return self.func.__module__
      
    def __repr__(self):
        return f"<LlamdaFunction {self.func.__name__}>"
      
    def __str__(self):
        return f"LlamdaFunction {self.func.__name__}"

    def __call__(self, handle_exceptions: bool = False, **kwargs) -> ToolCallResult:

        # Check if all required parameters are present and of the correct type
        parameter_error = None
        for name, parameter in self.signature.parameters.items():           
            if is_argument_required(parameter):
                if name not in kwargs:                 
                  parameter_error = ParameterError(name=name, description=f"Parameter '{name}' is required")
                  break
                elif not isinstance(kwargs[name], parameter.annotation):                 
                  parameter_error = ParameterError(name=name, description=f"Parameter '{name}' is not of the correct type")
                  break
            else:
                # if the parameter is present and of the wrong type, add it to the parameter_errors
                if name in kwargs and not type(kwargs[name]) in get_args(parameter.annotation):                   
                    parameter_error = ParameterError(name=name, description=f"Parameter '{name}' is not of the correct type")

        if parameter_error:                      
            if handle_exceptions:
                return ToolCallResult(success=False, parameter_error=parameter_error, result=None)
            else:               
                raise ValueError(parameter_error.description)

        try:
            result = self.func(**kwargs)
            return ToolCallResult(result=result, success=True)
        except Exception as e:
            if handle_exceptions:
                return ToolCallResult(success=False, exception=str(e))
            else:
                raise e
    
    def to_schema(self):
        return {
            "name": self.func.__name__,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: {
                        "type": get_type_str(parameter.annotation),
                        "description": self.param_descriptions.get(name, "")
                    }
                    for name, parameter in self.signature.parameters.items()
                },
                "required": [
                    name
                    for name, parameter in self.signature.parameters.items()
                    # unless the parameter is optional or has a default value
                    if is_argument_required(parameter)
                ]
            }
        }