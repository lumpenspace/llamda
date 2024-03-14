import json
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, List, Any

from openai.types.chat import ChatCompletionMessage

from .response_types import ToolCallResult, ExecutionResponse
from .LlamdaFunction import LlamdaFunction

    
class Llamdas(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
    functions: Dict[str, LlamdaFunction] = Field(..., json_schema_extra={"description": "Mapping of function names to LlamdaFunction instances", "example": {"exampleFunction": LlamdaFunction}})
    handle_exceptions: bool = Field(default=True, json_schema_extra={"description": "Flag to determine if exceptions should be automatically handled"})

    @field_validator('functions', mode='before')
    @classmethod
    def transform_functions(cls, v):

        if (isinstance(v, List) and not all(isinstance(func, LlamdaFunction) for func in v))\
                or isinstance(v, Dict) and not all(isinstance(func, LlamdaFunction) for func in v.values()):
            raise ValueError('All values in functions must be instances of LlamdaFunction')

        if isinstance(v, List):
            func_names = [func.__name__ for func in v]
            if len(func_names) != len(set(func_names)):
                raise ValueError('Function names must be unique')

        return {func.__name__: func for func in v} if isinstance(v, List) else v

    def to_openai_tools(self):
        return [
            {
                "type": "function",
                "function": func.to_schema()
            }
            for func in self.functions.values()
        ]

    def execute(self, message: ChatCompletionMessage) -> ExecutionResponse:
        if "tool_calls" in message:
            tool_calls = message["tool_calls"]
            results = {}
            for call in tool_calls:
                func_name = call["function"]["name"]
                if func_name in self.functions:
                    func = self.functions[func_name]
                    try:
                        args:Dict[str, Any] = json.loads(call["function"]["arguments"])
                        result = func(**args, handle_exceptions=self.handle_exceptions)


                        results[call["id"]] = {
                            "function_name": func_name,
                            "result": result
                        }
                    except Exception as e:
                        if self.handle_exceptions:
                            results[call["id"]] = {
                                "function_name": func_name,
                                "result": ToolCallResult(**{"success": False, "exception": str(e), "result": None})
                            }
                        else:
                            raise e
            return ExecutionResponse(results=results)
        return None