import json
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any

from openai.types.chat import ChatCompletionMessage

from .response_types import ToolCallResult, ExecutionResponse
from .LlamdaFunction import LlamdaFunction

    
class Llamdas(BaseModel):
    functions: Dict[str, LlamdaFunction] = Field(..., description="Mapping of function names to LlamdaFunction instances", example={"exampleFunction": LlamdaFunction})
    handle_exceptions: bool = Field(default=True, description="Flag to determine if exceptions should be automatically handled")

    @validator('functions', pre=True)
    def transform_functions(cls, v):
        assert isinstance(v, List)
        if not all(isinstance(func, LlamdaFunction) for func in v):
            raise ValueError('All values in functions must be instances of LlamdaFunction')

        return {func.__name__: func for func in v}
    class Config:
        arbitrary_types_allowed = True

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
                        print(f"Args: {args}")
                        print(f"Function: {func}")

                        print(f"Executing function: {func_name}")
                        result = func(**args, handle_exceptions=self.handle_exceptions)

                        print(f"Function execution completed. Result: {result}")


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