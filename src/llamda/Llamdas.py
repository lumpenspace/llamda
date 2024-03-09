import json
from pydantic import BaseModel, Field
from typing import Dict, Any, Callable
from llamda.response_types import ExecutionResponseItem, ResultObject

class ExecutionResponse(BaseModel):
    results: Field[Dict[str, ExecutionResponseItem]]

    def to_tool_response(self) -> str:
        tool_responses = []
        for call_id, result in self.results.items():
            content = result
            tool_response = {
                "role": "tool",
                "content": content,
                "tool_call_id": call_id
            }
            tool_responses.append(tool_response)
        return json.dumps(tool_responses)

class Llamdas(BaseModel):
    functions: Field[Dict[str, Callable[..., ResultObject]]]     
    def __init__(self, functions):
        self.functions = {func.__name__: func for func in functions}

    def to_openai_tools(self):
        return [
            {
                "type": "function",
                "function": func.to_schema()
            }
            for func in self.functions.values()
        ]

    def execute(self, message: Dict[str, Any]) -> ExecutionResponse:
        if "tool_calls" in message:
            tool_calls = message["tool_calls"]
            results = {}
            for call in tool_calls:
                func_name = call["function"]["name"]
                if func_name in self.functions:
                    func = self.functions[func_name]
                    try:
                        args = json.loads(call["function"]["arguments"])
                        result = func(**args)
                        results[call["id"]] = {
                            "function_name": func_name,
                            "result": result
                        }
                    except Exception as e:
                        if func.handle_exceptions:
                            results[call["id"]] = {
                                "function_name": func_name,
                                "result": ResultObject(success=False, exception=str(e))
                            }
                        else:
                            raise e
            return ExecutionResponse(results)
        return None
