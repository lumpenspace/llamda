from typing import TypeVar, Optional, Generic, Dict
import json
from pydantic import BaseModel, Field

ExecReturnType = TypeVar('ExecReturnType')



class ParameterError(BaseModel):
    name: str
    description: str

    def __str__(self) -> str:
        return self.description
  
    def __repr__(self) -> str:
        return repr(self.description)
  
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParameterError):
            return self.description == other
        return self.description == other.description
      
class ToolCallResult(BaseModel, Generic[ExecReturnType]):
  result: Optional[ExecReturnType] = Field(...)
  success: bool = Field(...)
  parameter_error: Optional[ParameterError] = Field(default={})
  exception: Optional[str] = Field(default=None)

  def __str__(self) -> str:
    return str(self.result)
  
  def __repr__(self) -> str:
    return repr(self.result)
  
  def __eq__(self, other: object) -> bool:
    if not isinstance(other, ToolCallResult):
      return self.result == other
    return self.result == other.result
  
class ExecutionResponseItem(BaseModel):
    function_name: str
    result: ToolCallResult

    def __str__(self) -> str:
        return str(self.result)
  
    def __repr__(self) -> str:
        return repr(self.result)
  
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExecutionResponseItem):
            return self.result == other
        return self.result == other.result

class ExecutionResponse(BaseModel):
    results: Dict[str, ExecutionResponseItem] = Field(...)

    def to_tool_response(self) -> str:
        tool_responses = []
        for call_id, result in self.results.items():
            content = result.result
            tool_response = {
                "role": "tool",
                "content": content,
                "tool_call_id": call_id
            }
            tool_responses.append(tool_response)
        return json.dumps(tool_responses)
