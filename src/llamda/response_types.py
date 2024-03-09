from typing import TypeVar, Optional
from pydantic import BaseModel, Field
from typing import Dict

ExecReturnType = TypeVar('ExecReturnType')

class ResultObject(BaseModel):
  result: Optional[ExecReturnType] = Field(...)
  success: bool = Field(...)
  parameter_errors: Optional[Dict[str, str]] = Field(default=None)
  exception: Optional[str] = Field(default=None)

  def __str__(self) -> str:
    return str(self.result)
  
  def __repr__(self) -> str:
    return repr(self.result)
  
class ExecutionResponseItem(BaseModel):
    function_name: str
    result: ResultObject
