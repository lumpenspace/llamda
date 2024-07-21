"""
LlamdaFunction is a Pydantic model that represents a function that can be called by an LLM.
"""

from typing import Any, Callable, Generic, Type, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class LlamdaFunction(BaseModel, Generic[T]):
    """
    LlamdaFunction is a Pydantic model that represents a function that can be called by an LLM.
    """

    name: str = Field(...)
    description: str = Field(...)
    parameters: dict[str, str] = Field(...)
    return_type: Type[T] | None = Field(default=None)
    fn: Callable[..., T] = Field(...)

    class Config:
        """
        Config for the LlamdaFunction model.
        """

        arbitrary_types_allowed = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not callable(self.fn):
            raise TypeError(f"{self.fn} is not callable")
        return self(*args, **kwargs)

    def to_schema(self) -> dict[str, Any]:
        """
        Convert the LlamdaFunction to a schema.
        """
        return self.model_json_schema()
