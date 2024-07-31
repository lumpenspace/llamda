"""
Defines the base classes for Llamda functions: LlamdaCallable and LlamdaBase.
These classes provide the foundation for creating and managing Llamda functions,
including abstract methods for execution and schema generation.
"""

from typing import Any, Callable, Dict, Generic, TypeVar
from pydantic import BaseModel, ConfigDict

from llamda_fn.llms.oai_api_types import OaiToolSpec

R = TypeVar("R")

__all__ = ["LlamdaCallable"]


class LlamdaCallable(Generic[R]):
    """
    Represents a callable to proxy the original Function or model internally.
    This abstract base class defines the interface for Llamda functions.
    """

    def run(self, **kwargs: Any) -> R:
        """
        Execute the Llamda function with the given parameters.
        This method should be implemented by subclasses.

        Args:
            **kwargs: Keyword arguments to be passed to the function.

        Returns:
            The result of the function execution.
        """
        raise NotImplementedError

    def to_tool_schema(self) -> OaiToolSpec:
        """
        Convert the Llamda function to a tool schema compatible with OpenAI's API.
        This method should be implemented by subclasses.

        Returns:
            A dictionary representing the OpenAI tool specification.
        """
        raise NotImplementedError

    @classmethod
    def create(
        cls,
        call_func: Callable[..., R],
        name: str = "",
        description: str = "",
        **kwargs: Any,
    ) -> "LlamdaCallable[R]":
        """
        Create a new LlamdaCallable instance.
        This method should be implemented by subclasses.

        Args:
            call_func: The function to be wrapped.
            name: The name of the Llamda function.
            description: A description of the Llamda function.
            **kwargs: Additional keyword arguments for function creation.

        Returns:
            A new instance of LlamdaCallable.
        """
        raise NotImplementedError


class LlamdaBase(BaseModel, LlamdaCallable[R]):
    """
    The base class for Llamda functions, combining Pydantic's BaseModel
    with the LlamdaCallable interface.
    """

    name: str
    description: str
    call_func: Callable[..., R]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for the Llamda function.
        This method should be implemented by subclasses.

        Returns:
            A dictionary representing the JSON schema of the function.
        """
        raise NotImplementedError

    def to_tool_schema(self) -> OaiToolSpec:
        """Get the JSON schema for the LlamdaPydantic."""
        schema = self.to_schema()
        return {
            "type": "function",
            "function": {
                "name": schema["title"],
                "description": schema["description"],
                "parameters": {
                    "type": "object",
                    "properties": schema["properties"],
                    "required": schema.get("required", []),
                },
            },
        }
