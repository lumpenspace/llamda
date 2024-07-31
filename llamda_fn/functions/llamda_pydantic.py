"""
Defines the LlamdaPydantic class, which implements a Llamda function
using a Pydantic model for input validation. It extends the LlamdaBase class and
provides methods for creating, running, and generating schemas for Pydantic-based
Llamda functions.
"""

from typing import Any, Callable, Type

from pydantic import BaseModel

from llamda_fn.functions.llamda_callable import LlamdaBase

from .llamda_callable import R

__all__ = ["LlamdaPydantic"]


class LlamdaPydantic(LlamdaBase[R]):
    """
    A Llamda function that uses a Pydantic model as the input.
    This class provides a way to create Llamda functions with
    built-in validation using Pydantic models.
    """

    model: Type[BaseModel]

    @classmethod
    def create(
        cls,
        call_func: Callable[..., R],
        name: str = "",
        description: str = "",
        model: Type[BaseModel] = BaseModel,
        **kwargs: Any,
    ) -> "LlamdaPydantic[R]":
        """
        Create a new LlamdaPydantic instance from a Pydantic model.

        Args:
            call_func: The function to be called when running the Llamda function.
            name: The name of the Llamda function.
            description: A description of the Llamda function.
            model: The Pydantic model to use for input validation.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A new LlamdaPydantic instance.
        """
        return cls(
            name=name,
            description=description,
            call_func=call_func,
            model=model,
        )

    def run(self, **kwargs: Any) -> R:
        """
        Run the LlamdaPydantic with the given parameters.

        Args:
            **kwargs: Keyword arguments to be validated and passed to the function.

        Returns:
            The result of the function execution.
        """
        validated_params = self.model(**kwargs)
        return self.call_func(validated_params)

    def to_schema(self) -> dict[str, Any]:
        """
        Get the JSON schema for the LlamdaPydantic.

        Returns:
            A dictionary representing the JSON schema of the function,
            including the Pydantic model schema.
        """
        schema: dict[str, Any] = self.model.model_json_schema(mode="serialization")
        schema["title"] = self.name
        schema["description"] = self.description
        return schema
