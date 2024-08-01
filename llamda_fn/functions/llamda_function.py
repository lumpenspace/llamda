"""
Defines the LlamdaFunction class, which implements a Llamda function
using a simple function model as input. It extends the LlamdaBase class and
provides methods for creating, running, and generating schemas for function-based
Llamda functions with built-in validation using Pydantic models.
"""

from typing import Any, Callable, Dict, Type

from pydantic import BaseModel, Field, create_model

from llamda_fn.functions.llamda_callable import LlamdaBase
from llamda_fn.functions.process_fields import JsonDict

from .llamda_callable import R

__all__ = ["LlamdaFunction"]


class LlamdaFunction(LlamdaBase[R]):
    """
    A Llamda function that uses a simple function model as the input.
    This class provides a way to create Llamda functions with
    built-in validation using Pydantic models.
    """

    parameter_model: Type[BaseModel]

    @classmethod
    def create(
        cls,
        call_func: Callable[..., R],
        name: str = "",
        description: str = "",
        fields: JsonDict = None,
        **kwargs: Any,
    ) -> "LlamdaFunction[R]":
        """
        Create a new LlamdaFunction from a function.

        Args:
            call_func: The function to be called when running the Llamda function.
            name: The name of the Llamda function.
            description: A description of the Llamda function.
            fields: A dictionary of field names and their types/default values.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A new LlamdaFunction instance.
        """
        model_fields = {}
        for field_name, (field_type, field_default) in fields.items():
            if field_default is ...:
                model_fields[field_name] = (field_type, Field(...))
            else:
                model_fields[field_name] = (field_type, Field(default=field_default))

        parameter_model: type[BaseModel] = create_model(
            f"{name}Parameters", **model_fields
        )

        return cls(
            name=name,
            description=description,
            parameter_model=parameter_model,
            call_func=call_func,
        )

    def run(self, **kwargs: Any) -> R:
        """
        Run the LlamdaFunction with the given parameters.

        Args:
            **kwargs: Keyword arguments to be validated and passed to the function.

        Returns:
            The result of the function execution.
        """
        validated_params = self.parameter_model(**kwargs)
        return self.call_func(**validated_params.model_dump())

    def to_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for the LlamdaFunction.

        Returns:
            A dictionary representing the JSON schema of the function,
            including the parameter model schema.
        """
        schema = self.parameter_model.model_json_schema()
        schema["title"] = self.name
        schema["description"] = self.description
        return schema

    @property
    def __name__(self) -> str:
        return self.name
