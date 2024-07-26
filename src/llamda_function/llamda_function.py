from ast import List
from typing import Any, Callable, Dict, Generic, Self, Type, TypeVar
import typing
from venv import logger

from pydantic import BaseModel, Field, create_model, ConfigDict

from .process_fields import LlamdaField, process_fields

R = TypeVar("R")
A = TypeVar("A", bound=list[Any])


class LlamdaFunction(BaseModel, Generic[R]):
    """
    Base class for Llamda functions.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    def get_type(cls) -> Type[R]:
        """
        Get the return type of the Llamda function.
        """
        return typing.get_type_hints(cls.run)["return"]

    @classmethod
    def run(cls, *args: Any, **kwargs: Any) -> R:
        """
        Run the Llamda function.
        """
        instance: Self = cls(**kwargs)
        return instance._run(*args, **kwargs)

    def _run(self) -> R:
        # should be overridden by subclass
        raise NotImplementedError("run is not implemented")

    @classmethod
    def create(
        cls,
        name: str,
        fields: Dict[str, Any],
        description: str,
        call_func: Callable[..., R],
    ) -> Type[Self]:
        """Create a new Llamda function."""

        processed_fields: Dict[str, tuple[Any, LlamdaField]] = process_fields(fields)
        model_fields: Dict[str, Any] = {}

        for field_name, (field_type, field_info) in processed_fields.items():
            model_fields[field_name] = (field_type, Field(**field_info))

        logger.warning("Creating model %s with fields %s", name, model_fields)
        model: type[Self] = create_model(
            name, __base__=LlamdaFunction[R], **model_fields
        )
        model.__doc__ = description
        model.run = call_func
        return model

    @classmethod
    def to_schema(
        cls, by_alias: bool = True, ref_template: str = "#/$defs/{model}"
    ) -> Dict[str, Any]:
        """
        Convert the Llamda function to a JSON schema.
        """
        schema = cls.model_json_schema(by_alias=by_alias, ref_template=ref_template)
        schema["description"] = cls.__doc__

        # Convert 'type' field for each property
        for _, details in schema.get("properties", {}).items():
            if details.get("type") == "number":
                details["type"] = "float"
            elif details.get("type") == "integer":
                details["type"] = "int"
            elif details.get("type") == "array":
                if "items" in details and "type" in details["items"]:
                    if details["items"]["type"] == "number":
                        details["items"]["type"] = "float"
                    elif details["items"]["type"] == "integer":
                        details["items"]["type"] = "int"

        return schema
