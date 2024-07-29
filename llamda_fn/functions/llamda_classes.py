from typing import Any, Callable, Dict, Generic, TypeVar, Type
from pydantic import BaseModel, Field, create_model, ConfigDict

from llamda_fn.llms.api_types import OaiToolParam

R = TypeVar("R")


class LlamdaCallable(Generic[R]):
    def run(self, **kwargs: Any) -> R:
        raise NotImplementedError

    def to_tool_schema(self) -> OaiToolParam:
        raise NotImplementedError

    @classmethod
    def create(
        cls,
        call_func: Callable[..., R],
        name: str = "",
        description: str = "",
        **kwargs: Any,
    ) -> "LlamdaCallable[R]":
        raise NotImplementedError


class LlamdaBase(BaseModel, LlamdaCallable[R]):
    """The base class for Llamda functions."""

    name: str
    description: str
    call_func: Callable[..., R]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for the Llamda function."""
        raise NotImplementedError

    def to_tool_schema(self) -> OaiToolParam:
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


class LlamdaFunction(LlamdaBase[R]):
    """A Llamda function that uses a simple function model as the input."""

    parameter_model: Type[BaseModel]

    @classmethod
    def create(
        cls,
        call_func: Callable[..., R],
        name: str = "",
        description: str = "",
        fields: Dict[str, tuple[type, Any]] = {},
        **kwargs: Any,
    ) -> "LlamdaFunction[R]":
        """Create a new LlamdaFunction from a function."""
        model_fields = {}
        for field_name, (field_type, field_default) in fields.items():
            print(field_name, field_default, field_type)
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
        """Run the LlamdaFunction with the given parameters."""
        validated_params = self.parameter_model(**kwargs)
        return self.call_func(**validated_params.model_dump())

    def to_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for the LlamdaFunction."""
        schema = self.parameter_model.model_json_schema()
        schema["title"] = self.name
        schema["description"] = self.description
        return schema


class LlamdaPydantic(LlamdaBase[R]):
    """A Llamda function that uses a Pydantic model as the input."""

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
        """Create a new LlamdaPydantic from a Pydantic model."""

        return cls(
            name=name,
            description=description,
            call_func=call_func,
            model=model,
        )

    def run(self, **kwargs: Any) -> R:
        """Run the LlamdaPydantic with the given parameters."""
        validated_params = self.model(**kwargs)
        return self.call_func(validated_params)

    def to_schema(self) -> dict[str, Any]:
        """Get the JSON schema for the LlamdaPydantic."""
        schema: dict[str, Any] = self.model.model_json_schema(mode="serialization")
        schema["title"] = self.name
        schema["description"] = self.description
        return schema
