from typing import Any, Callable, Dict, Generic, TypeVar
from pydantic import BaseModel, Field, create_model, ConfigDict

R = TypeVar("R")


class LlamdaBase(BaseModel, Generic[R]):
    name: str
    description: str
    call_func: Callable[..., Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def run(self, **kwargs: Any) -> Any:
        """Run the Llamda function with the given parameters."""
        raise NotImplementedError

    def to_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for the Llamda function."""
        raise NotImplementedError


class LlamdaFunction(LlamdaBase[R]):
    parameter_model: type[BaseModel]

    @classmethod
    def create(
        cls,
        name: str,
        fields: Dict[str, tuple[type, Any]],
        description: str,
        call_func: Callable[..., Any],
    ) -> "LlamdaFunction[R]":
        """Create a new LlamdaFunction from a function."""
        model_fields = {}
        for field_name, (field_type, field_default) in fields.items():
            if field_default is ...:
                model_fields[field_name] = (field_type, Field(...))
            else:
                model_fields[field_name] = (field_type, Field(default=field_default))

        parameter_model = create_model(f"{name}Parameters", **model_fields)

        return cls(
            name=name,
            description=description,
            parameter_model=parameter_model,
            call_func=call_func,
        )

    def run(self, **kwargs: Any) -> Any:
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
    model: type[BaseModel]

    @classmethod
    def create(
        cls,
        name: str,
        model: type[BaseModel],
        description: str,
        call_func: Callable[..., Any],
    ) -> "LlamdaPydantic[R]":
        """Create a new LlamdaPydantic from a Pydantic model."""
        return cls(
            name=name,
            description=description,
            model=model,
            call_func=call_func,
        )

    def run(self, **kwargs: Any) -> Any:
        """Run the LlamdaPydantic with the given parameters."""
        validated_params = self.model(**kwargs)
        return self.call_func(validated_params)

    def to_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for the LlamdaPydantic."""
        schema = self.model.model_json_schema()
        schema["title"] = self.name
        schema["description"] = self.description
        return schema
