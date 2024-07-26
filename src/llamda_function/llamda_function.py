from typing import Any, Callable, Dict, Generic, TypeVar

from pydantic import BaseModel

from .process_fields import create_model_from_fields, process_fields

R = TypeVar("R")
A = TypeVar("A", bound=list[Any])


class LlamdaFunction(BaseModel, Generic[R]):
    """
    A function that can be called from the Llamda API.
    """

    name: str
    description: str
    parameter_model: type[BaseModel]
    call_func: Callable[..., Any]

    @classmethod
    def create(
        cls,
        name: str,
        fields: Dict[str, Any],
        description: str,
        call_func: Callable[..., Any],
    ) -> "LlamdaFunction[R]":
        """
        Create a new LlamdaFunction from a function.
        """
        processed_fields = process_fields(fields)
        parameter_model = create_model_from_fields(
            f"{name}Parameters", processed_fields
        )

        return cls(
            name=name,
            description=description,
            parameter_model=parameter_model,
            call_func=call_func,
        )

    def run(self, **kwargs: Any) -> Any:
        """
        Run the LlamdaFunction with the given parameters.
        """
        # Validate inputs using the parameter_model
        validated_params = self.parameter_model(**kwargs)
        # Call the actual function with validated parameters
        return self.call_func(**validated_params.model_dump())

    def to_schema(self) -> dict[str, Any]:
        """
        Get the JSON schema for the LlamdaFunction.
        """
        schema: dict[str, Any] = {}
        schema["title"] = self.name
        schema["description"] = self.description
        model_schema = self.parameter_model.model_json_schema()
        schema["required"] = model_schema.get("required", [])
        schema["properties"] = self._process_properties(model_schema["properties"])

        # Add definitions if present
        if "$defs" in model_schema:
            schema["$defs"] = model_schema["$defs"]

        return schema

    def _process_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """
        Process properties to include nested Pydantic model schemas.
        """
        processed_properties = {}
        for prop_name, prop_schema in properties.items():
            if "$ref" in prop_schema:
                # This is a reference to a nested Pydantic model
                ref_name = prop_schema["$ref"].split("/")[-1]
                nested_schema = self.parameter_model.model_json_schema()["$defs"][
                    ref_name
                ]
                processed_properties[prop_name] = {
                    "type": "object",
                    "properties": self._process_properties(nested_schema["properties"]),
                    "required": nested_schema.get("required", []),
                }
            else:
                processed_properties[prop_name] = prop_schema
        return processed_properties
