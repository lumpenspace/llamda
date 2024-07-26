from ast import List
from typing import Any, Dict, Optional, Union, get_args, get_origin
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic.fields import FieldInfo
from pydantic_core import SchemaError

JsonDict = Dict[str, Any]


def process_field(
    field_type: Any, field_info: Union[JsonDict, FieldInfo]
) -> tuple[Any, JsonDict]:
    """
    Process a field type and info, using Pydantic's model_json_schema for schema generation.
    """
    try:
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            # Handle nested Pydantic models
            nested_schema = field_type.model_json_schema()
            field_schema = {
                "type": "object",
                "properties": nested_schema.get("properties", {}),
                "required": nested_schema.get("required", []),
            }
        else:
            # Create a temporary model with the field
            if isinstance(field_info, FieldInfo):
                temp_field = field_info
            else:
                temp_field = Field(**field_info)

            TempModel = create_model("TempModel", field=(field_type, temp_field))

            # Get the JSON schema for the entire model
            full_schema = TempModel.model_json_schema()

            # Extract the schema for our specific field
            field_schema = full_schema["properties"]["field"]

        # Handle Optional types
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            if type(None) in args:
                # This is an Optional type
                non_none_type = next(arg for arg in args if arg is not type(None))
                if non_none_type is float:
                    field_schema["type"] = "number"
                elif non_none_type is int:
                    field_schema["type"] = "integer"
                elif non_none_type is str:
                    field_schema["type"] = "string"
                elif isinstance(non_none_type, type) and issubclass(
                    non_none_type, BaseModel
                ):
                    field_schema["type"] = "object"
                field_schema["nullable"] = True

        # Ensure 'type' is always set
        if "type" not in field_schema:
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                field_schema["type"] = "object"
            elif field_type is int:
                field_schema["type"] = "integer"
            elif field_type is float:
                field_schema["type"] = "number"
            elif field_type is str:
                field_schema["type"] = "string"
            elif field_type is bool:
                field_schema["type"] = "boolean"
            elif field_type is list or field_type is List:
                field_schema["type"] = "array"
            elif field_type is dict or field_type is Dict:
                field_schema["type"] = "object"
            else:
                field_schema["type"] = "any"

        # Merge field_info with the generated schema
        if isinstance(field_info, dict):
            for key, value in field_info.items():
                if key not in field_schema or field_schema[key] is None:
                    field_schema[key] = value

        return field_type, field_schema
    except (SchemaError, ValidationError) as e:
        print(f"Error processing field: {e}")
        return Any, {"type": "any", "error": str(e)}

        # If schema generation fails, return Any type a


def process_fields(fields: Dict[str, Any]) -> Dict[str, tuple[Any, JsonDict]]:
    """
    Process all fields in a model, using Pydantic for complex types.
    """
    processed_fields = {}
    for field_name, field_value in fields.items():
        if isinstance(field_value, FieldInfo):
            field_type = field_value.annotation
            field_info = field_value
        elif isinstance(field_value, tuple):
            field_type, field_info = field_value
        else:
            raise ValueError(
                f"Unexpected field value type for {field_name}: {type(field_value)}"
            )

        processed_type, processed_info = process_field(field_type, field_info)

        # Ensure 'type' is set for nested Pydantic models
        if isinstance(processed_type, type) and issubclass(processed_type, BaseModel):
            processed_info["type"] = "object"

        processed_fields[field_name] = (processed_type, processed_info)

    return processed_fields


def create_model_from_fields(
    name: str, fields: Dict[str, tuple[Any, JsonDict]]
) -> type[BaseModel]:
    """
    Create a Pydantic model from a dictionary of fields.
    """
    model_fields = {}
    for field_name, (field_type, field_info) in fields.items():
        default = field_info.get("default", ...)
        if default is None:
            field_type = Optional[field_type]
            field_info["default"] = None
        model_fields[field_name] = (field_type, Field(**field_info))

    return create_model(name, **model_fields)
