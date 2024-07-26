from typing import Any, Dict, List, Union, TypedDict, TypeVar, get_args, get_origin
from pydantic import BaseModel, PydanticUndefinedAnnotation, TypeAdapter
from pydantic.fields import FieldInfo


R = TypeVar("R")


class LlamdaField(TypedDict, total=False):
    """
    Recursive field in a Llamda function.
    """

    description: str
    default: Any
    nested: Dict[str, "LlamdaField"]
    error: str


def process_field(field_type: Any, field_info: LlamdaField) -> tuple[Any, LlamdaField]:
    """
    Process a field type and info, reducing it to a nested object of type:
    (type, LlamdaField)
    """
    llamda_field: LlamdaField = field_info.copy()

    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is Union:
        # Handle Union types, including Optional
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return process_field(non_none_args[0], llamda_field)
        else:
            # For complex unions, use TypeAdapter to generate schema
            return _handle_complex_type(field_type, llamda_field)

    if origin is list:
        if len(args) == 1:
            item_type, item_llamda_field = process_field(args[0], {})
            llamda_field["nested"] = {"items": item_llamda_field}
            return (List[item_type], llamda_field)
        else:
            return _handle_complex_type(field_type, llamda_field)

    if origin is dict:
        if len(args) == 2 and args[0] is str:
            value_type, value_llamda_field = process_field(args[1], {})
            llamda_field["nested"] = {"values": value_llamda_field}
            return (Dict[str, value_type], llamda_field)
        else:
            return _handle_complex_type(field_type, llamda_field)

    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        # Convert Pydantic models to nested object schemas
        return _handle_complex_type(field_type, llamda_field)

    # Handle basic types
    if field_type in (str, int, float, bool):
        return (field_type, llamda_field)

    # Use TypeAdapter for other types
    return _handle_complex_type(field_type, llamda_field)


def _handle_complex_type(
    field_type: Any, llamda_field: LlamdaField
) -> tuple[Any, LlamdaField]:
    """
    Handle complex types using Pydantic's TypeAdapter.
    """
    try:
        type_adapter = TypeAdapter(field_type)
        json_schema = type_adapter.json_schema()
        llamda_field["nested"] = _convert_json_schema_to_llamda_field(json_schema)
        return (field_type, llamda_field)
    except PydanticUndefinedAnnotation:
        # If the type is not fully defined, return as is
        return (field_type, llamda_field)
    except Exception as e:
        # If TypeAdapter fails for any other reason, default to Any
        llamda_field["error"] = str(e)
        return (Any, llamda_field)


def _convert_json_schema_to_llamda_field(
    json_schema: Dict[str, Any]
) -> Dict[str, LlamdaField]:
    """
    Convert a JSON schema to a nested LlamdaField structure.
    """
    result: Dict[str, LlamdaField] = {}
    if "properties" in json_schema:
        for prop, prop_schema in json_schema["properties"].items():
            result[prop] = _json_schema_to_llamda_field(prop_schema)
    return result


def _json_schema_to_llamda_field(schema: Dict[str, Any]) -> LlamdaField:
    """
    Convert a single property JSON schema to a LlamdaField.
    """
    llamda_field: LlamdaField = {}
    if "description" in schema:
        llamda_field["description"] = schema["description"]
    if "default" in schema:
        llamda_field["default"] = schema["default"]
    if "properties" in schema:
        llamda_field["nested"] = _convert_json_schema_to_llamda_field(schema)
    return llamda_field


def process_fields(fields: Dict[str, Any]) -> Dict[str, tuple[Any, LlamdaField]]:
    """
    Process all fields in a model.
    """
    processed_fields: Dict[str, tuple[Any, LlamdaField]] = {}
    for field_name, field_value in fields.items():
        if isinstance(field_value, FieldInfo):
            field_type = field_value.annotation
            field_info: LlamdaField = {}
            if field_value.description:
                field_info["description"] = field_value.description
            if field_value.default is not None:
                field_info["default"] = field_value.default
        elif isinstance(field_value, tuple):
            field_type, field_info = field_value
        else:
            raise ValueError(
                f"Unexpected field value type for {field_name}: {type(field_value)}"
            )

        processed_type, processed_info = process_field(field_type, field_info)
        processed_fields[field_name] = (processed_type, processed_info)
    return processed_fields
