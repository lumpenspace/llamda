from typing import Dict, List, Union, Optional, get_args, get_origin
from pydantic import BaseModel, Field

from src.llamda_function.process_fields import process_field, process_fields


def test_process_field_optional():
    """Test processing of optional types."""
    processed_type, field_schema = process_field(Union[str, None], {})
    assert processed_type == Union[str, None]
    assert isinstance(field_schema, dict)
    assert field_schema["type"] == "string"
    assert field_schema["nullable"] is True

    # Test with Optional[str] as well
    processed_type, field_schema = process_field(Optional[str], {})
    assert processed_type == Optional[str]
    assert isinstance(field_schema, dict)
    assert field_schema["type"] == "string"
    assert field_schema["nullable"] is True


def test_process_field_nested_model():
    """Test processing of nested model types."""

    class NestedModel(BaseModel):
        """Nested model for testing."""

        nested_field: str

    processed_type, field_schema = process_field(NestedModel, {})
    assert processed_type == NestedModel
    assert field_schema["type"] == "object"
    assert "properties" in field_schema
    assert "nested_field" in field_schema["properties"]
    assert field_schema["properties"]["nested_field"]["type"] == "string"


def test_process_fields():
    class TestModel(BaseModel):
        string_field: str = Field(description="A string field")
        int_field: int
        list_field: List[str]
        nested_field: Dict[str, int]
        optional_field: Optional[float]

    processed = process_fields(TestModel.model_fields)

    # Check string_field
    assert processed["string_field"][0] == str
    assert isinstance(processed["string_field"][1], dict)
    assert processed["string_field"][1].get("description") == "A string field"
    assert processed["string_field"][1]["type"] == "string"

    # Check int_field
    assert processed["int_field"][0] == int
    assert isinstance(processed["int_field"][1], dict)
    assert processed["int_field"][1]["type"] == "integer"

    # Check list_field
    assert get_origin(processed["list_field"][0]) is list
    assert get_args(processed["list_field"][0])[0] == str
    assert isinstance(processed["list_field"][1], dict)
    assert processed["list_field"][1]["type"] == "array"
    assert processed["list_field"][1]["items"]["type"] == "string"

    # Check nested_field
    assert get_origin(processed["nested_field"][0]) is dict
    assert get_args(processed["nested_field"][0]) == (str, int)
    assert isinstance(processed["nested_field"][1], dict)
    assert processed["nested_field"][1]["type"] == "object"
    assert processed["nested_field"][1]["additionalProperties"]["type"] == "integer"

    # Check optional_field
    assert processed["optional_field"][0] == Optional[float]
    assert isinstance(processed["optional_field"][1], dict)
    assert processed["optional_field"][1]["type"] == "number"
    assert processed["optional_field"][1]["nullable"] is True
