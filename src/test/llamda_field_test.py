"""Test for the `llamda_fn.llamda_function.process_fields` module."""

from typing import Dict, List, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, Field

from src.llamda_function.process_fields import process_field, process_fields


def test_process_field_basic_types():
    """Test processing of basic types."""
    for type_ in (str, int, float, bool):
        processed_type, llamda_field = process_field(type_, {})
        assert processed_type == type_
        assert isinstance(llamda_field, dict)
        assert not llamda_field  # Should be empty for basic types


def test_process_field_list():
    """Test processing of list types."""
    processed_type, llamda_field = process_field(List[int], {})
    assert get_origin(processed_type) is list
    assert get_args(processed_type)[0] == int
    assert "nested" in llamda_field
    assert "items" in llamda_field["nested"]


def test_process_field_dict():
    """Test processing of dict types."""
    processed_type, llamda_field = process_field(Dict[str, float], {})
    assert get_origin(processed_type) is dict
    assert get_args(processed_type) == (str, float)
    assert "nested" in llamda_field
    assert "values" in llamda_field["nested"]


def test_process_field_optional():
    """Test processing of optional types."""
    processed_type, llamda_field = process_field(Union[str, None], {})
    assert processed_type == str
    assert isinstance(llamda_field, dict)
    assert not llamda_field  # Should be empty for basic types


def test_process_field_union():
    """Test processing of union types."""
    processed_type, llamda_field = process_field(Union[int, str], {})
    assert get_origin(processed_type) is Union
    assert set(get_args(processed_type)) == {int, str}
    assert "nested" in llamda_field


def test_process_field_nested_model():
    """Test processing of nested model types."""

    class NestedModel(BaseModel):
        """Nested model for testing."""

        nested_field: str

    processed_type, llamda_field = process_field(NestedModel, {})
    assert processed_type == NestedModel
    assert "nested" in llamda_field
    assert "nested_field" in llamda_field["nested"]


def test_process_fields():
    class TestModel(BaseModel):
        string_field: str = Field(description="A string field")
        int_field: int
        list_field: List[str]
        nested_field: Dict[str, int]

    processed = process_fields(TestModel.model_fields)

    # Check string_field
    assert processed["string_field"][0] == str
    assert isinstance(processed["string_field"][1], dict)
    assert processed["string_field"][1].get("description") == "A string field"

    # Check int_field
    assert processed["int_field"][0] == int
    assert isinstance(processed["int_field"][1], dict)

    # Check list_field
    assert get_origin(processed["list_field"][0]) is list
    assert get_args(processed["list_field"][0])[0] == str
    assert isinstance(processed["list_field"][1], dict)
    assert "nested" in processed["list_field"][1]
    assert "items" in processed["list_field"][1]["nested"]

    # Check nested_field
    assert get_origin(processed["nested_field"][0]) is dict
    assert get_args(processed["nested_field"][0]) == (str, int)
    assert isinstance(processed["nested_field"][1], dict)
    assert "nested" in processed["nested_field"][1]
    assert "values" in processed["nested_field"][1]["nested"]


def test_process_fields_with_pydantic_field():
    class TestModel(BaseModel):
        field_with_default: int = Field(default=42, description="Field with default")

    processed = process_fields(TestModel.model_fields)

    assert processed["field_with_default"][0] == int
    assert isinstance(processed["field_with_default"][1], dict)
    assert processed["field_with_default"][1].get("default") == 42
    assert processed["field_with_default"][1].get("description") == "Field with default"


if __name__ == "__main__":
    pytest.main([__file__])
