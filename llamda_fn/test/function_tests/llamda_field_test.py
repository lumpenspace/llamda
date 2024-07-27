import pytest
from typing import Dict, List, Union, Optional, Any
from pydantic import BaseModel, Field

from llamda_fn.functions.process_fields import process_field, process_fields


class TestProcessField:
    @pytest.mark.parametrize(
        "field_type, expected_type, expected_schema",
        [
            (str, str, {"type": "string"}),
            (int, int, {"type": "integer"}),
            (float, float, {"type": "number"}),
            (bool, bool, {"type": "boolean"}),
            (List[str], List[str], {"type": "array", "items": {"type": "string"}}),
            (
                Dict[str, int],
                Dict[str, int],
                {"type": "object", "additionalProperties": {"type": "integer"}},
            ),
        ],
    )
    def test_basic_types(
        self, field_type: Any, expected_type: Any, expected_schema: Dict[str, Any]
    ):
        processed_type, field_schema = process_field(field_type, {})
        assert processed_type == expected_type
        assert field_schema == expected_schema

    def test_optional_types(self):
        for field_type in [Optional[str], Union[str, None]]:
            processed_type, field_schema = process_field(field_type, {})
            assert processed_type == Optional[str]
            assert field_schema == {"type": "string", "nullable": True}

    def test_nested_model(self):
        class NestedModel(BaseModel):
            nested_field: str

        processed_type, field_schema = process_field(NestedModel, {})
        assert processed_type == NestedModel
        assert field_schema["type"] == "object"
        assert "properties" in field_schema
        assert field_schema["properties"]["nested_field"]["type"] == "string"

    def test_field_info(self):
        field_info = Field(description="A test field", default="default")
        processed_type, field_schema = process_field(str, field_info)
        assert processed_type == str
        assert field_schema["type"] == "string"
        assert field_schema["description"] == "A test field"
        assert field_schema["default"] == "default"


class TestProcessFields:
    def test_process_fields(self):
        class TestModel(BaseModel):
            string_field: str = Field(description="A string field")
            int_field: int
            list_field: List[str]
            nested_field: Dict[str, int]
            optional_field: Optional[float]

        processed = process_fields(TestModel.model_fields)

        assert processed["string_field"][0] == str
        assert processed["string_field"][1]["type"] == "string"
        assert processed["string_field"][1]["description"] == "A string field"

        assert processed["int_field"][0] == int
        assert processed["int_field"][1]["type"] == "integer"

        assert processed["list_field"][0] == List[str]
        assert processed["list_field"][1]["type"] == "array"
        assert processed["list_field"][1]["items"]["type"] == "string"

        assert processed["nested_field"][0] == Dict[str, int]
        assert processed["nested_field"][1]["type"] == "object"
        assert processed["nested_field"][1]["additionalProperties"]["type"] == "integer"

        assert processed["optional_field"][0] == Optional[float]
        assert processed["optional_field"][1]["type"] == "number"
        assert processed["optional_field"][1]["nullable"] is True


if __name__ == "__main__":
    pytest.main([__file__])
