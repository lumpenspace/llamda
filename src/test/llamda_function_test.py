import pytest
from typing import List, Dict, Any

from src.llamda_function.llamda_function import LlamdaFunction


def test_llamda_function_creation():
    """Test creation of a Llamda function."""

    def sample_func(a: int, b: str) -> str:
        return f"{b} repeated {a} times"

    fields = {
        "a": (int, {"description": "Number of repetitions"}),
        "b": (str, {"description": "String to repeat"}),
    }

    func = LlamdaFunction.create(
        name="SampleFunction",
        fields=fields,
        description="A sample function that repeats a string",
        call_func=sample_func,
    )

    assert func.name == "SampleFunction"
    assert func.description == "A sample function that repeats a string"

    result = func.run(a=3, b="hello")
    assert result == "hello repeated 3 times"


def test_llamda_function_schema():
    """Test creation of a Llamda function with complex types."""

    def complex_func(a: int, b: List[str], c: Dict[str, float]) -> Dict[str, Any]:
        return {"result": f"{a} items: {', '.join(b)}, values: {c}"}

    fields = {
        "a": (int, {"description": "Number of items"}),
        "b": (List[str], {"description": "List of strings"}),
        "c": (Dict[str, float], {"description": "Dictionary of float values"}),
    }

    func = LlamdaFunction.create(
        name="ComplexFunction",
        fields=fields,
        description="A complex function with various types",
        call_func=complex_func,
    )

    schema = func.to_schema()

    assert schema["title"] == "ComplexFunction"
    assert schema["description"] == "A complex function with various types"
    assert "properties" in schema

    properties = schema["properties"]
    assert properties["a"]["type"] == "integer"
    assert properties["b"]["type"] == "array"
    assert properties["b"]["items"]["type"] == "string"
    assert properties["c"]["type"] == "object"
    assert properties["c"]["additionalProperties"]["type"] == "number"


def test_llamda_function_with_default_values():
    """Test creation of a Llamda function with default values."""

    def default_func(a: int = 5, b: str = "default") -> str:
        return f"{b} repeated {a} times"

    fields = {
        "a": (int, {"description": "Number of repetitions", "default": 5}),
        "b": (str, {"description": "String to repeat", "default": "default"}),
    }

    func = LlamdaFunction.create(
        name="DefaultFunction",
        fields=fields,
        description="A function with default values",
        call_func=default_func,
    )

    schema = func.to_schema()

    assert "default" in schema["properties"]["a"]
    assert schema["properties"]["a"]["default"] == 5
    assert "default" in schema["properties"]["b"]
    assert schema["properties"]["b"]["default"] == "default"

    result = func.run()
    assert result == "default repeated 5 times"

    result = func.run(a=3, b="custom")
    assert result == "custom repeated 3 times"


if __name__ == "__main__":
    pytest.main([__file__])
