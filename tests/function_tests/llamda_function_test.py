"""
This module defines the LlamdaFunction class, which implements a Llamda function
using a simple function model as input. It extends the LlamdaBase class and
provides methods for creating, running, and generating schemas for function-based
Llamda functions with built-in validation using Pydantic models.
"""

from typing import Any, Dict
from llamda_fn.functions.llamda_function import LlamdaFunction


def test_llamda_function_create():
    def test_func(a: int, b: int) -> int:
        return a + b

    llamda_func = LlamdaFunction.create(
        call_func=test_func,
        name="test_func",
        description="A test function",
        fields={"a": (int, ...), "b": (int, ...)},
    )

    assert llamda_func.name == "test_func"
    assert llamda_func.description == "A test function"
    assert isinstance(llamda_func.parameter_model, type)


def test_llamda_function_run():
    def test_func(a: int, b: int) -> int:
        return a + b

    llamda_func = LlamdaFunction.create(
        call_func=test_func,
        name="test_func",
        description="A test function",
        fields={"a": (int, ...), "b": (int, ...)},
    )

    result = llamda_func.run(a=2, b=3)
    assert result == 5


def test_llamda_function_to_schema():
    def test_func(a: int, b: int) -> int:
        return a + b

    llamda_func: LlamdaFunction[int] = LlamdaFunction.create(
        call_func=test_func,
        name="test_func",
        description="A test function",
        fields={"a": (int, ...), "b": (int, ...)},
    )

    schema: Dict[str, Any] = llamda_func.to_schema()
    assert schema["title"] == "test_func"
    assert schema["description"] == "A test function"
    assert "a" in schema["properties"]
    assert "b" in schema["properties"]
