"""
This module defines the LlamdaPydantic class, which implements a Llamda function
using a Pydantic model for input validation. It extends the LlamdaBase class and
provides methods for creating, running, and generating schemas for Pydantic-based
Llamda functions.
"""

from llamda_fn.functions.llamda_pydantic import LlamdaPydantic
from pydantic import BaseModel


class TestModel(BaseModel):
    a: int
    b: int


def test_llamda_pydantic_create():
    def test_func(model: TestModel) -> int:
        return model.a + model.b

    llamda_func = LlamdaPydantic.create(
        call_func=test_func,
        name="test_func",
        description="A test function",
        model=TestModel,
    )

    assert llamda_func.name == "test_func"
    assert llamda_func.description == "A test function"
    assert llamda_func.model == TestModel


def test_llamda_pydantic_run():
    def test_func(model: TestModel) -> int:
        return model.a + model.b

    llamda_func = LlamdaPydantic.create(
        call_func=test_func,
        name="test_func",
        description="A test function",
        model=TestModel,
    )

    result = llamda_func.run(a=2, b=3)
    assert result == 5


def test_llamda_pydantic_to_schema():
    def test_func(model: TestModel) -> int:
        return model.a + model.b

    llamda_func = LlamdaPydantic.create(
        call_func=test_func,
        name="test_func",
        description="A test function",
        model=TestModel,
    )

    schema = llamda_func.to_schema()
    assert schema["title"] == "test_func"
    assert schema["description"] == "A test function"
    assert "a" in schema["properties"]
    assert "b" in schema["properties"]
