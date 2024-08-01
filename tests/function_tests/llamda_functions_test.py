"""
This module defines the LlamdaFunctions class, which serves as a registry and manager
for Llamda functions. It provides functionality to register, execute, and manage
Llamda functions, including conversion of regular functions to Llamda functions
and generation of OpenAI-compatible tool specifications.
"""

from llamda_fn.functions import (
    LlamdaFunctions,
    LlamdaPydantic,
)
from llamda_fn.llms.ll_tool import LLToolCall, LLToolResponse
from pydantic import BaseModel


def test_llamdafy_decorator():
    lf = LlamdaFunctions()

    @lf.llamdafy(name="test_func", description="A test function")
    def test_func(a: int, b: int) -> int:
        return a + b

    assert "test_func" in lf.tools
    assert lf.tools["test_func"].name == "test_func"
    assert lf.tools["test_func"].description == "A test function"


def test_llamdafy_with_pydantic_model():
    lf = LlamdaFunctions()

    class TestModel(BaseModel):
        a: int
        b: int

    @lf.llamdafy()
    def test_func(model: TestModel) -> int:
        return model.a + model.b

    assert "test_func" in lf.tools
    assert isinstance(lf.tools["test_func"], LlamdaPydantic)


def test_execute_function():
    lf = LlamdaFunctions()

    @lf.llamdafy()
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    tool_call = LLToolCall(tool_call_id="1", name="add", arguments='{"a": 2, "b": 3}')
    response: LLToolResponse = lf.execute_function(tool_call)

    assert response.result == "5"


def test_spec_generation():
    lf = LlamdaFunctions()

    @lf.llamdafy(name="add", description="Add two numbers")
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    spec = lf.spec
    assert len(spec) == 1
    assert spec[0]["type"] == "function"
    assert spec[0]["function"]["name"] == "add"
    assert spec[0]["function"].get("description") == "Add two numbers"
