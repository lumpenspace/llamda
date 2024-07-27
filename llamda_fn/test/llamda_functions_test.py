from typing import Any, Dict, Optional
import json
import pytest
from pydantic import BaseModel

from llamda_fn.functions import (
    LlamdaFunctions,
    create_llamda_function,
)

llamda = LlamdaFunctions()


@llamda.llamdafy()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@llamda.llamdafy(name="subtract", description="Subtract two numbers")
def sub_numbers(a: int, b: int) -> int:
    return a - b


class UserModel(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


@llamda.llamdafy()
def create_user(user: UserModel) -> Dict[str, Any]:
    """Create a user from a Pydantic model."""
    return user.model_dump()


def test_basic_function():
    func = llamda.tools["add_numbers"]
    assert func.name == "add_numbers"
    assert func.description == "Add two numbers."
    schema = func.to_schema()
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["type"] == "integer"
    assert func.run(a=1, b=2) == 3


def test_decorator_info():
    func = llamda.tools["subtract"]
    assert func.name == "subtract"
    assert func.description == "Subtract two numbers"
    assert func.run(a=5, b=3) == 2


def test_pydantic_model():
    func = llamda.tools["create_user"]
    schema = func.to_schema()

    assert schema["title"] == "create_user"
    assert "properties" in schema

    user_properties = schema["properties"]
    assert user_properties["name"]["type"] == "string"
    assert user_properties["age"]["type"] == "integer"
    assert user_properties["email"]["anyOf"] == [{"type": "string"}, {"type": "null"}]

    result = func.run(name="Alice", age=30, email="alice@example.com")
    assert result == {"name": "Alice", "age": 30, "email": "alice@example.com"}


def test_prepare_tools():
    tools = llamda.prepare_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0
    assert all(isinstance(tool, dict) for tool in tools)
    assert all("type" in tool and tool["type"] == "function" for tool in tools)
    assert all("function" in tool for tool in tools)

    specific_tools = llamda.prepare_tools(["add_numbers", "subtract"])
    assert len(specific_tools) == 2
    assert [tool["function"]["name"] for tool in specific_tools] == [
        "add_numbers",
        "subtract",
    ]


def test_execute_function():
    result = llamda.execute_function("add_numbers", '{"a": 5, "b": 3}')
    assert result["role"] == "function"
    assert result["name"] == "add_numbers"
    assert json.loads(result["content"]) == 8

    result = llamda.execute_function("create_user", '{"name": "Bob", "age": 25}')
    assert result["role"] == "function"
    assert result["name"] == "create_user"

    content = json.loads(result["content"])
    assert content["name"] == "Bob"
    assert content["age"] == 25
    assert content["email"] is None

    result = llamda.execute_function("non_existent", "{}")
    content = json.loads(result["content"])
    assert content["error"] == "Error: Function not found"

    result = llamda.execute_function("add_numbers", '{"a": "invalid", "b": 3}')
    content = json.loads(result["content"])
    assert "error" in content
    assert content["error"].startswith("Error: Validation failed")
    assert "Input should be a valid integer" in content["error"]

    # Test for a general exception
    def failing_function(x: int) -> int:
        raise ValueError("This is a test error")

    llamda.llamdafy()(failing_function)
    result = llamda.execute_function("failing_function", '{"x": 1}')
    content = json.loads(result["content"])
    assert "error" in content
    assert content["error"] == "Error: This is a test error"


def test_create_llamda_function():
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    func = create_llamda_function(
        multiply, name="multiplication", description="Multiply two integers"
    )
    assert func.name == "multiplication"
    assert func.description == "Multiply two integers"
    schema = func.to_schema()
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["type"] == "integer"
    assert func.run(a=3, b=4) == 12


if __name__ == "__main__":
    pytest.main([__file__])
