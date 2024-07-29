import json
import pytest
from typing import Any, Dict, Optional
from pydantic import BaseModel

from llamda_fn.functions import LlamdaFunctions
from llamda_fn.llms.api import ToolCall


@pytest.fixture
def decorated_functions():
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

    return llamda


class TestLlamdaFunctions:
    def test_basic_function(self, decorated_functions):
        func = decorated_functions.tools["add_numbers"]
        assert func.name == "add_numbers"
        assert func.description == "Add two numbers."
        schema = func.to_schema()
        assert schema["properties"]["a"]["type"] == "integer"
        assert schema["properties"]["b"]["type"] == "integer"
        assert func.run(a=1, b=2) == 3

    def test_decorator_info(self, decorated_functions):
        func = decorated_functions.tools["subtract"]
        assert func.name == "subtract"
        assert func.description == "Subtract two numbers"
        assert func.run(a=5, b=3) == 2

    def test_pydantic_model(self, decorated_functions):
        func = decorated_functions.tools["create_user"]
        schema = func.to_schema()

        assert schema["title"] == "create_user"
        assert "properties" in schema

        user_properties = schema["properties"]
        assert user_properties["name"]["type"] == "string"
        assert user_properties["age"]["type"] == "integer"
        assert user_properties["email"]["anyOf"] == [
            {"type": "string"},
            {"type": "null"},
        ]

        result = func.run(name="Alice", age=30, email="alice@example.com")
        assert result == {"name": "Alice", "age": 30, "email": "alice@example.com"}

    def test_get_tools(self, decorated_functions):
        tools = decorated_functions.get()
        assert isinstance(tools, list)
        assert len(tools) == 3
        assert all(isinstance(tool, dict) for tool in tools)
        assert all("type" in tool and tool["type"] == "function" for tool in tools)
        assert all("function" in tool for tool in tools)

        specific_tools = decorated_functions.get(["add_numbers", "subtract"])
        assert len(specific_tools) == 2
        assert [tool["function"]["name"] for tool in specific_tools] == [
            "add_numbers",
            "subtract",
        ]

    def test_execute_function(self, decorated_functions):
        add_call = ToolCall(
            id="1",
            function={"name": "add_numbers", "arguments": '{"a": 5, "b": 3}'},
            type="function",
        )
        result = decorated_functions.execute_function(add_call)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "1"
        assert json.loads(result["content"]) == 8

    def test_execute_function_errors(self, decorated_functions):
        non_existent_call = ToolCall(
            id="3",
            function={"name": "non_existent", "arguments": "{}"},
            type="function",
        )
        result = decorated_functions.execute_function(non_existent_call)
        content = json.loads(result["content"])
        assert "error" in content
        assert (
            "non_existent" in content["error"]
            and "not found" in content["error"].lower()
        )

    def test_general_exception(self, decorated_functions):
        @decorated_functions.llamdafy()
        def failing_function(x: int) -> int:
            raise ValueError("This is a test error")

        error_call = ToolCall(
            id="5",
            function={"name": "failing_function", "arguments": '{"x": 1}'},
            type="function",
        )
        result = decorated_functions.execute_function(error_call)
        content = json.loads(result["content"])
        assert "error" in content
        assert "test error" in content["error"].lower()

    def test_llamda_functions_dict_like_behavior(self, decorated_functions):
        assert "add_numbers" in decorated_functions
        assert len(decorated_functions) == 3
        assert set(decorated_functions) == {"add_numbers", "subtract", "create_user"}

        add_func = decorated_functions["add_numbers"]
        assert add_func.name == "add_numbers"
        assert add_func.run(a=1, b=2) == 3


if __name__ == "__main__":
    pytest.main([__file__])
