"""
testd=s
"""

import email
import json
from typing import Any

import pytest
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import BaseModel, Field

from llamda_fn.functions import LlamdaCallable, LlamdaFunctions
from llamda_fn.llms.api_types import LlToolCall, OaiToolParam, ToolResponse


@pytest.fixture
def decorated_functions() -> LlamdaFunctions:
    llamda = LlamdaFunctions()

    @llamda.llamdafy()
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @llamda.llamdafy(name="subtract", description="Subtract two numbers")
    def sub_numbers(a: int, b: int) -> int:
        return a - b

    class UserModel(BaseModel):
        name: str = Field(...)
        age: int = Field(...)
        email: str | None = Field(default=None)

    @llamda.llamdafy()
    def create_user(name: str, age: int, email: str | None = None) -> dict[str, Any]:
        """Create a user from a Pydantic model."""
        return UserModel(name=name, age=age, email=email).model_dump()

    return llamda


class TestLlamdaFunctions:
    def test_basic_function(self, decorated_functions: LlamdaFunctions):
        func: LlamdaCallable[Any] = decorated_functions.tools["add_numbers"]
        schema = func.to_tool_schema()
        assert schema.get("function").get("description") == "Add two numbers."
        assert schema.get("function").get("parameters") == {
            "properties": {
                "a": {"title": "A", "type": "integer"},
                "b": {"title": "B", "type": "integer"},
            },
            "required": ["a", "b"],
            "type": "object",
        }
        assert func.run(a=1, b=2) == 3

    def test_decorator_info(self, decorated_functions: LlamdaFunctions):
        func: LlamdaCallable[Any] = decorated_functions.tools["subtract"]
        schema: OaiToolParam = func.to_tool_schema()

        assert schema.get("function").get("name") == "subtract"
        assert schema.get("function").get("description") == "Subtract two numbers"

        assert func.run(a=5, b=3) == 2

    def test_pydantic_model(self, decorated_functions: LlamdaFunctions):
        func: LlamdaCallable[Any] = decorated_functions.tools["create_user"]
        schema_fn: OaiToolParam = func.to_tool_schema()

        assert schema_fn.get("function").get("name") == "create_user"
        assert (
            schema_fn.get("function").get("description")
            == "Create a user from a Pydantic model."
        )

        assert schema_fn.get("function") is not None

        user_properties: FunctionDefinition = schema_fn.get("function")

        assert user_properties == {
            "name": "create_user",
            "description": "Create a user from a Pydantic model.",
            "parameters": {
                "required": ["name", "age"],
                "type": "object",
                "properties": {
                    "age": {
                        "title": "Age",
                        "type": "integer",
                    },
                    "name": {
                        "title": "Name",
                        "type": "string",
                    },
                    "email": {
                        "anyOf": [
                            {
                                "type": "string",
                            },
                            {
                                "type": "null",
                            },
                        ],
                        "default": None,
                        "title": "Email",
                    },
                },
            },
        }

        result = func.run(name="Alice", age=30, email="lala")
        assert result == {"name": "Alice", "age": 30, "email": "lala"}

        # Add another test case for when email is not provided
        result_no_email = func.run(name="Bob", age=25)
        assert result_no_email == {"name": "Bob", "age": 25, "email": None}

    def test_get_tools(self, decorated_functions: LlamdaFunctions) -> None:
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

    def test_execute_function(self, decorated_functions: LlamdaFunctions):
        add_call = LlToolCall(
            id="1",
            name="add_numbers",
            arguments='{"a": 5, "b": 3}',
        )
        result: ToolResponse = decorated_functions.execute_function(add_call)
        assert result == ToolResponse(
            id="1",
            result="8",
        )

    def test_execute_function_errors(self, decorated_functions: LlamdaFunctions):
        non_existent_call = LlToolCall(
            id="3", name="nonexistent", arguments="""{"lol": "no}"""
        )
        result: ToolResponse = decorated_functions.execute_function(non_existent_call)
        content: Any = json.loads(result.result)
        assert "error" in content
        assert (
            "nonexistent" in content["error"]
            and "not found" in content["error"].lower()
        )

    def test_general_exception(self, decorated_functions: LlamdaFunctions):
        @decorated_functions.llamdafy()
        def failing_function(x: int) -> int:
            raise ValueError("This is a test error")

        error_call = LlToolCall(
            id="5",
            name="failing_function",
            arguments='{"x": 1}',
        )
        result: ToolResponse = decorated_functions.execute_function(error_call)
        content: dict[str, Any] = json.loads(result.result)
        assert "error" in content
        assert "test error" in content["error"].lower()

    def test_llamda_functions_dict_like_behavior(
        self, decorated_functions: LlamdaFunctions
    ):
        assert "add_numbers" in decorated_functions
        assert len(decorated_functions) == 3
        assert set(decorated_functions) == {"add_numbers", "subtract", "create_user"}

        add_func = decorated_functions["add_numbers"].to_tool_schema().get("function")
        assert add_func.get("name") == "add_numbers"
        assert add_func.get("description") == "Add two numbers."


if __name__ == "__main__":
    pytest.main([__file__])
