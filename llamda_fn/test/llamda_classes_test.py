import pytest
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from llamda_fn.functions.llamda_classes import LlamdaFunction, LlamdaPydantic


def test_llamda_function_creation():
    def sample_func(a: int, b: str, c: str = "default") -> str:
        return f"{b} repeated {a} times, {c}"

    fields = {
        "a": (int, ...),
        "b": (str, ...),
        "c": (str, "default"),
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
    assert result == "hello repeated 3 times, default"

    result = func.run(a=2, b="world", c="custom")
    assert result == "world repeated 2 times, custom"


def test_llamda_function_schema():
    def complex_func(a: int, b: List[str], c: Dict[str, float]) -> Dict[str, Any]:
        return {"result": f"{a} items: {', '.join(b)}, values: {c}"}

    fields = {
        "a": (int, ...),
        "b": (List[str], ...),
        "c": (Dict[str, float], ...),
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


def test_llamda_pydantic_creation():
    class UserModel(BaseModel):
        name: str
        age: int
        email: Optional[str] = None

    def create_user(user: UserModel) -> Dict[str, Any]:
        return user.model_dump()

    func = LlamdaPydantic.create(
        name="CreateUser",
        model=UserModel,
        description="Create a user from a Pydantic model",
        call_func=create_user,
    )

    assert func.name == "CreateUser"
    assert func.description == "Create a user from a Pydantic model"

    result = func.run(name="Alice", age=30, email="alice@example.com")
    assert result == {"name": "Alice", "age": 30, "email": "alice@example.com"}

    result = func.run(name="Bob", age=25)
    assert result == {"name": "Bob", "age": 25, "email": None}


def test_llamda_pydantic_schema():
    class ProductModel(BaseModel):
        name: str
        price: float
        tags: List[str] = []

    def create_product(product: ProductModel) -> Dict[str, Any]:
        return product.model_dump()

    func = LlamdaPydantic.create(
        name="CreateProduct",
        model=ProductModel,
        description="Create a product from a Pydantic model",
        call_func=create_product,
    )

    schema = func.to_schema()

    assert schema["title"] == "CreateProduct"
    assert schema["description"] == "Create a product from a Pydantic model"
    assert "properties" in schema

    properties = schema["properties"]
    assert properties["name"]["type"] == "string"
    assert properties["price"]["type"] == "number"
    assert properties["tags"]["type"] == "array"
    assert properties["tags"]["items"]["type"] == "string"


if __name__ == "__main__":
    pytest.main([__file__])
