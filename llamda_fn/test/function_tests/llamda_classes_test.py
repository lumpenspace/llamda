import pytest
from typing import List, Dict, Optional, Any, Type
from pydantic import BaseModel
from llamda_fn.functions.llamda_classes import LlamdaPydantic

# ... (previous TestLlamdaFunction class remains unchanged)


class TestLlamdaPydantic:
    @pytest.fixture
    def user_model(self) -> Type[BaseModel]:
        class UserModel(BaseModel):
            name: str
            age: int
            email: Optional[str] = None

        return UserModel

    def test_llamda_pydantic_creation(self, user_model: Type[BaseModel]):
        def create_user(user: user_model) -> Dict[str, Any]:
            return user.model_dump()

        func = LlamdaPydantic.create(
            name="CreateUser",
            model=user_model,
            description="Create a user from a Pydantic model",
            call_func=create_user,
        )

        assert func.name == "CreateUser"
        assert func.description == "Create a user from a Pydantic model"

        assert func.run(name="Alice", age=30, email="alice@example.com") == {
            "name": "Alice",
            "age": 30,
            "email": "alice@example.com",
        }
        assert func.run(name="Bob", age=25) == {"name": "Bob", "age": 25, "email": None}

    def test_llamda_pydantic_schema(self):
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

    def test_llamda_pydantic_tool_schema(self):
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

        tool_schema = func.to_tool_schema()

        assert tool_schema["type"] == "function"
        assert tool_schema["function"]["name"] == "CreateProduct"
        assert (
            tool_schema["function"]["description"]
            == "Create a product from a Pydantic model"
        )
        assert "parameters" in tool_schema["function"]

        parameters = tool_schema["function"]["parameters"]
        assert parameters["type"] == "object"
        assert "properties" in parameters

        properties = parameters["properties"]
        assert properties["name"]["type"] == "string"
        assert properties["price"]["type"] == "number"
        assert properties["tags"]["type"] == "array"
        assert properties["tags"]["items"]["type"] == "string"

        assert "required" in parameters
        assert set(parameters["required"]) == {"name", "price"}


if __name__ == "__main__":
    pytest.main([__file__])
