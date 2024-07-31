"""Testd for the LLamda base classes"""

from typing import Any, Self

import pytest

from llamda_fn.functions.llamda_callable import LlamdaBase, LlamdaCallable


def test_llamda_callable_abstract_methods():
    with pytest.raises(TypeError):
        LlamdaCallable()


def test_llamda_base_abstract_methods():
    class TestLlamdaBase(LlamdaBase[Any]):
        def run(self, **kwargs: Any) -> Any:
            return kwargs

        def to_schema(self) -> dict[str, Any]:
            return {
                "type": "object",
                "title": self.name,  # Add this line
                "description": self.description,  # Add this line
                "properties": {},  # Add this line
            }

        @classmethod
        def create(
            cls, call_func: Any, name: str = "", description: str = "", **kwargs: Any
        ):
            return cls(
                name=name, description=description, call_func=call_func, **kwargs
            )

    test_base = TestLlamdaBase(
        name="test", description="test desc", call_func=lambda: None
    )
    assert test_base.to_tool_schema()["type"] == "function"
    assert test_base.to_tool_schema()["function"]["name"] == "test"
    assert test_base.to_tool_schema()["function"].get("description") == "test desc"
