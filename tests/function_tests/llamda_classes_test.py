import pytest
from llamda_fn.functions.llamda_callable import LlamdaCallable, LlamdaBase
from typing import Any


def test_llamda_callable_abstract_methods():
    with pytest.raises(TypeError):
        LlamdaCallable()


def test_llamda_base_abstract_methods():
    class TestLlamdaBase(LlamdaBase[Any]):
        def run(self, **kwargs: Any) -> Any:
            return kwargs

        def to_schema(self) -> dict[str, Any]:
            return {"type": "object"}

    test_base = TestLlamdaBase(
        name="test", description="test desc", call_func=lambda: None
    )
    assert test_base.to_tool_schema()["type"] == "function"
    assert test_base.to_tool_schema()["function"]["name"] == "test"
    assert test_base.to_tool_schema()["function"].get("description") == "test desc"
