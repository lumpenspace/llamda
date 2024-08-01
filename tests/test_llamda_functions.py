from typing import Sequence
from llamda_fn.functions.llamda_functions import LlamdaFunctions
from llamda_fn.llms.oai_api_types import OaiToolSpec


def test_spec_property():
    lf = LlamdaFunctions()

    @lf.llamdafy()
    def func1():
        pass

    @lf.llamdafy()
    def func2():
        pass

    spec: Sequence[OaiToolSpec] = lf.spec
    assert len(spec) == 2
    assert spec[0]["function"]["name"] in ["func1", "func2"]
    assert spec[1]["function"]["name"] in ["func1", "func2"]


def test_get_spec_method():
    lf = LlamdaFunctions()

    @lf.llamdafy()
    def func1():
        pass

    @lf.llamdafy()
    def func2():
        pass

    @lf.llamdafy()
    def func3():
        pass

    all_spec: Sequence[OaiToolSpec] = lf.get_spec()
    assert len(all_spec) == 3

    filtered_spec: Sequence[OaiToolSpec] = lf.get_spec(["func1", "func3"])
    assert len(filtered_spec) == 2
    assert filtered_spec[0]["function"]["name"] in ["func1", "func3"]
    assert filtered_spec[1]["function"]["name"] in ["func1", "func3"]

    non_existent_spec: Sequence[OaiToolSpec] = lf.get_spec(["non_existent"])
    assert len(non_existent_spec) == 0
