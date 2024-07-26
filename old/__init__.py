"""
LLamda is a Python package that allows you turn any function into an
LLM tool with a simple decorator.
"""

from old.llamda.llamda_fn.llamda_function import LlamdaFunction
from llamda.llamdafy import llamdafy

__all__: list[str] = ["LlamdaFunction", "llamdafy"]
