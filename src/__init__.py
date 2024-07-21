"""
LLamda is a Python package that allows you turn any function into an
LLM tool with a simple decorator.
"""

from llamda.llamda_function import LlamdaFunction
from llamda.llamdafy import llamdafy

__all__: list[str] = ["LlamdaFunction", "llamdafy"]
