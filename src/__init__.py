"""
LLamda is a Python package that allows you turn any function into an
LLM tool with a simple decorator.
"""

from .llamda import LlamdaFunction, llamdafy

__all__: list[str] = ["LlamdaFunction", "llamdafy"]
