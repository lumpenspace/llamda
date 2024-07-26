"""
Llamda is a Python library for transforming Python functions into LLM function calls.
"""

from .llamda_function.llamda_functions import LlamdaFunctions
from .llamda_function import __all__ as llamda_fn


__all__: list[str] = [
    "LlamdaFunctions",
    "llamda_fn",
]
