"""
Llamda is a Python library for transforming Python functions into LLM function calls.
"""

from .llamda import Llamda
from .llamda_function import __all__ as llamda_fn


__all__: list[str] = [
    "Llamda",
    "llamda_fn",
]
