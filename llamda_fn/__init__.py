"""
Llamda is a Python library for transforming Python functions into LLM function calls.
"""

from .functions import LlamdaFunctions, llamda_callable
from .llamda import Llamda

__all__: list[str] = ["llamda_callable", "LlamdaFunctions", "Llamda"]
