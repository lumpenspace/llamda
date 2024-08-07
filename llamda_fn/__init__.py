"""
Llamda is a Python library for transforming Python functions into LLM function calls.
"""

from .functions import (
    llamda_classes,
    LlamdaFunctions,
)

from .llamda import Llamda


__all__: list[str] = ["Llamda", "LlamdaFunctions", "llamda_classes"]
