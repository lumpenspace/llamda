# Welcome to Llamda's Documentation

Llamda is a Python library that provides easy-to-use decorators for making your functions and Pydantic models compatible with Large Language Models (LLMs).

## Installation

You can install Llamda using pip:

```bash
pip install llamda
```

## Quick Start

Here's a simple example of how to use Llamda:

```python
from llamda_fn import Llamda
from typing import List

llamda = Llamda()

@llamda.llamdafy()
def greet(name: str, times: int = 1) -> str:
    """Greet a person multiple times."""
    return f"Hello, {name}! " * times

# Use the function
result = greet("Alice", 3)
print(result)  # Output: Hello, Alice! Hello, Alice! Hello, Alice!

# Get the JSON schema
schema = llamda.tools["greet"].to_schema()
print(schema)
```

```{toctree}
:maxdepth: 2
:caption: Contents:

usage
api_reference
```
