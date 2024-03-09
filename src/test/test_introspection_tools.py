import pytest
from llamda.introspection_tools import (
    strip_meta_from_docstring,
    get_docstring_descriptions,
    get_type_str,
    is_argument_required
)
import inspect
from typing import Optional, Union

def test_strip_meta_from_docstring():
    docstring = """
        Retrieve the weather information for a given location and date.
        Returns the weather forecast as a string.
    
        @param location: The location for which to retrieve the weather information.
        @param date: The date for which to retrieve the weather information.
        @return: The weather forecast as a string.
    """
    expected_output = "Retrieve the weather information for a given location and date.\nReturns the weather forecast as a string."
    assert strip_meta_from_docstring(docstring) == expected_output

def test_get_docstring_descriptions():
    docstring = """
        Retrieve the weather information for a given location and date.
        
        @param location: The location for which to retrieve the weather information.
        @param date: The date for which to retrieve the weather information.
        @return: The weather forecast as a string.
    """
    expected_output = {
        "location": "The location for which to retrieve the weather information.",
        "date": "The date for which to retrieve the weather information."
    }
    assert get_docstring_descriptions(docstring) == expected_output

def test_get_type_str():
    assert get_type_str(int) == "int"
    assert get_type_str(str) == "str"
    assert get_type_str(Optional[int]) == "int"
    assert get_type_str(Union[int, str]) == ["int", "str"]

def test_is_argument_required():
    def func(a: int, b: Optional[str], c: Union[int, None], d: int = 0):
        pass
    
    signature = inspect.signature(func)
    assert is_argument_required(signature.parameters["a"]) == True
    assert is_argument_required(signature.parameters["b"]) == False
    assert is_argument_required(signature.parameters["c"]) == False
    assert is_argument_required(signature.parameters["d"]) == False