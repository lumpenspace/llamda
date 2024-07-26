"""
Test introspection tools for the Llamda library.
"""

from typing import Optional, Union

from old.old.introspection_tools import (
    get_docstring_descriptions,
    get_type_str,
    is_argument_required,
    strip_meta_from_docstring,
)


def test_strip_meta_from_docstring() -> None:
    """
    Test the strip_meta_from_docstring function.
    """
    docstring = """
        Retrieve the weather information for a given location and date.
        Returns the weather forecast as a string.
    
        @param location: The location for which to retrieve the weather information.
        @param date: The date for which to retrieve the weather information.
        @return: The weather forecast as a string.
    """
    expected_output = (
        "Retrieve the weather information for a given location and date.\n"
        "Returns the weather forecast as a string."
    )
    assert strip_meta_from_docstring(docstring) == expected_output


def test_get_docstring_descriptions() -> None:
    """
    Test the get_docstring_descriptions function.
    """
    docstring = """
        Retrieve the weather information for a given location and date.
        
        @param location: The location for which to retrieve the weather information.
        @param date: The date for which to retrieve the weather information.
        @return: The weather forecast as a string.
    """
    expected_output = {
        "location": "The location for which to retrieve the weather information.",
        "date": "The date for which to retrieve the weather information.",
    }
    assert get_docstring_descriptions(docstring) == expected_output


def test_get_type_str() -> None:
    """
    Test the get_type_str function.
    """
    assert get_type_str(int) == "int"
    assert get_type_str(str) == "str"
    assert get_type_str(Optional[int]) == "int"
    assert get_type_str(Union[int, str]) == ["int", "str"]


def test_is_argument_required() -> None:
    """
    Test the is_argument_required function.
    """

    def func(
        a: int, b: Optional[str], c: Union[int, None], d: int = 0
    ) -> list[int | str | None]:
        return [a, b, c, d]

    assert is_argument_required(func, "a")
    assert not is_argument_required(func, "b")
    assert not is_argument_required(func, "c")
    assert not is_argument_required(func, "d")
