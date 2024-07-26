# pylint: disable=missing-docstring

"""
Test the @llamdafy decorator.
"""

from typing import Any, Optional

import pytest

from llamda import llamdafy

from .function_fixtures import get_weather


@llamdafy()
def llamdafied_get_weather(location: str, date: Optional[str]) -> str:
    """
    Retrieve the weather information for a given location and date.
    Returns the weather forecast as a string.

    @param location: The location for which to retrieve the weather information.
    @param date: The date for which to retrieve the weather information.
    @return: The weather forecast as a string.
    """

    return get_weather(location, date)


def test_schema_generation():

    schema = llamdafied_get_weather.to_json_schema()
    assert schema["name"] == "llamdafied_get_weather"
    assert (
        schema["description"]
        == "Retrieve the weather information for a given location and date.\nReturns the weather forecast as a string."
    )
    assert (
        schema["parameters"]["properties"]["location"]["description"]
        == "The location for which to retrieve the weather information."
    )


def test_llamda_decorator() -> None:
    """
    Test the @llamdafy decorator.
    """
    # Test the decorated function
    result = get_weather(location="London", date="2023-06-01")

    assert result
    assert result == "Rainy on 2023-06-01"

    # Test the schema generation
    schema: dict[str, Any] = llamdafied_get_weather.to_json_schema()
    assert schema["name"] == "llamdafied_get_weather"
    assert (
        schema["description"]
        == "Retrieve the weather information for a given location and date."
    )
    assert (
        schema["parameters"]["properties"]["location"]["description"]
        == "The location for which to retrieve the weather information."
    )
    assert (
        schema["parameters"]["properties"]["date"]["description"]
        == "The date for which to retrieve the weather information."
    )
    assert schema["parameters"]["required"] == [
        "location"
    ]  # "date" is optional, so it shouldn't be in the "required" list


def test_llamda_decorator_with_docstring_params():

    # Test the decorated function
    result = llamdafied_get_weather(location="Philadelphia", date="2023-06-01")
    assert result.success
    assert result.result == "Sunny on 2023-06-01"

    # Test the schema generation
    schema: dict[str, Any] = llamdafied_get_weather.to_json_schema()
    assert schema["name"] == "llamdafied_get_weather"
    assert (
        schema["description"]
        == "Retrieve the weather information for a given location and date.\nReturns the weather forecast as a string."
    )
    assert (
        schema["parameters"]["properties"]["location"]["description"]
        == "The location for which to retrieve the weather information."
    )
    assert (
        schema["parameters"]["properties"]["date"]["description"]
        == "The date for which to retrieve the weather information."
    )
    assert schema["parameters"]["required"] == ["location"]


def test_llamda_decorator_missing_description():
    with pytest.raises(
        ValueError,
        match="Description missing for function 'get_weather_missing_description'",
    ):

        @llamdafy()
        def get_weather_missing_description(location: str, date: Optional[str]) -> str:
            return get_weather(location, date)

        get_weather_missing_description.to_json_schema()


def test_llamda_decorator_missing_param_description():
    with pytest.raises(
        ValueError,
        match="Description missing for function 'get_weather_missing_param_description'",
    ):

        @llamdafy(date="The date for which to retrieve the weather information.")
        def get_weather_missing_param_description(
            location: str, date: Optional[str]
        ) -> str:
            return get_weather(location, date)

        get_weather_missing_param_description.to_json_schema()


def test_llamda_decorator_missing_type_annotation():
    with pytest.raises(
        ValueError,
        match="Type annotation missing for parameter 'location' in function 'get_weather_missing_type_annotation'",
    ):

        @llamdafy(
            main="Retrieve the weather information for a given location and date.",
            location="The location for which to retrieve the weather information.",
            date="The date for which to retrieve the weather information.",
        )
        def get_weather_missing_type_annotation(location, date: Optional[str]) -> str:
            return get_weather(location, date)

        get_weather_missing_type_annotation.to_json_schema()
