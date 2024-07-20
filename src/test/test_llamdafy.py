import pytest
from typing import Optional
from llamda import llamdafy
from .function_fixtures import get_weather, get_weather_decorated


@llamdafy
def get_weather_with_param_annotations(location: str, date: Optional[str]) -> str:
    """
    Retrieve the weather information for a given location and date.
    Returns the weather forecast as a string.

    @param location: The location for which to retrieve the weather information.
    @param date: The date for which to retrieve the weather information.
    @return: The weather forecast as a string.
    """
    return get_weather(location, date)


def test_llamda_decorator() -> None:
    """
    Test the @llamdafy decorator.
    """
    # Test the decorated function
    result = get_weather_decorated(location="London", date="2023-06-01")

    assert result.success
    assert result == "Rainy on 2023-06-01"

    # Test the schema generation
    schema = get_weather_decorated.to_schema()
    assert schema["name"] == "get_weather_decorated"
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
    result = get_weather_with_param_annotations(
        location="Philadelphia", date="2023-06-01"
    )
    assert result.success
    assert result.result == "Sunny on 2023-06-01"

    # Test the schema generation
    schema = get_weather_with_param_annotations.to_schema()
    assert schema["name"] == "get_weather_with_param_annotations"
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
