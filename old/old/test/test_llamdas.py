import pytest
from llamda import Llamdas
from test.function_fixtures import get_weather_decorated

@pytest.fixture
def llamdas():
    return Llamdas(functions=[get_weather_decorated], handle_exceptions=False)

def test_llamdas_to_openai_tools(llamdas):
    openai_tools = llamdas.to_openai_tools()
    assert len(openai_tools) == 1
    for tool in openai_tools:
        assert tool["type"] == "function"
        assert tool["function"]["name"] in ["get_weather", "get_weather_decorated"]
        assert tool["function"]["description"] is not None
        assert len(tool["function"]["parameters"]["required"]) == 1
        assert tool["function"]["parameters"]["required"][0] == "location"

def test_llamdas_execute_success(llamdas):
    message = {
        "tool_calls": [
            {
                "id": "2",
                "function": {
                    "name": "get_weather_decorated",
                    "arguments": '{"location": "Philadelphia"}'
                }
            }
        ]
    }
    response = llamdas.execute(message)
    assert response.results["2"].function_name == "get_weather_decorated"
    assert response.results["2"].result == "Sunny"
    assert response.results["2"].result.success == True


def test_llamdas_execute_missing_required_param(llamdas):
    llamdas.handle_exceptions = True
    message = {
        "tool_calls": [
            {
                "id": "2",
                "function": {
                    "name": "get_weather_decorated",
                    "arguments": '{}'
                }
            }
        ]
    }
    response = llamdas.execute(message)
    assert response.results["2"].function_name == "get_weather_decorated"
    assert response.results["2"].result.success == False
    assert response.results["2"].result.parameter_error.name == "location"
    assert response.results["2"].result.parameter_error.description == "Parameter 'location' is required"

def test_llamdas_execute_invalid_function(llamdas):
    message = {
        "tool_calls": [
            {
                "id": "1",
                "function": {
                    "name": "invalid_function",
                    "arguments": '{}'
                }
            }
        ]
    }
    response = llamdas.execute(message)
    assert "1" not in response.results