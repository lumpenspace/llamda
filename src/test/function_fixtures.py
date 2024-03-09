from typing import Optional
from llamda import llamdafy
import json
def get_weather(location: str, date: Optional[str]) -> str:
    """
    Retrieve the weather information for a given location and date.
    Returns the weather forecast as a string.
    """
    on_date = f" on {date}" if date else ""
    if location == 'London':
        return f'Rainy{on_date}'
    elif location == 'Philadelphia':
        return f'Sunny{on_date}'
    else:
        return f'Unknown{on_date}'
    
def get_weather_with_param_annotations(location: str, date: Optional[str]) -> str:
    """
    Retrieve the weather information for a given location and date.
    Returns the weather forecast as a string.

    @param location: The location for which to retrieve the weather information.
    @param date: The date for which to retrieve the weather information.
    @return: The weather forecast as a string.
    """
    return get_weather(location, date)

@llamdafy(main="Retrieve the weather information for a given location and date.", location="The location for which to retrieve the weather information.", date="The date for which to retrieve the weather information.")
def get_weather_decorated(location: str, date: Optional[str] = None) -> str:
    return get_weather(location, date)
  
if __name__=='__main__':
    message = {
        "tool_calls": [
            {
                "id": "2",
                "function": {
                    "name": "get_weather_decorated",
                    "arguments": '{ "location": "London", "date": "2023-06-01"}'
                }
            }
        ]
    }
    args = json.loads(message['tool_calls'][0]['function']['arguments'])
    result = get_weather_decorated(**args)
    print(result.success)
    print(result.result)
    print(result.exception)
