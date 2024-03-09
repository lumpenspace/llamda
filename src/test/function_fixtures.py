from typing import Optional
from llamda import llamda

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

@llamda(main="Retrieve the weather information for a given location and date.", location="The location for which to retrieve the weather information.", date="The date for which to retrieve the weather information.")
def get_weather_decorated(location: str, date: Optional[str]) -> str:
    return get_weather(location, date)
  
if __name__=='__main__':
    schema = get_weather_decorated.to_schema()
    print(schema)