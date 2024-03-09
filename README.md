# llamda

`llamda` is a Python library that provides a decorator and utility functions for simplifying function calls / tool use in LLM.

Features include:

- A decorator for automatically converting python function definitions to LLM requests,
- A utility function to pass tools lists and execute the required ones in the response.

## Installation

You can install `llamda` using pip:

```bash
pip install llamda
```

## Decorator usage

The `llamda.make` decorator is used to convert a Python function to an LLM request.

The function requires:

- type annotations for the function parameters and return type,
- a docstring that describes the function.

The decorator is passed a named argument for each parameter of the function, the value of which is a description of the parameter.

It can be passed a `handle_exceptions` argument to specify whether to handle exceptions in the function, see the [Return Values](#return-values) section for more details.

```python
import llamda
from typing import Optional, Union

@llamda.make(
    location='The location for the weather',
    date='The date for the weather'
)
def get_weather(location: str, date: Optional[str]) -> str:
    """
    Retrieve the weather information for a given location and date.
    Returns the weather forecast as a string.
    """
    if location == 'London':
        return 'Rainy'
    elif location == 'Philadelphia':
        return 'Sunny'
    else:
        return 'Unknown'
```

The decorator will:

1. create a `get_schema` method that can be used to retrieve the schema for the function.
2. add a `success` attribute to return values, indicating whether the function was successful,
   and an `exception` attribute to return values, indicating the exception that was raised, if `handle_exceptions` is set to `True`.

### `get_weather.get_request()` results

<details>
<summary> Click to expand</summary>

```json
{
"name": "get_weather",
"description": "Retrieve the weather information for a given location and date.\nReturns the weather forecast as a string.",
"parameters": {
        "type": "object",
        "properties": {
        "location": {
            "type": "str",
            "description": "The location for the weather"
        },
        "date": {
            "type": "str",
            "description": "The date for the weather"
        }
        },
        "required": ["location"]
    }
}
```

</details>

### Return Values

The decorated function  will return the response of the base function, augmented with the following attributes:

- `success`: `True` if the function was successful, `False` otherwise.
- `exception`: The exception that was raised, if any.

If the `handle_exceptions` argument is set to `True`, the function will handle exceptions and return a response with `success` set to `False` and `exception` set to the exception that was raised.

Otherwise, the function will have a `success` value of `True` and raise the exception as normal.

## Usage with LLMs: the Llamdas class

`Llamdas` is a class that takes a set of decorated functions to handle their usage with LLMs.

It returns an instance of the `LlamdaFunctions` class, which will take care of transforming the functions to OpenAI tools and handling the responses.

```python

import llamda
import openai


llamda_functions = Llamdas(get_weather)


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
]

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=messages,
  tools=llamda_functions.to_openai_tools() # Pass the tools to the LLM
)

message = response['choices'][0]['message']

function_results = llamda_functions.handle_response(message)
if not function_results:
    print(message)
else:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[...messages, message, function_results.to_openai_message()]
    )
    messages.append(function_results.to_openai_message())
    print(response['choices'][0]['message'])
```
