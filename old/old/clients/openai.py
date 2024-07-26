from typing import List, Dict, Any
import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessage


def function_completion(
    llamda_functions: Llamdas,
    messages: List[ChatCompletionMessage],
    **kwargs: CompletionCreateParams
) -> Any:
    # Map the arguments to the OpenAI completion arguments
    openai_kwargs: dict[str, type[ChatCompletionMessageParam]] = kwargs

    # Add the LlamdaFunctions as tools to the completion request
    tools = llamda_functions.to_schema()

    while True:
        # Call the completion function with the mapped arguments
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            **openai_kwargs,
            messages=messages,
            tools=tools,
        )

        # Get the first message from the response
        message = response.choices[0].message

        # If there are no tool calls in the message, return the message
        if "tool_calls" not in message:
            return message

        # Execute the tool calls in the message
        execution_response = llamda_functions.execute(message)
        parameter_errors = [
            result.result.parameter_error
            for result in execution_response.results.values()
            if isinstance(result.result, ParameterError)
        ]

        # If there are parameter errors, prepare a message to retry the function calls
        if parameter_errors:
            error_message = "The following errors occurred during function execution:\n"
            error_message += "\n".join(str(error) for error in parameter_errors)
            error_message += "\nPlease provide the correct parameters and try again."

            # Prepare the messages for the next request
            messages = [
                {"role": "assistant", "content": message.content},
                {"role": "tool", "content": error_message},
            ]
        else:
            # If no parameter errors, prepare the messages with the execution results
            messages = [
                message,
                {"role": "tool", "content": execution_response.to_tool_response()},
            ]

        # Update the messages for the next request
        openai_kwargs["messages"] = messages
