import json
from typing import Any, List, Optional, Sequence
from openai import OpenAI
from openai.types.chat.chat_completion_function_message_param import (
    ChatCompletionFunctionMessageParam,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from .llamda_function.llamda_functions import LlamdaFunctions


class Llamda:
    """
    Llamda manages LLM interactions and delegates function execution to LlamdaFunctions.
    """

    def __init__(
        self,
        llamda_functions: Optional[LlamdaFunctions] = None,
        openai_api: Optional[OpenAI] = None,
        retry: int = 3,
        model: str = "gpt-4-0613",
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ) -> None:
        self.llamda_functions: LlamdaFunctions = llamda_functions or LlamdaFunctions()
        self.retry: int = retry
        self.model: str = model

        if openai_api:
            self.openai_api: OpenAI = openai_api
        else:
            openai_args = {}
            if openai_api_key:
                openai_args["api_key"] = openai_api_key
            if openai_base_url:
                openai_args["base_url"] = openai_base_url
            self.openai_api = OpenAI(**openai_args)

    def llamdafy(self, *args: Any, **kwargs: Any) -> Any:
        """
        Decorator to make a function available to the Llamda instance.

        Proxies [LlamdaFunctions.llamdafy]
        """
        return self.llamda_functions.llamdafy(*args, **kwargs)

    def run(
        self,
        messages: List[ChatCompletionMessageParam],
        function_names: Optional[List[str]] = None,
    ) -> str:
        """
        Run the OpenAI API with the prepared data.
        """
        tools: Sequence[ChatCompletionToolParam] = self.llamda_functions.prepare_tools(
            function_names
        )

        for _ in range(self.retry + 1):
            response: ChatCompletion = self.openai_api.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
            )
            assistant_message: ChatCompletionMessage = response.choices[0].message
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    function_result = self.llamda_functions.execute_function(
                        tool_call.function.name, tool_call.function.arguments
                    )
                    messages.append(
                        ChatCompletionFunctionMessageParam(
                            role="function",
                            content=json.dumps(function_result),
                            name=tool_call.function.name,
                        )
                    )
            else:
                return assistant_message.content or "No response"

        return "Max retries reached. Unable to complete the request."
