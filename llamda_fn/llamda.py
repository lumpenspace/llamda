from typing import Any, Callable, List, Optional, Sequence

from openai import OpenAI

from llamda_fn.utils.api import (
    ChatMessage,
    ToolCall,
    ToolParam,
    ChatCompletion,
    ToolMessage,
    Message,
)

from llamda_fn.functions import LlamdaFunctions
from llamda_fn.utils import LlamdaValidator
from llamda_fn.exchanges import Exchange


class Llamda:
    """
    Llamda class to create, decorate, and run Llamda functions.
    """

    def __init__(
        self,
        api: Optional[OpenAI] = None,
        api_config: Optional[dict[str, Any]] = None,
        llm_name: str = "gpt-4-turbo-preview",
        system_message: Optional[str] = None,
    ):
        validator = LlamdaValidator(
            api=api, api_config=api_config or {}, llm_name=llm_name
        )
        if not validator.api:
            raise ValueError("API is not set.")

        self.api: OpenAI = validator.api
        self.llm_name: str = validator.llm_name
        self.functions: LlamdaFunctions = LlamdaFunctions()
        self.exchange = Exchange(system_message=system_message)

    def llamdafy(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        """
        Decorator method to create a Llamda function.
        """
        return self.functions.llamdafy(*args, **kwargs)

    @property
    def tools(self) -> Sequence[ToolParam]:
        """
        Get the tools available to the Llamda instance.
        """
        return self.functions.get()

    def run(
        self,
        tool_names: Optional[List[str]] = None,
        exchange: Optional[Exchange] = None,
        llm_name: Optional[str] = None,
    ) -> ChatMessage | ToolMessage:
        """
        Run the OpenAI API with the prepared data.
        """
        current_exchange = exchange or self.exchange
        request: dict[str, Any] = {
            "model": llm_name or self.llm_name,
            "messages": current_exchange,
        }

        tools: Sequence[ToolParam] = self.functions.get(tool_names)
        if tools:
            request["tools"] = tools

        response: ChatCompletion = self.api.chat.completions.create(**request)
        message: Message = response.choices[0].message

        if message.tool_calls:
            self._handle_tool_calls(message.tool_calls, current_exchange)
        else:
            current_exchange.append(message.content or "No response", role="assistant")

        return current_exchange[-1]

    def _handle_tool_calls(
        self, tool_calls: List[ToolCall], exchange: Exchange
    ) -> None:
        """
        Handle tool calls and update the exchange.
        """
        for tool_call in tool_calls:
            result = self.functions.execute_function(tool_call=tool_call)
            exchange.append(result)
        self.run(exchange=exchange)

    def send_message(self, message: str) -> ChatMessage | ToolMessage:
        """
        Send a message and get a response.
        """
        self.exchange.append(message, role="user")
        return self.run()


__all__: list[str] = ["Llamda"]
