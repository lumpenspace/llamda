"""
Llamda class to create, decorate, and run Llamda functions.
"""

from typing import Any, Callable, List, Optional, Sequence
from pdb import set_trace as debugger
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

from llamda_fn.llms.api_types import (
    OaiToolParam,
    LLMessage,
    ToolCall,
    ToolResponse,
    OaiMessage,
    OaiChatCompletion,
)

from llamda_fn.functions import LlamdaFunctions
from llamda_fn.llms.llm_manager import LLManager
from llamda_fn.exchanges import Exchange


class Llamda:
    """
    Llamda class to create, decorate, and run Llamda functions.
    """

    def __init__(
        self,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ):
        self.api = LLManager(**kwargs)

        self.functions: LlamdaFunctions = LlamdaFunctions()
        self.exchange = Exchange(system_message=system_message)

    def fy(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        """
        Decorator method to create a Llamda function.
        """
        return self.functions.llamdafy(*args, **kwargs)

    @property
    def tools(self) -> Sequence[OaiToolParam]:
        """
        Get the tools available to the Llamda instance.
        """
        return self.functions.get()

    def run(
        self,
        tool_names: Optional[List[str]] = None,
        exchange: Optional[Exchange] = None,
        llm_name: Optional[str] = None,
    ) -> LLMessage:
        """
        Run the OpenAI API with the prepared data.
        """
        current_exchange: Exchange = exchange or self.exchange
        request: dict[str, Any] = {
            "model": llm_name or self.api.llm_name,
            "messages": current_exchange,
        }
        tools: Sequence[OaiToolParam] = self.functions.get(tool_names)
        if tools:
            request["tools"] = tools

        message: LLMessage = OaiChatCompletion(self.api.chat_completion(**request))

        current_exchange.append(message)
        if message.tool_calls:
            tool_calls: List[ToolCall] = message.tool_calls
            self._handle_tool_calls(tool_calls, current_exchange)

        return current_exchange[-1]

    def _handle_tool_calls(
        self, tool_calls: List[ToolCall], exchange: Exchange
    ) -> None:
        """
        Handle tool calls concurrently and update the exchange.
        """
        execution_results: list[ToolResponse] = []
        with ThreadPoolExecutor() as executor:
            futures: list[Future[ToolResponse]] = []
            for tool_call in tool_calls:
                futures.append(executor.submit(self._process_tool_call, tool_call))

            for future in as_completed(futures):
                result: ToolResponse = future.result()
                execution_results.append(result)
                exchange.append(result)

        self.run(exchange=exchange)

    def _process_tool_call(self, tool_call: ToolCall) -> ToolResponse:
        """
        Process a single tool call and return the result.
        """
        result: ToolResponse = self.functions.execute_function(tool_call=tool_call)
        return result

    def send_message(self, message: str) -> LLMessage:
        """
        Send a message and get a response.
        """
        self.exchange.ask(message)
        return self.run()


__all__: list[str] = ["Llamda"]
